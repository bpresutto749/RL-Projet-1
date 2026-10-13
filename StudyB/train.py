import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from agent import Actor, Critic, ReplayBuffer, evaluate, soft_update

EVAL_FIELDS = ["step", "return_mean", "len_mean", "frac_terminated", "bias_q1", "bias_min",
               "q1_mean", "mc_mean", "critic_loss"]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ddpg", "td3"], required=True)
    p.add_argument("--layernorm", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-id", default="LunarLander-v3")
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-hidden", type=int, default=2)
    p.add_argument("--exploration-noise", type=float, default=0.1)
    p.add_argument("--policy-noise", type=float, default=0.2)
    p.add_argument("--noise-clip", type=float, default=0.5)
    p.add_argument("--policy-delay", type=int, default=None)
    p.add_argument("--utd", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--mc-horizon-cut", type=int, default=300)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)
    if args.policy_delay is None:
        args.policy_delay = 1 if args.algo == "ddpg" else 2
    args.buffer_size = min(args.buffer_size, args.total_steps)
    return args


def config_tag(args):
    return f"{args.algo}-ln{args.layernorm}-L{args.n_hidden}-utd{args.utd}-d{args.policy_delay}"


def run_name(args):
    return f"{config_tag(args)}-s{args.seed}"


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    name = run_name(args)
    eval_path = os.path.join(args.out_dir, f"{name}.eval.csv")
    meta_path = os.path.join(args.out_dir, f"{name}.json")

    env = gym.make(args.env_id, continuous=True)
    eval_env = gym.make(args.env_id, continuous=True)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high
    act_scale = (high - low) / 2
    td3 = args.algo == "td3"
    eval_seeds = [1_000_000 + 100 * args.seed + i for i in range(args.eval_episodes)]

    actor = Actor(obs_dim, act_dim, args.hidden, args.n_hidden, low, high)
    actor_t = Actor(obs_dim, act_dim, args.hidden, args.n_hidden, low, high)
    actor_t.load_state_dict(actor.state_dict())
    critics = [Critic(obs_dim, act_dim, args.hidden, args.n_hidden, args.layernorm) for _ in range(2 if td3 else 1)]
    critics_t = [Critic(obs_dim, act_dim, args.hidden, args.n_hidden, args.layernorm) for _ in critics]
    for c, ct in zip(critics, critics_t):
        ct.load_state_dict(c.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    critic_opt = torch.optim.Adam([p for c in critics for p in c.parameters()], lr=args.lr_critic)
    low_t, high_t = torch.as_tensor(low), torch.as_tensor(high)
    scale_t = torch.as_tensor(act_scale, dtype=torch.float32)

    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
    n_updates = 0
    losses = []

    def update():
        nonlocal n_updates
        o, a, r, o2, term = buf.sample(args.batch_size, rng)
        with torch.no_grad():
            a2 = actor_t(o2)
            if td3:
                noise = (torch.randn_like(a2) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip) * scale_t
                a2 = torch.max(torch.min(a2 + noise, high_t), low_t)
            q_next = critics_t[0](o2, a2)
            if td3:
                q_next = torch.min(q_next, critics_t[1](o2, a2))
            y = r + args.gamma * (1.0 - term) * q_next
        loss = sum(F.mse_loss(c(o, a), y) for c in critics)
        critic_opt.zero_grad()
        loss.backward()
        critic_opt.step()
        n_updates += 1
        losses.append(loss.item())
        if n_updates % args.policy_delay == 0:
            actor_loss = -critics[0](o, actor(o)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            soft_update(actor, actor_t, args.tau)
            for c, ct in zip(critics, critics_t):
                soft_update(c, ct, args.tau)

    with open(eval_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=EVAL_FIELDS).writeheader()
    meta = vars(args) | {"run_name": name, "config_tag": config_tag(args), "finished": False,
                         "torch": torch.__version__, "gymnasium": gym.__version__, "numpy": np.__version__}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    t0 = time.time()
    o, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    for step in range(1, args.total_steps + 1):
        if step <= args.learning_starts:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(o, dtype=torch.float32)).numpy()
            a = np.clip(a + rng.normal(0.0, args.exploration_noise * act_scale, size=act_dim), low, high)
        o2, r, terminated, truncated, _ = env.step(a)
        buf.add(o, a, r, o2, float(terminated))
        o = o2
        if terminated or truncated:
            o, _ = env.reset()

        if step >= args.learning_starts:
            for _ in range(args.utd):
                update()

        if step % args.eval_every == 0:
            row = {"step": step} | evaluate(actor, critics, eval_env, eval_seeds, args.gamma, args.mc_horizon_cut)
            row["critic_loss"] = float(np.mean(losses)) if losses else np.nan
            losses.clear()
            with open(eval_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=EVAL_FIELDS).writerow(row)
            print(f"[{name}] step {step} | return {row['return_mean']:.1f} | bias {row['bias_q1']:.1f}"
                  f" | {step / (time.time() - t0):.0f} steps/s", flush=True)

    meta |= {"finished": True, "wallclock_min": round((time.time() - t0) / 60, 2)}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def is_finished(out_dir, name):
    path = os.path.join(out_dir, f"{name}.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return json.load(f).get("finished", False)


def run_all(argv):
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--algos", nargs="+", default=["ddpg", "td3"])
    p.add_argument("--ln", nargs="+", type=int, default=[0, 1])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    jobs = []
    for seed in range(args.seeds):
        for algo in args.algos:
            for ln in args.ln:
                cmd = ["--algo", algo, "--layernorm", str(ln), "--seed", str(seed), "--out-dir", out_dir] + extra
                name = run_name(parse_args(cmd))
                if not is_finished(out_dir, name):
                    jobs.append((name, cmd))
    print(f"{len(jobs)} runs, {args.workers} in parallel -> {out_dir}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    t0 = time.time()

    def run(job):
        name, cmd = job
        with open(os.path.join(log_dir, f"{name}.log"), "w") as log:
            rc = subprocess.run([sys.executable, os.path.abspath(__file__)] + cmd,
                                stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        return name, rc

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, fut in enumerate(as_completed([ex.submit(run, j) for j in jobs]), 1):
            name, rc = fut.result()
            if rc != 0:
                failed.append(name)
            print(f"[{k}/{len(jobs)}] {'OK' if rc == 0 else 'FAILED'} {name} ({(time.time() - t0) / 60:.0f} min)",
                  flush=True)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Failed: {failed or 'none'}")


if __name__ == "__main__":
    if "--all" in sys.argv:
        run_all([a for a in sys.argv[1:] if a != "--all"])
    else:
        main()
