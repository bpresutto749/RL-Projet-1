# -*- coding: utf-8 -*-

# main_vec.py does the same thing as main.py but for k agents at the same time where there is one agent per seed and k=5
#           we compare the critic's Q(s,a) with the Monte Carlo return G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... at the end of each episode
#
# Here the actor is a neural network that takes the state as input and outputs an action
# The critic is a neural network that takes the state and action as input and outputs a single number Q(s,a) which is the expected return from that pair (state, action)
# Each step is stored in a replay buffer and the agent learns from random samples of that buffer (i.i.d. data)
# Overestimation bias is defined as: bias = Q(s,a) - G_t. A positive bias means the critic is too optimistic
#
# networks are small and since we are using the gpu, one agent's computation is done very quickly and we must wait for python
# doing k agents in the same gpu call costs almost the same time as one agent
#
# instead of using k separate weight matrices (in, out), we can keep one tensor of shape (k, in, out), get the k-th slice
# of that tensor. this represents the weights of the agent k. 
#
# k = number of agents, n = number of samples (e.g. 256).
#
# example: python src/main_vec.py --algo td3 --use-layernorm --seeds 1 2 3 4 5

import argparse
import math
import os
import time

import gymnasium as gym  # LunarLander
import numpy as np
import torch# neural networks
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # writes the curves that we will read using plot.py


def parse_args():
    # all settings of an experiment that are read from CLI (e.g. --seeds 1 2 3). same meaning as main.py
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default="rl_exp")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="one agent per seed, trained in parallel")
    p.add_argument("--algo", type=str, default="td3", choices=["ddpg", "td3"])
    p.add_argument("--use-layernorm", action="store_true")
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--utd-ratio", type=int, default=1, help="gradient steps per environment step")
    p.add_argument("--env-id", type=str, default="LunarLanderContinuous-v3")
    p.add_argument("--total-timesteps", type=int, default=300000)
    p.add_argument("--buffer-size", type=int, default=int(1e5))
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--policy-noise", type=float, default=0.2, help="TD3 target policy smoothing noise")
    # TD3 uses a trick called "Target Policy Smoothing Regularization": when computing the learning target, it adds a little noise to the next action
    # this prevents the critic from exploiting narrow peaks of Q that are too high (overestimation). The noise is clipped to avoid moving too far away from the next action
    # Fujimoto et al. (2018)
    p.add_argument("--noise-clip", type=float, default=0.5)
    p.add_argument("--exploration-noise", type=float, default=0.1)
    p.add_argument("--learning-starts", type=int, default=10000)
    p.add_argument("--policy-frequency", type=int, default=2, help="critic-to-actor update ratio (TD3; DDPG uses 1)")
    p.add_argument("--eval-freq", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=5)
    # truncated episodes (cut at 1000 steps by the time limit) are not finished: the rewards that
    # would have come after step 1000 are unknown. G_t is too small for the steps close to the end
    # for a step at least 500 steps before the cut, the missing part is weighted by gamma^500 < 0.007,
    # which is negligible.
    p.add_argument("--mc-horizon", type=int, default=500, help="for truncated episodes, drop the last mc-horizon steps from the bias")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--threads", type=int, default=2)
    return p.parse_args()


# define folder where the curves are saved, this is just the folder from where the file is launched
RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs") # v2/runs


# batched networks class
class BatchedMLP(nn.Module):
    """K independent multi layer perceptrons. Input (K, N, in) -> (K, N, out). Same init as nn.Linear / nn.LayerNorm"""
    # dim is a list of layer sizes, for instance [10, 256, 256, 1] for a critic
    # there is a stack of k small neural networks (one per agent) that are computed together

    def __init__(self, K, dims, use_layernorm):
        super().__init__()
        self.use_layernorm = use_layernorm
        # w = weights, b = biases, g / beta = layernorm scale and shift. one entry per layer
        # and each entry has the values of all k agents, k is the first dimension
        self.W, self.b, self.g, self.beta = nn.ParameterList(), nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for i in range(len(dims) - 1):
            # a random start in [-1/sqrt(in), 1/sqrt(in)] is the same as pytorch usual nn.Linear layer
            bound = 1 / math.sqrt(dims[i])
            self.W.append(nn.Parameter(torch.empty(K, dims[i], dims[i + 1]).uniform_(-bound, bound)))
            self.b.append(nn.Parameter(torch.empty(K, 1, dims[i + 1]).uniform_(-bound, bound)))
            if use_layernorm and i < len(dims) - 2: # if we use layernorm or hidden layers only not on the output
                self.g.append(nn.Parameter(torch.ones(K, 1, dims[i + 1]))) # scale starts at 1
                self.beta.append(nn.Parameter(torch.zeros(K, 1, dims[i + 1])))  # shift starts at 0

    def forward(self, x, k=None):
        """k=None: x is (K, N, in). k=int: x is (N, in) and only agent k is used."""
        # k=None: we compute all k agents at once and used for training
        # k=3 : we compute only agent 3 which is used to measure the bias of one agent at the end of the episode
        # each hidden layer is a Linear -> (LayerNorm if enabled) -> ReLU.
        n_layers = len(self.W)
        for i in range(n_layers):
            if k is None:
                x = torch.baddbmm(self.b[i], x, self.W[i]) # X @ W + B for all
            else:
                x = torch.addmm(self.b[i][k], x, self.W[i][k])  # X @ W + B for agent k 
            if i < n_layers - 1:  # hidden layers only, the last layer has no normalisation and no relu
                if self.use_layernorm:
                    # a LayerNorm rescales the 256 values of a layer to mean 0 / variance 1 then applies a learned scale and shift
                    x = F.layer_norm(x, x.shape[-1:])
                    x = x * (self.g[i] if k is None else self.g[i][k]) + (self.beta[i] if k is None else self.beta[i][k])
                x = F.relu(x)
        return x


class Actor(nn.Module):
    # policy is input = state, output = action (i.e. 2 numbers between -1 and 1 for LunarLander)
    def __init__(self, K, obs_dim, act_dim, args, act_low, act_high):
        super().__init__()
        self.net = BatchedMLP(K, [obs_dim] + [args.hidden_size] * args.hidden_layers + [act_dim], args.use_layernorm)
        # action rescaling: tanh gives values in [-1, 1] and these two numbers map them to [low, high]
        # of the environment (for LunarLander low=-1, high=1, so scale=1 and bias=0)
        self.register_buffer("scale", torch.tensor((act_high - act_low) / 2, dtype=torch.float32))
        self.register_buffer("bias", torch.tensor((act_high + act_low) / 2, dtype=torch.float32))

    def forward(self, obs, k=None):
        # squash to [-1, 1]
        # rescale to the allowed action range
        return torch.tanh(self.net(obs, k)) * self.scale + self.bias


class Critic(nn.Module):
    # DDPG only uses one Q-network whereas TD3 uses two independent ones noted q1 and q2 and trusts the smaller value
    def __init__(self, K, obs_dim, act_dim, args):
        super().__init__()
        # list of layer sizes: [state size + action size, 256, 256, (256,) 1]
        # state: 8 numbers (position, velocity, angle, contact of legs...)
        # action: 2 numbers (main engine and side engines)
        # input: 8 + 2 = 10
        # 2 or 3 hidden layers defined by --hidden-layers of 256 neurons each
        # 1 output: Q(s, a)
        dims = [obs_dim + act_dim] + [args.hidden_size] * args.hidden_layers + [1]
        self.twin = args.algo == "td3"
        self.q1 = BatchedMLP(K, dims, args.use_layernorm)
        self.q2 = BatchedMLP(K, dims, args.use_layernorm) if self.twin else None

    def forward(self, obs, act, k=None):
        # observations is a batch of states, act is a batch of actions and they are both together as the input
        x = torch.cat([obs, act], -1)
        # TD3: two estimates
        # DDPG: one estimate
        return (self.q1(x, k), self.q2(x, k)) if self.twin else (self.q1(x, k),)

    def q_min(self, obs, act, k=None):
        # for TD3 we take min(q1, q2). it is the value TD3 itself uses in its learning targets
        return torch.min(*self(obs, act, k)) if self.twin else self(obs, act, k)[0]


def discounted_returns(rewards, gamma):
    # Monte Carlo return of every step of an episode: G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    # which is computed backwards because G_t = r_t + gamma * G_{t+1}.
    G, out = 0.0, np.zeros(len(rewards), dtype=np.float32)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


def episode_bias(critic, k, obs, acts, rews, truncated, args, device):
    """Mean of Q(s_t,a_t) - G_t over the episode of agent k (min of the twin critics for TD3)."""
    # mean of Q(s_t,a_t) - G_t over an episode. for any truncated episodes, MC return
    # misses the tail so the last mc_horizon steps are discarded from the bias estimate
    # we then compare, step by step, what the critic predicted (Q) with what the agent really got (G)
    G = discounted_returns(rews, args.gamma)

    # an episode stopped by the step 1000 time limit has not been finished so the last steps
    # are missing their future rewards. We ignore the last 500 steps of those episodes
    n = len(rews) - args.mc_horizon if truncated else len(rews)
    if n <= 0:
        return None
    # returns what the critic predicts for a list of (state, action) pairs (HERE: THE CRITIC OF AGENT k)
    with torch.no_grad(): # no learning here because we are just reading the predictions
        o = torch.as_tensor(np.array(obs[:n]), dtype=torch.float32, device=device)
        a = torch.as_tensor(np.array(acts[:n]), dtype=torch.float32, device=device)
        q = critic.q_min(o, a, k).squeeze(-1).cpu().numpy()

    # returns: average prediction, average real return, average bias (prediction - reality)
    return q.mean(), G[:n].mean(), (q - G[:n]).mean()


def evaluate(actor, critic, eval_envs, seeds, args, device):
    """Noise-free episodes for all agents at once; returns per-agent mean return and mean bias."""
    # we run the deterministic policy with no exploration noise
    # all k agents play their five test episodes together with one step at a time and a single actor call
    K = len(seeds)
    rets, biases = [[] for _ in range(K)], [[] for _ in range(K)] # results of each agent
    n_done = [0] * K # finished test episodes per agent
    obs = [eval_envs[k].reset(seed=10_000 + seeds[k] * 100)[0] for k in range(K)] # same starting positions at every evaluation
    traj = [([], [], []) for _ in range(K)] # (states, actions, rewards) of the current episode of each agent
    while min(n_done) < args.eval_episodes: # until every agent has finished its five episodes
        with torch.no_grad():
            a_all = actor(torch.as_tensor(np.array(obs), dtype=torch.float32, device=device).unsqueeze(1)).squeeze(1).cpu().numpy() # the actor's action with no noise
        for k in range(K):
            if n_done[k] >= args.eval_episodes:
                continue # this agent has finished and it waits for the others
            o2, r, term, trunc, _ = eval_envs[k].step(a_all[k])
            traj[k][0].append(obs[k]); traj[k][1].append(a_all[k]); traj[k][2].append(r)
            obs[k] = o2
            if term or trunc: # term = landed/crashed & trunc = time limit reached
                rets[k].append(sum(traj[k][2])) # score of the episode
                res = episode_bias(critic, k, *traj[k], trunc, args, device)
                if res is not None:
                    biases[k].append(res[2])
                n_done[k] += 1
                traj[k] = ([], [], [])
                if n_done[k] < args.eval_episodes:
                    obs[k] = eval_envs[k].reset(seed=10_000 + seeds[k] * 100 + n_done[k])[0]
    return [np.mean(r) for r in rets], [np.mean(b) if b else np.nan for b in biases]


def soft_update(target, source, tau):
    # update the target networks: move each target weight 0.5% (tau = 0.005) towards the real weight
    with torch.no_grad():
        torch._foreach_lerp_(list(target.parameters()), list(source.parameters()), tau)


# main
def main():
    # main execution
    args = parse_args() # get experiment settings
    seeds, K = args.seeds, len(args.seeds) #k number of agents trained together

    # torch configuration, if a CPU or a GPU is available we can use it
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    # random generators are seeded with the first seed and shared by the rest
    torch.manual_seed(seeds[0])
    np.random.seed(seeds[0])

    # each run has a folder in v2/runs where the name contains its settings (plot.py can then read data back from it).
    # there is one folder per agent
    t0 = int(time.time())
    writers = []
    for s in seeds:
        run_name = f"{args.env_id}__{args.algo}__{args.exp_name}__HL{args.hidden_layers}__LN{args.use_layernorm}__UTD{args.utd_ratio}__{s}__{t0}"
        w = SummaryWriter(os.path.join(RUNS_DIR, run_name)) # tensorboard
        hp = dict(vars(args), seed=s, batched_with=seeds, device=str(device))
        w.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in hp.items()))
        writers.append(w)

    # gym environments for experiment
    envs = [gym.wrappers.RecordEpisodeStatistics(gym.make(args.env_id)) for _ in seeds] # one training env per agent
    eval_envs = [gym.make(args.env_id) for _ in seeds]      # for the tests without noise
    for env, s in zip(envs, seeds):
        env.action_space.seed(s)   # makes the random actions of the first steps reproducible
    obs_dim = envs[0].observation_space.shape[0]   # state: 8 numbers (position, velocity, angle, contact of legs...)
    act_dim = envs[0].action_space.shape[0]        # action: 2 numbers (main engine and side engines)
    low, high = envs[0].action_space.low, envs[0].action_space.high

    # in both algorithms, we have two actors networks where one is used for learning and the other one follows the first one slowly
    # same applies for critic: one network will learn and the other one will follow slowly
    actor = Actor(K, obs_dim, act_dim, args, low, high).to(device)
    critic = Critic(K, obs_dim, act_dim, args).to(device)
    target_actor = Actor(K, obs_dim, act_dim, args, low, high).to(device)
    target_critic = Critic(K, obs_dim, act_dim, args).to(device)
    target_actor.load_state_dict(actor.state_dict()) # we can initiate both networks as copies
    target_critic.load_state_dict(critic.state_dict())
    fused = device.type == "cuda" # "fused" is a faster adam run on the gpu
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.learning_rate, fused=fused)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.learning_rate, fused=fused)
    low_t = torch.as_tensor(low, device=device) # action limits as gpu tensors for clamping
    high_t = torch.as_tensor(high, device=device)

    # create a replay buffer to store the transitions
    # memory of past transitions (state, action, reward, next state, done)
    # when it is full, the oldest transitions are overwritten (FIFO order circular buffer)
    # one buffer per agent, all stored in one GPU tensor. buf[k, i] is the transition number i of agent k
    # they are written as one row.
    width = 2 * obs_dim + act_dim + 2
    buf = torch.zeros(K, args.buffer_size, width, device=device)
    ptr, size = 0, 0 # where the next transition will be written / how many transitions are stored so far
    agent_idx = torch.arange(K, device=device).unsqueeze(1)   # [[0], [1], ..., [K-1]] are used to sample each agent's own buffer
    o0, o1 = 0, obs_dim
    a0, a1 = o1, o1 + act_dim
    r0 = a1
    n0, n1 = r0 + 1, r0 + 1 + obs_dim
    d0 = n1

    obs = np.array([env.reset(seed=s)[0] for env, s in zip(envs, seeds)], dtype=np.float32) # create the first state of the first episode
    trajs = [([], [], []) for _ in range(K)] # lists to track overestimation bias. We remember the current episode to compare Q with the real return at its end
    n_updates = 0 # number of critic updates done
    critic_losses = actor_losses = None
    start = time.time()

    # main loop where one iteration = one step in the game (FOR EVERY AGENT)
    for global_step in range(args.total_timesteps):
        # every eval_freq steps (i.e. 10,000) we test the agent without noise and store its score and bias
        if global_step % args.eval_freq == 0:
            ev_r, ev_b = evaluate(actor, critic, eval_envs, seeds, args, device)
            for k in range(K):
                writers[k].add_scalar("eval/episodic_return", ev_r[k], global_step)
                writers[k].add_scalar("eval/overestimation_bias", ev_b[k], global_step)
            sps = int(global_step / (time.time() - start + 1e-9))       # SPEED IN STEPS PER SECOND
            print(f"step {global_step}  eval return {np.round(ev_r, 0)}  SPS {sps}", flush=True)

        # CHOOSE ACTION
        # to begin (first 10000 steps) we play randomly because we have to fill the memory with various examples
        if global_step < args.learning_starts:
            actions = np.array([env.action_space.sample() for env in envs], dtype=np.float32) # one random action per agent
        # after that, we ask the actor what to do and add a little noise so the agent can keep trying new things
        else:
            with torch.no_grad(): # no learning here we just play
                a = actor(torch.as_tensor(obs, device=device).unsqueeze(1)).squeeze(1) # the actor's action
                a = a + torch.randn_like(a) * actor.scale * args.exploration_noise # add a small random change
                actions = a.clamp(low_t, high_t).cpu().numpy() # clamp between -1 and 1

        # play what we have in the simulator with each agent in its own environment
        rows = np.zeros((K, width), dtype=np.float32) # the k new transitions with one row per agent
        next_obs = np.empty_like(obs)
        for k, env in enumerate(envs):
            o2, r, term, trunc, info = env.step(actions[k])
            # we store transitions in RB
            # if we are at "done" then terminations only. if episode was cut by time limit, the game was not over and
            # so the learning target still adds the value of the next state.
            rows[k, o0:o1], rows[k, a0:a1], rows[k, r0] = obs[k], actions[k], r
            rows[k, n0:n1], rows[k, d0] = o2, float(term)
            # save trajectory for Monte-Carlo return
            trajs[k][0].append(obs[k]); trajs[k][1].append(actions[k]); trajs[k][2].append(r)
            # END OF AN EPISODE, we log the score and the overestimation bias
            if term or trunc:
                writers[k].add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writers[k].add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                # we get the Q-values vs true discounted MC returns on the behaviour trajectory
                res = episode_bias(critic, k, *trajs[k], trunc, args, device)
                if res is not None:
                    writers[k].add_scalar("charts/avg_q_value", res[0], global_step)
                    writers[k].add_scalar("charts/avg_mc_return", res[1], global_step)
                    writers[k].add_scalar("charts/overestimation_bias", res[2], global_step)
                trajs[k] = ([], [], []) # we now can start remembering the next episode
                o2, _ = env.reset() # we start a new episode directly
            next_obs[k] = o2
        buf[:, ptr] = torch.as_tensor(rows, device=device)  # write the k transitions in a singular buffer
        ptr = (ptr + 1) % args.buffer_size # go back to 0 after the end (circular)
        size = min(size + 1, args.buffer_size)
        obs = next_obs

        # LEARN (only happens after the first 10,000 random steps)
        if global_step > args.learning_starts:
            # UTD ratio (update-to-data) = number of learning updates per step played (1 by default, 2 in one ablation)
            # ablation = experiment where one element is changed to see how it affects the result
            for _ in range(args.utd_ratio):
                # 256 random transitions from each agent's own buffer to a batch of shape (k, 256, width)
                batch = buf[agent_idx, torch.randint(0, size, (K, args.batch_size), device=device)]
                b_obs, b_act, b_rew = batch[..., o0:o1], batch[..., a0:a1], batch[..., r0:r0 + 1]   # CUT EACH ROW BACK INTO ITS PARTS
                b_next, b_done = batch[..., n0:n1], batch[..., d0:d0 + 1]

                # compute learning target y = r + gamma * Q_target(next state, next action)
                # => the reward now + the discounted value of what comes next. if done=1 there is no next
                with torch.no_grad():
                    next_act = target_actor(b_next)
                    if args.algo == "td3":
                        # TD3 trick 1 (target policy smoothing): we add a small clipped noise to the next action
                        # so the critic cannot exploit a little high peak of Q
                        noise = (torch.randn_like(next_act) * args.policy_noise * actor.scale).clamp(
                            -args.noise_clip * actor.scale, args.noise_clip * actor.scale)
                        next_act = (next_act + noise).clamp(low_t, high_t)
                    # TD3 trick 2 (clipped double Q-learning): we take the smaller of the two critics' estimates
                    # this helps fight overestimation
                    # ddpg: a single critic and no protection against overestimation
                    y = b_rew + (1 - b_done) * args.gamma * target_critic.q_min(b_next, next_act)

                # train the critic. we attempt to bring Q(s, a) closer to the target y using mean squared error
                qs = critic(b_obs, b_act)
                # both critics learn the same target
                # there is one loss per agent (shape (K,)). we sum each agent k weights to get the gradient of agent k's loss
                critic_losses = sum(((q - y) ** 2).mean(dim=(1, 2)) for q in qs)
                critic_opt.zero_grad(set_to_none=True) # clear old gradients
                critic_losses.sum().backward() # compute new gradients
                critic_opt.step() # move the weights a little
                n_updates += 1

                # train the actor (TD3 trick 3: only every 2 critic updates) = delayed policy updates
                # and DDPG after every critic update.
                # CTA (critic-to-actor) ratio = policy_frequency (counted in gradient steps so it also holds for UTD > 1)
                if args.algo == "ddpg" or n_updates % args.policy_frequency == 0:
                    # actor wants actions with a high Q-value so we try to minimise -Q
                    # overestimation: the actor tries actions where the critic is too optimistic
                    actor_losses = -critic.q1(torch.cat([b_obs, actor(b_obs)], -1)).mean(dim=(1, 2))   # ONE LOSS PER AGENT
                    actor_opt.zero_grad(set_to_none=True)
                    actor_losses.sum().backward()
                    actor_opt.step()
                    # update the target networks: move each target weight 0.5% (tau = 0.005) towards the real weight
                    soft_update(target_actor, actor, args.tau)
                    soft_update(target_critic, critic, args.tau)

            # we log the losses every 1000 steps (tensorboard)
            if global_step % 1000 == 0:
                cl = critic_losses.detach().cpu().numpy()
                al = actor_losses.detach().cpu().numpy() if actor_losses is not None else None
                for k in range(K):
                    writers[k].add_scalar("losses/critic_loss", cl[k], global_step)
                    if al is not None:
                        writers[k].add_scalar("losses/actor_loss", al[k], global_step)

    # final test after the last training step
    ev_r, ev_b = evaluate(actor, critic, eval_envs, seeds, args, device)
    for k in range(K):
        writers[k].add_scalar("eval/episodic_return", ev_r[k], args.total_timesteps)
        writers[k].add_scalar("eval/overestimation_bias", ev_b[k], args.total_timesteps)
        writers[k].close()
    print(f"done in {time.time() - start:.0f} s, final eval return {np.round(ev_r, 0)}") # we print the total time and the scores


if __name__ == "__main__":
    main()
