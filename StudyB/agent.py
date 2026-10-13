import numpy as np
import torch
import torch.nn as nn


def mlp(in_dim, out_dim, hidden, n_hidden, layernorm):
    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(d, hidden))
        if layernorm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, n_hidden, layernorm):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden, n_hidden, layernorm)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, n_hidden, low, high):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden, n_hidden, layernorm=False)
        self.register_buffer("scale", torch.as_tensor((high - low) / 2, dtype=torch.float32))
        self.register_buffer("center", torch.as_tensor((high + low) / 2, dtype=torch.float32))

    def forward(self, obs):
        return torch.tanh(self.net(obs)) * self.scale + self.center


def soft_update(net, target, tau):
    with torch.no_grad():
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.mul_(1 - tau).add_(p, alpha=tau)




class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)
        self.rew = np.zeros(size, np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.term = np.zeros(size, np.float32)
        self.size, self.ptr, self.n = size, 0, 0

    def add(self, o, a, r, o2, term):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.term[i] = o, a, r, o2, term
        self.ptr = (self.ptr + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.n, batch_size)
        t = torch.from_numpy
        return t(self.obs[idx]), t(self.act[idx]), t(self.rew[idx]), t(self.next_obs[idx]), t(self.term[idx])




def discounted_returns(rewards, gamma):
    G = np.zeros(len(rewards), dtype=np.float64)
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        G[t] = g
    return G


def rollout(actor, env, seed):
    o, _ = env.reset(seed=seed)
    states, actions, rewards = [], [], []
    while True:
        with torch.no_grad():
            a = actor(torch.as_tensor(o, dtype=torch.float32)).numpy()
        states.append(o)
        actions.append(a)
        o, r, terminated, truncated, _ = env.step(a)
        rewards.append(float(r))
        if terminated or truncated:
            return np.asarray(states), np.asarray(actions), rewards, terminated


def evaluate(actor, critics, env, seeds, gamma, tail_cut):
    returns, lengths, n_term = [], [], 0
    q1_all, qmin_all, g_all = [], [], []
    for seed in seeds:
        states, actions, rewards, terminated = rollout(actor, env, seed)
        returns.append(sum(rewards))
        lengths.append(len(rewards))
        n_term += int(terminated)
        valid = len(rewards) if terminated else max(0, len(rewards) - tail_cut)
        if valid == 0:
            continue
        s = torch.as_tensor(states[:valid], dtype=torch.float32)
        a = torch.as_tensor(actions[:valid], dtype=torch.float32)
        with torch.no_grad():
            q1 = critics[0](s, a).numpy().astype(np.float64)
            qmin = np.minimum(q1, critics[1](s, a).numpy()) if len(critics) == 2 else q1
        q1_all.append(q1)
        qmin_all.append(qmin)
        g_all.append(discounted_returns(rewards, gamma)[:valid])
    row = {"return_mean": float(np.mean(returns)), "len_mean": float(np.mean(lengths)),
           "frac_terminated": n_term / len(seeds)}
    if g_all:
        q1, qmin, G = np.concatenate(q1_all), np.concatenate(qmin_all), np.concatenate(g_all)
        row.update(bias_q1=float(np.mean(q1 - G)), bias_min=float(np.mean(qmin - G)),
                   q1_mean=float(q1.mean()), mc_mean=float(G.mean()))
    else:
        row.update(bias_q1=np.nan, bias_min=np.nan, q1_mean=np.nan, mc_mean=np.nan)
    return row
