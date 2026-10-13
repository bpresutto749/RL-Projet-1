# -*- coding: utf-8 -*-

# main.py:  allows one agent to train using DDPG or TD3 on LunarLanderContinuous-v3 and
#           returns how much the critic may overestimate the returns. We compare 
#           the critic's Q(s,a) with the Monte Carlo return G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... at the end of each episode.
#
# Here the actor is a neural network that takes the state as input and outputs an action.
# The critic is a neural network that takes the state and action as input and outputs a single number Q(s,a) which is the expected return from that pair (state, action)
# Each step is stored in a replay buffer and the agent learns from random samples of that buffer (i.i.d. data). 
# Overestimation bias is defined as: bias = Q(s,a) - G_t. A positive bias means the critic is too optimistic.
#
# Example:  python src/main.py --algo td3 --use-layernorm --seed 1
# (main_vec.py does the same thing, but trains several seeds at once on the GPU.)

import os
import random
import time
import argparse
import gymnasium as gym  # LunarLander
import numpy as np
import torch                 # neural networks
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter   # writes the curves that we will read using plot.py

def parse_args():
    # all settings of an experiment. can be read from the command line (e.g. --seed 3).
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="rl_exp", help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--algo", type=str, default="td3", choices=["ddpg", "td3"], help="algorithm to use")
    parser.add_argument("--use-layernorm", action="store_true", help="whether to use layer normalization")
    parser.add_argument("--hidden-layers", type=int, default=2, help="number of hidden layers (2 or 3)")
    parser.add_argument("--utd-ratio", type=int, default=1, help="update to data ratio")
    parser.add_argument("--env-id", type=str, default="LunarLanderContinuous-v3", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000, help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e5), help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient")
    parser.add_argument("--batch-size", type=int, default=256, help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--policy-noise", type=float, default=0.2, help="the scale of policy noise (TD3 only)")
    parser.add_argument("--exploration-noise", type=float, default=0.1, help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=10000, help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2, help="the frequency of training policy (delayed) (TD3 only)")
    # TD3 uses a trick called "Target Policy Smoothing Regularization": when computing the learning target, it adds a little noise to the next action. 
    # This prevents the critic from exploiting narrow peaks of Q that are too high (overestimation). The noise is clipped to avoid moving too far away from the next action.
    # Fujimoto et al. (2018)
    parser.add_argument("--noise-clip", type=float, default=0.5, help="noise clip parameter of the Target Policy Smoothing Regularization (TD3 only)") 
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="auto = cuda if available")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads per process (0 = torch default); use 2-3 when running several runs in parallel")
    parser.add_argument("--eval-freq", type=int, default=10000, help="evaluate the deterministic policy every eval-freq steps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="number of evaluation episodes")
    # Truncated episodes (cut at 1000 steps by the time limit) are not finished: the rewards that
    # would have come after step 1000 are unknown. G_t is too small for the steps close to the end.
    # For a step at least 500 steps before the cut, the missing part is weighted by gamma^500 < 0.007,
    # which is negligible.
    parser.add_argument("--mc-horizon", type=int, default=500, help="for truncated episodes, drop the last mc-horizon steps from the bias estimate (gamma^500 < 0.007)")
    args = parser.parse_args()
    return args

def make_env(env_id, seed, idx, capture_video, run_name):
    # returns a function that builds the simulator
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env) # RecordEpisodeStatistics adds the total reward and the length of each finished episode to the "info" dictionary
        env.action_space.seed(seed)   # makes the random actions of the first steps reproducible
        return env
    return thunk

class QNetwork(nn.Module):
    # one critic network is composed of input = state + action together. output is a single number Q(s, a).
    def __init__(self, env, hidden_layers, use_layernorm):
        super().__init__()
        # list of layer sizes: [state size + action size, 256, 256, (256,) 1]
        # state: 8 numbers (position, velocity, angle, contact of legs...)
        # action: 2 numbers (main engine and side engines)
        # input: 8 + 2 = 10
        # 2 or 3 hidden layers defined by --hidden-layers of 256 neurons each
        # 1 output: Q(s, a)
        dims = [np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)]
        for _ in range(hidden_layers):
            dims.append(256)
        dims.append(1)

        # each hidden layer is a Linear -> (LayerNorm if enabled) -> ReLU.
        # a LayerNorm rescales the 256 values of a layer to mean 0 / variance 1 then applies a learned scale and shift.
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1])) # last layer: no normalisation and outputs Q

        self.network = nn.Sequential(*layers)

    def forward(self, x, a):
        # x = batch of states, a = batch of actions -> batch of Q-values
        return self.network(torch.cat([x, a], 1))

class Critic(nn.Module):
    # DDPG only uses one Q-network whereas TD3 uses two independent ones noted q1 and q2 and trusts the smaller value
    def __init__(self, env, hidden_layers, use_layernorm, algo):
        super().__init__()
        self.algo = algo
        self.q1 = QNetwork(env, hidden_layers, use_layernorm)
        if self.algo == "td3":
            self.q2 = QNetwork(env, hidden_layers, use_layernorm)

    def forward(self, x, a):
        q1 = self.q1(x, a)
        if self.algo == "td3":
            q2 = self.q2(x, a)
            return q1, q2 # TD3: two estimates
        return q1# DDPG: one estimate

class Actor(nn.Module):
    # policy is input = state, output = action (i.e. 2 numbers between -1 and 1 for LunarLander)
    def __init__(self, env, hidden_layers, use_layernorm):
        super().__init__()
        dims = [np.array(env.single_observation_space.shape).prod()]
        for _ in range(hidden_layers):
            dims.append(256)

        # same hidden layers as the critic: Linear -> (LayerNorm) -> ReLU
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(dims[-1], np.prod(env.single_action_space.shape))# output layer has one value per action

        # action rescaling: tanh gives values in [-1, 1] and these two numbers map them to [low, high]
        # of the environment (for LunarLander low=-1, high=1, so scale=1 and bias=0).
        self.register_buffer("action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = self.network(x)
        x = torch.tanh(self.fc_mu(x))# squash to [-1, 1]
        return x * self.action_scale + self.action_bias # rescale to the allowed action range

class ReplayBuffer:
    # memory of past transitions (state, action, reward, next state, done)
    # when it is full, the oldest transitions are overwritten (FIFO order circular buffer)
    def __init__(self, max_size, obs_shape, action_shape, device):
        self.max_size = max_size
        self.ptr = 0 # where the next transition will be written
        self.size = 0 # how many transitions are stored so far

        self.obs = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((max_size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32) # 1 if the episode really ended (e.g. crash or landing)

        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size # go back to 0 after the end (circular)
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # we pick batch_size random transitions and send them to the GPU/CPU as tensors.
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[ind]).to(self.device),
            torch.FloatTensor(self.actions[ind]).to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.next_obs[ind]).to(self.device),
            torch.FloatTensor(self.dones[ind]).to(self.device)
        )

def discounted_returns(rewards, gamma):
    # Monte Carlo return of every step of an episode: G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    # which is computed backwards because G_t = r_t + gamma * G_{t+1}.
    G = 0.0
    returns = np.zeros(len(rewards), dtype=np.float32)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

def critic_q(critic, algo, obs, actions, device):
    # returns what the critic predicts for a list of (state, action) pairs
    # for TD3 we take min(q1, q2). it is the value TD3 itself uses in its learning targets
    with torch.no_grad(): # no learning here because we are just reading the predictions
        o = torch.FloatTensor(np.array(obs)).to(device)
        a = torch.FloatTensor(np.array(actions)).to(device)
        if algo == "td3":
            q1, q2 = critic(o, a)
            return torch.min(q1, q2).cpu().numpy().flatten()
        return critic(o, a).cpu().numpy().flatten()

def episode_bias(critic, algo, obs, actions, rewards, truncated, args, device):
    # mean of Q(s_t,a_t) - G_t over an episode. for any truncated episodes, MC return
    # misses the tail so the last mc_horizon steps are discarded from the bias estimate
    # we then compare, step by step, what the critic predicted (Q) with what the agent really got (G)
    returns = discounted_returns(rewards, args.gamma)

    # an episode stopped by the step 1000 time limit has not been finished so the last steps
    # are missing their future rewards. We ignore the last 500 steps of those episodes
    n = len(rewards) - args.mc_horizon if truncated else len(rewards)
    if n <= 0:
        return None
    q_vals = critic_q(critic, algo, obs[:n], actions[:n], device)

    # returns: average prediction, average real return, average bias (prediction - reality)
    return np.mean(q_vals), np.mean(returns[:n]), np.mean(q_vals - returns[:n])

def evaluate(actor, critic, args, device, eval_env, seed):
    # we run the deterministic policy with no exploration noise
    ep_returns, biases = [], []
    for ep in range(args.eval_episodes):
        o, _ = eval_env.reset(seed=seed + ep) # same starting positions at every evaluation
        obs_l, act_l, rew_l = [], [], []
        done, truncated = False, False
        while not done:
            with torch.no_grad():
                a = actor(torch.Tensor(o).unsqueeze(0).to(device)).cpu().numpy()[0] # the actor's action with no noise
            o2, r, term, trunc, _ = eval_env.step(a)
            obs_l.append(o); act_l.append(a); rew_l.append(r)
            o = o2
            done, truncated = term or trunc, trunc # term = landed/crashed & trunc = time limit reached
        ep_returns.append(sum(rew_l)) # score of the episode
        res = episode_bias(critic, args.algo, obs_l, act_l, rew_l, truncated, args, device)
        if res is not None:
            biases.append(res[2])
    return np.mean(ep_returns), (np.mean(biases) if biases else np.nan)

def main():
    # main execution
    args = parse_args() # get experiment settings

    # each run has a folder in v2/runs where the name contains its settings (plot.py can then read data back from it).
    run_name = f"{args.env_id}__{args.algo}__{args.exp_name}__HL{args.hidden_layers}__LN{args.use_layernorm}__UTD{args.utd_ratio}__{args.seed}__{int(time.time())}"
    runs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs") # v2/runs
    writer = SummaryWriter(os.path.join(runs_dir, run_name)) # tensorboard
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # for reproducibility, every random generator will get the same seed so we get the same run
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # torch configuration, if a CPU or a GPU is available we can use it
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # gym environments for experiment
    env = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, run_name)]) # LunaLander env for training (autoreset does not store transitions)
    eval_env = gym.make(args.env_id)      # for the tests without noise

    # in both algorithms, we have two actors networks where one is used for learning and the other one follows the first one slowly
    actor = Actor(env, args.hidden_layers, args.use_layernorm).to(device)
    target_actor = Actor(env, args.hidden_layers, args.use_layernorm).to(device)
    target_actor.load_state_dict(actor.state_dict()) # we can initiate both networks as copies
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    # same applies for critic: one network will learn and the other one will follow slowly
    critic = Critic(env, args.hidden_layers, args.use_layernorm, args.algo).to(device)
    target_critic = Critic(env, args.hidden_layers, args.use_layernorm, args.algo).to(device)
    target_critic.load_state_dict(critic.state_dict())
    critic_optimizer = optim.Adam(list(critic.parameters()), lr=args.learning_rate)

    # create a replay buffer to store the transitions 
    rb = ReplayBuffer(args.buffer_size, env.single_observation_space.shape, env.single_action_space.shape, device)

    obs, _ = env.reset(seed=args.seed) # create the first state of the first episode

    # lists to track overestimation bias. We remember the current episode to compare Q with the real return at its end
    ep_obs = []
    ep_actions = []
    ep_rewards = []

    # according to docs, gymnasium>=1.0 vector envs use NEXT_STEP autoreset so the step following a terminal step
    # only resets the env (action ignored and reward 0). we shouldn't store that transition
    autoreset = False
    n_updates = 0 # number of critic updates done

    # main loop where one iteration = one step in the game
    for global_step in range(args.total_timesteps):
        # every eval_freq steps (i.e. 10,000) we test the agent without noise and store its score and bias
        if global_step % args.eval_freq == 0:
            eval_ret, eval_bias = evaluate(actor, critic, args, device, eval_env, seed=10_000 + args.seed * 100)
            writer.add_scalar("eval/episodic_return", eval_ret, global_step)
            writer.add_scalar("eval/overestimation_bias", eval_bias, global_step)

        # CHOOSE ACTION
        # to begin (first 10000 steps) we play randomly because we have to fill the memory with various examples
        if global_step < args.learning_starts:
            actions = np.array([env.single_action_space.sample() for _ in range(env.num_envs)]) # one random action
        # after that, we ask the actor what to do and add a little noise so the agent can keep trying new things
        else:
            with torch.no_grad(): # no learning here we just play
                actions = actor(torch.Tensor(obs).to(device)) # the actor's action
                actions += torch.normal(0, actor.action_scale * args.exploration_noise) # add a small random change
                actions = actions.cpu().numpy().clip(env.single_action_space.low, env.single_action_space.high) # clamp between -1 and 1

        # PLAY IN SIMULATOR
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        if autoreset:
            # reset step. there is nothing to store so we just continue from the new initial observation
            autoreset = False
            obs = next_obs
            continue
        autoreset = bool(terminations[0] or truncations[0]) # was this step the end of episode?

        # save trajectory for Monte-Carlo return
        ep_obs.append(obs[0])
        ep_actions.append(actions[0])
        ep_rewards.append(rewards[0])

        real_next_obs = next_obs.copy()

        # END OF AN EPISODE, we log the score and the overestimation bias
        if "episode" in infos and infos["_episode"][0]:
            r = infos["episode"]["r"][0] # total reward of the episode
            l = infos["episode"]["l"][0] # number of steps of the episode
            writer.add_scalar("charts/episodic_return", r, global_step)
            writer.add_scalar("charts/episodic_length", l, global_step)

            # we get the Q-values vs true discounted MC returns on the behaviour trajectory
            res = episode_bias(critic, args.algo, ep_obs, ep_actions, ep_rewards, bool(truncations[0]), args, device)
            if res is not None:
                avg_q, avg_g, bias = res
                writer.add_scalar("charts/overestimation_bias", bias, global_step)
                writer.add_scalar("charts/avg_q_value", avg_q, global_step)
                writer.add_scalar("charts/avg_mc_return", avg_g, global_step)

            ep_obs = [] # we now can start remembering the next episode
            ep_actions = []
            ep_rewards = []

        # STORE TRANSITION IN RB
        # if we are at "done" then terminations only. if episode was cut by time limit, the game was not over and
        # so the learning target still adds the value of the next state.
        rb.add(obs[0], actions[0], rewards[0], real_next_obs[0], terminations[0])
        obs = next_obs

        # LEARN (only happens after the first 10,000 random steps)
        if global_step > args.learning_starts:
            # UTD ratio (update-to-data) = number of learning updates per step played (1 by default, 2 in one ablation) 
            # ablation = experiment where one element is changed to see how it affects the result
            for _ in range(args.utd_ratio):
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = rb.sample(args.batch_size)   # 256 random transitions

                # compute learning target y = r + gamma * Q_target(next state, next action)
                # => the reward now + the discounted value of what comes next. if done=1 there is no next
                with torch.no_grad():
                    next_state_actions = target_actor(b_next_obs)
                    if args.algo == "td3":
                        # TD3 trick 1 (target policy smoothing): we add a small clipped noise to the next action
                        # so the critic cannot exploit a little high peak of Q
                        noise = (torch.randn_like(b_actions) * args.policy_noise * actor.action_scale)
                        noise = noise.clamp(-args.noise_clip * actor.action_scale, args.noise_clip * actor.action_scale)
                        next_state_actions = (next_state_actions + noise).clamp(
                            torch.tensor(env.single_action_space.low).to(device),
                            torch.tensor(env.single_action_space.high).to(device)
                        )
                        # TD3 trick 2 (clipped double Q-learning): we take the smaller of the two critics' estimates
                        # this helps fight overestimation
                        q1_next, q2_next = target_critic(b_next_obs, next_state_actions)
                        min_q_next = torch.min(q1_next, q2_next)
                        next_q_value = b_rewards + (1 - b_dones) * args.gamma * min_q_next
                    else: # ddpg: a single critic and no protection against overestimation
                        next_q_value = b_rewards + (1 - b_dones) * args.gamma * target_critic(b_next_obs, next_state_actions)

                # train the critic. we attempt to bring Q(s, a) closer to the target y using mean squared error
                if args.algo == "td3":
                    q1, q2 = critic(b_obs, b_actions)
                    critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value) # both critics learn the same target
                else:
                    q = critic(b_obs, b_actions)
                    critic_loss = F.mse_loss(q, next_q_value)

                critic_optimizer.zero_grad() # clear old gradients
                critic_loss.backward() # compute new gradients
                critic_optimizer.step() # move the weights a little
                n_updates += 1

                # train the actor (TD3 trick 3: only every 2 critic updates) = delayed policy updates
                # and DDPG after every critic update.
                # CTA (critic-to-actor) ratio = policy_frequency (counted in gradient steps so it also holds for UTD > 1)
                if n_updates % args.policy_frequency == 0 or args.algo == "ddpg":
                    # actor wants actions with a high Q-value so we try to minimise -Q
                    # overestimation: the actor tries actions where the critic is too optimistic
                    if args.algo == "td3":
                        actor_loss = -critic.q1(b_obs, actor(b_obs)).mean()
                    else:
                        actor_loss = -critic(b_obs, actor(b_obs)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # update the target networks: move each target weight 0.5% (tau = 0.005) towards the real weight
                    for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # we log the losses every 1000 steps (tensorboard)
            if global_step % 1000 == 0:
                writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                if (global_step % args.policy_frequency == 0 or args.algo == "ddpg") and 'actor_loss' in locals():
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

    # final test after the last training step
    eval_ret, eval_bias = evaluate(actor, critic, args, device, eval_env, seed=10_000 + args.seed * 100)
    writer.add_scalar("eval/episodic_return", eval_ret, args.total_timesteps)
    writer.add_scalar("eval/overestimation_bias", eval_bias, args.total_timesteps)

    env.close()
    eval_env.close()
    writer.close()

if __name__ == "__main__":
    main()
