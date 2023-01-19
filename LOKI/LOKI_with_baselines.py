# /usr/bin/env python3

"""LOKI (Locally Optimal search after K-step Imitation) is a algorithm for finding the optimal policy for a model"""

# Imports

import numpy as np
import random
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PPO.PPO_CfC import initiate_PPO_CfC
from Expert_policy.get_expert_policy import Expert_policy
from baselines.baselines.trpo_mpi import trpo_mpi

# Functions

def parse_args():
    """
    Parse the arguments from the command line. !!!!! Must be first in Main() in run_experiment.py !!!!!
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter ,description="LOKI (Locally Optimal search after K-step Imitation) is a algorithm for finding the optimal policy for a model")
    parser.add_argument("-n", "--n_iter", type=int, default=1000, help="number of iterations to run the algorithm")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the current iteration and reward")
    parser.add_argument("-N", "--Nmax", type=int, default=50, help="maximum number of samples")
    parser.add_argument("-d", "--d", type=int, default=3, help="number of dimensions")
    parser.add_argument('--timesteps_per_batch', type=int, default=1000)  # number of timesteps per batch per worker
    parser.add_argument('--mode', type=str, choices=['train_expert', 'pretrain_il', 'online_il', 'batch_il', 'thor', 'lols'])
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--workspace', type=str, default='')
    parser.add_argument('--log_dir', default='', type=str, help='this is where the policy and results will be saved')
    parser.add_argument('--policy_save_freq', default=50, type=int,
                        help='learner policy will be saved per these iterations')
    parser.add_argument('--expert_dir', type=str, help='the directory of the expert policy to be loaded')
    parser.add_argument('--pretrain_dir', type=str, help='the directory of a pretrained policy for learner initialization')
    parser.add_argument('--save_no_policy', action='store_true',
                        help='whether the learner policies will be saved')
    parser.add_argument('--ilrate_multi', type=float, default=0.0,
                        help='the multiplier of il objective [online_il, batch_il]')
    parser.add_argument('--ilrate_decay', default=1.0, type=float, help='the decay of il objective [online_il, batch_il]')
    parser.add_argument('--hard_switch_iter', type=int, default=50, help='switch from il to rl at this iteration [online_il, batch_il]')
    parser.add_argument('--random_sample_switch_iter', action='store_true',
                        help='random sample the iteration to switch from il to rl, with a distribution parameterized by ' +
                        '--hard_switch_iter')
    parser.add_argument('--deterministic_expert', action='store_true',
                        help='whether to use deterministic expert for BC')
    parser.add_argument('--il_gae', action='store_true')
    parser.add_argument('--initialize_with_expert_logstd', action='store_true')
    args = parser.parse_args()
    return args

# Logg the action, observation and reward of one trial of policy pi against a sample of the expert policy

def logg_action_observation_reward(pi, expert_policy, env, n=1000, verbose=True):
    """
    Logg the action, observation and reward of one trial of policy pi against a sample of the expert policy.
    
    Input:
        pi: the policy to be tested
        expert_policy: the expert policy
        env: the environment
        n: number of samples
        verbose: print the current iteration and reward

    Output:
        action: the action of the policy pi
        observation: the observation of the policy pi
        reward: the reward of the policy pi
    """
    action = []
    observation = []
    reward = []
    for i in range(n):
        o = env.reset()
        done = False
        while not done:
            a = pi.act(False, o)
            o2, r, done, _ = env.step(a)
            action.append(a)
            observation.append(o)
            reward.append(r)
            o = o2
            if verbose:
                print("Iteration: ", i, "Reward: ", r)
    return action, observation, reward

def policy_fn(name, ob_space, ac_space): # This is going to be changed to a CfC network
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=32, num_hid_layers=2)

def random_sample_switcher(d=3, Nmax=50):
    """
    Choose a random moment for switching from IL to RL based on a uniform distribution.

    The values are chosen between Nmin and Nmax. d, Nmin and Nmax are set from suggestions 
    in the original paper. https://arxiv.org/abs/1805.10413.

    Input:
        d: number of dimensions
        Nmax: maximum number of samples
        
    Output:
        K: number of samples before switching from IL to RL
        
    """
    Nmin = Nmax // 2
    prob_mass = np.arange(Nmin, Nmax+1) ** d 
    prob_mass = prob_mass / np.sum(prob_mass)
    switch_iter = np.random.choice(np.arange(Nmin//2, Nmax+1), p=prob_mass)

    return switch_iter


    

# Classes

class LOKI:
    """
    LOKI (Locally Optimal search after K-step Imitation) is a algorithm for finding the optimal policy for a model.
    """
    def __init__(self, model, Nmax=50, d=3):
        """
        Initialize the LOKI algorithm.

        Input:
            model: the model to be used for the algorithm
            Nmax: maximum number of samples
            d: number of dimensions
        """
        self.model = model
        self.Nmax = Nmax
        self.d = d
        self.K = random_sample_switcher(d=self.d, Nmax=self.Nmax)
        self.N = 0
        self.best_reward = -np.inf
        self.best_policy = None
        self.env = ABB.RobotEnv() # Or something
        self.hard_switch_iter = args.hard_switch_iter
        self.adv_fn = None


    def perform_IL(self, policy_fn, value_fn, expert=None, n_iter=1000, verbose=True):
        """
        Run the LOKI algorithms for imitation learning.

        Input:
            policy_fn: the policy function to be used
            expert: the expert policy
            n_iter: number of iterations to run the algorithm
            verbose: print the current iteration and reward

        Output:
            best_reward: the best reward found
            best_policy: the best policy found
        """
        # Initial values for variables for IL
        IL_MAX_KL= 0.1

        self.best_policy, self.adv_fn = trpo_mpi.learn(policy_fn, value_fn, self.env, cg_iters=10, cg_damping=0.1, total_timesteps=args.num_timesteps,
                   gamma=0.99, lam=0.98, ent_coef=0.0, seed=None, vf_iters=5, vf_stepsize=1e-3,
                   timesteps_per_batch=args.timesteps_per_batch, max_iters=args.max_iters,
                   max_episodes=0, policy_save_freq=args.policy_save_freq,
                   expert_dir=expert_dir,
                   ilrate_multi=args.ilrate_multi, ilrate_decay=args.ilrate_decay,
                   hard_switch_iter=args.hard_switch_iter,
                   save_no_policy=args.save_no_policy,
                   pretrain_dir=pretrain_dir,
                   il_gae=args.il_gae,
                   initialize_with_expert_logstd=args.initialize_with_expert_logstd)


    def perform_RL(self, policy_fn, value_fn, expert=None, n_iter=1000, verbose=True):
        """
        Run the LOKI algorithms for imitation learning.

        Input:
            policy_fn: the policy function to be used
            expert: the expert policy
            n_iter: number of iterations to run the algorithm
            verbose: print the current iteration and reward

        Output:
            best_reward: the best reward found
            best_policy: the best policy found
        """
        # Initial values for variables for RL
        RL_MAX_KL= 0.01

        self.best_policy = trpo_mpi.learn(policy_fn, value_fn, self.env, cg_iters=10, cg_damping=0.1, total_timesteps=args.num_timesteps,
                   gamma=0.99, lam=0.98, ent_coef=0.0, seed=None, vf_iters=5, vf_stepsize=1e-3,
                   timesteps_per_batch=args.timesteps_per_batch, max_iters=args.max_iters,
                   max_episodes=0, policy_save_freq=args.policy_save_freq,
                   expert_dir=expert_dir,
                   ilrate_multi=args.ilrate_multi, ilrate_decay=args.ilrate_decay,
                   hard_switch_iter=hard_switch_iter,
                   save_no_policy=args.save_no_policy,
                   pretrain_dir=pretrain_dir,
                   il_gae=args.il_gae,
                   initialize_with_expert_logstd=args.initialize_with_expert_logstd)

        return self.best_policy
    