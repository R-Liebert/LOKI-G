# /usr/bin/env python3

"""LOKI (Locally Optimal search after K-step Imitation) is a algorithm for finding the optimal policy for a model"""

# Imports

import numpy as np
import random
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PPO.PPO_CfC import initiate_PPO_CfC
from Expert_policy.get_expert_policy import Expert_policy

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
    parser.add_argument('--expert_file_prefix', type=str, help='the prefix for the policy checkpoint', default='policy')
    parser.add_argument('--pretrain_dir', type=str, help='the directory of a pretrained policy for learner initialization')
    parser.add_argument('--pretrain_file_prefix', type=str, help='the prefix for the policy checkpoint', default='policy')
    parser.add_argument('--save_no_policy', action='store_true',
                        help='whether the learner policies will be saved')
    parser.add_argument('--ilrate_multi', type=float, default=0.0,
                        help='the multiplier of il objective [online_il, batch_il]')
    parser.add_argument('--ilrate_decay', default=1.0, type=float, help='the decay of il objective [online_il, batch_il]')
    parser.add_argument('--hard_switch_iter', type=int, help='switch from il to rl at this iteration [online_il, batch_il]')
    parser.add_argument('--random_sample_switch_iter', action='store_true',
                        help='random sample the iteration to switch from il to rl, with a distribution parameterized by ' +
                        '--hard_switch_iter')
    parser.add_argument('--truncated_horizon', type=int, help='horizon [thor]')
    parser.add_argument('--deterministic_expert', action='store_true',
                        help='whether to use deterministic expert for BC')
    parser.add_argument('--il_gae', action='store_true')
    parser.add_argument('--initialize_with_expert_logstd', action='store_true')
    args = parser.parse_args()
    return args


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

# Calculate KL divergence between two policies

def KL_divergence(p, q):
    """
    Calculate the KL divergence between two policies.
    
    Input:
        p: policy 1
        q: policy 2
        
    Output:
        KL: the KL divergence between p and q
    """
    KL = 0
    for i in range(len(p)):
        KL += p[i] * np.log(p[i]/q[i])
    return KL


def mirror_decent(self, expert_policy, ppo_policy, n_iter=1000, verbose=True):
    """
    function for doing mirror decent between the expert and ray.rllib ppo agent.
    
    Input:
        expert_policy: the expert policy
        ppo_policy: the ppo policy
        n_iter: number of iterations to run the algorithm
        verbose: print the current iteration and reward
    
    Output:
        best_reward: the best reward found
        best_policy: the best policy found
    """
    for i in range(n_iter):
        # sample from the expert
        obs, actions, rewards, dones, infos = expert_policy.sample()
        # train the ppo policy on the expert data
        ppo_policy.train(obs, actions, rewards, dones, infos)
        # sample from the ppo policy
        obs, actions, rewards, dones, infos = ppo_policy.sample()
        # train the expert policy on the ppo data
        expert_policy.train(obs, actions, rewards, dones, infos)
        # evaluate the ppo policy
        reward = ppo_policy.evaluate()
        # save the best policy
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_policy = ppo_policy
        # print the current iteration and reward
        if verbose:
            print(f"iteration: {i}, reward: {reward}")
    return self.best_reward, self.best_policy

    

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


    def run(self, expert_policy, n_iter=1000, verbose=True):
        """
        Run the LOKI algorithm.

        Input:
            n_iter: number of iterations to run the algorithm
            verbose: print the current iteration and reward

        Output:
            best_policy: the best policy found by the algorithm
        """

        # Initial values for variables for IL
        max_kl = 0.1

        for i in range(n_iter):
            # Sample a policy
            policy = self.model.sample_policy()
            # Find imitation gradient
            self.model.find_imitation_gradient(policy, expert_policy)
            # Perform Mirrored Descent and update the policy
            best_mirror_reward, best_policy = self.model.mirror_decent(expert_policy, policy)

            # Evaluate the policy
            reward = self.model.evaluate_policy(best_policy)
            # Update the best policy
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_policy = policy
            # Update the number of samples
            self.N += 1
            # Print the current iteration and reward
            if verbose:
                print(f"Iteration: {i}, Reward: {reward}")
            # Check if we should switch from IL to RL
            if self.N == self.K:
                # Switch to RL values for variables
                max_kl = 0.01

                self.model.switch_to_rl()
                print("Switching to RL")
        return self.best_policy
    