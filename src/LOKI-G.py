#!/usr/bin/env python3

import numpy as np
import argparse
from importlib.machinery import SourceFileLoader

from CfCmodels import ConvCfC
from BC import train_BC
from PPG import run_PPG

# Setting up args
def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--env_file", type=str, default="env.py",
    #                    help="Path to the environment file")
    parser.add_argument("--demonstration_path", type=str, default="../data", 
                        help="Directory containing the demonstration data. Default is './data'.")
    parser.add_argument("--num_outputs", default=6, type=int, 
                        help="Number of actions the model can output. Default is 6.")
    parser.add_argument('--hard_switch_iter', default=18, type=int, 
                        help='Iteration to switch from imitation learning (IL) to reinforcement learning (RL) during training. Default is 18.')
    parser.add_argument('--random_sample_switch_iter', default=True, action='store_true',
                        help='If set, randomly choose the iteration to switch from IL to RL, with a distribution parameterized by hard_switch_iter. Default is True.')
    parser.add_argument("--il_epochs", default=10, type=int, 
                        help="Number of epochs to train the imitation learning model. Default is 10.")
    parser.add_argument("--rl_epochs", default=10, type=int, 
                        help="Number of epochs to train the reinforcement learning model. Default is 10.")
    parser.add_argument("--render", action="store_true", 
                        help="If set, render the environment during training. Default is False.")
    
    args = parser.parse_args()

    return args

def main():
    # Parse command line arguments
    args=get_args()

    env = SourceFileLoader("env", args.env_file).load_module()

    # If random_sample_switch_iter is true, randomly select an iteration to switch from IL to RL
    if args.random_sample_switch_iter:
        T = args.hard_switch_iter 
        assert T is not None
        low = T // 2
        high = T + 1
        prob_mass = np.arange(low, high) ** 2 
        prob_mass = prob_mass / np.sum(prob_mass)
        K = np.round(np.random.choice(np.arange(low, high), p=prob_mass))
        print (f"Switching from IL to RL after training on {K} examples")
    else:
        K = args.hard_switch_iter

    # Perform imitation learning. There is a known issue with the model not training properly due to an InvalidArgumentError during initial run. Hence the loop.
    model_trained = False
    while model_trained == False:
        model_trained = train_BC(K=K, num_outputs=args.num_outputs, data_path=args.demonstration_path, epochs=args.il_epochs)
    
    print("IL done")
    
    # Perform reinforcement learning
    try:
        run_PPG(env=env, epochs=args.rl_epochs, render=args.render)
        print("RL done")
    except:
        print("Failed to do RL")
        exit()

if __name__ == "__main__":
    main()
