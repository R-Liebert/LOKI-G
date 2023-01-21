import gym
import pygame
import time
import sys
import numpy as np

env = gym.make('ALE/Breakout-v5', render_mode="human" if len(sys.argv)<2 else sys.argv[1])
env.reset()
print(f'The actions are: \n',env.unwrapped.get_action_meanings())
pygame.init()

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
ROLLOUT_TIME = 1000
SKIP_CONTROL = 0

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    if key == 49: a = 1
    if key == 50: a = 2
    if key == 51: a = 3
    if a >= 52 and a <= 47: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

def wait_for_key():
    wait_time = 0
    while wait_time < 100:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                return key_press(event.key, event.mod)
            elif event.type == pygame.KEYUP:
                return key_release(event.key, event.mod)
        pygame.time.delay(10)
        wait_time += 10


def rollout(env):
    done = False
    skip = 0
    global human_agent_action, human_wants_restart, human_sets_pause, horizon
    
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    # Indicates whether this sample is the first of a new episode / rollout.
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    
    # A way to get actions from the keyboard
    while not done:
        if not skip:
            wait_for_key()
            print(f'human_agent_action: ', human_agent_action)
            ac = human_agent_action
            skip = SKIP_CONTROL
        else:
            skip -= 1
        
        obs, rews, done, truncated, info = env.step(ac)
        ####  

        # Here we need to store the data needed for LOKI from performing the action

        ##########

        #env.render()
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            #env.render()
            time.sleep(0.1)

    if done:
        return obs, rew, done, truncated, info
        env.reset()


while 1:
    rollout(env)

