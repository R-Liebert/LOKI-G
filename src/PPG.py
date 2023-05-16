#!/usr/bin/env python3

""" 
This scripts originates from the following repo: https://github.com/wisnunugroho21/reinforcement_learning_phasic_policy_gradient/blob/master/discrete/tensorflow/ppg_dis_tf.py
But has been adapted to be a Single Network PPG that uses a Closed-form Continous-time Neural Network. The model is not initiated in this script, but loaded from saved_model.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

class PolicyMemory():
    def __init__(self):
        self.actions        = [] 
        self.states         = []
        self.rewards        = []
        self.dones          = []
        self.next_states    = []

    def __len__(self):
        return len(self.dones)

    def get_all_tensor(self):
        states = tf.ragged.constant(self.states, dtype = tf.float32)
        actions = tf.ragged.constant(self.actions, dtype = tf.int32)
        rewards = tf.expand_dims(tf.ragged.constant(self.rewards, dtype = tf.int32), 1)
        done = tf.expand_dims(tf.ragged.constant(self.dones, dtype = tf.int32), 1)
        next_states = tf.ragged.constant(self.next_states, dtype = tf.float32)
        
        return tf.data.Dataset.from_tensor_slices((states, actions, rewards, done, next_states))

    def get_all(self):
        return self.states, self.actions, self.rewards, self.dones, self.next_states     

    def save_eps(self, state, action, reward, done, next_state):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
        self.next_states.append(next_state)        

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]

class AuxMemory():
    def __init__(self):
        self.states = []

    def __len__(self):
        return len(self.states)

    def get_all_tensor(self):
        states = tf.constant(self.states, dtype = tf.float32)        
        return tf.data.Dataset.from_tensor_slices(states)

    def save_all(self, states):
        self.states = self.states + states

    def clear_memory(self):
        del self.states[:]

class Discrete():
    def sample(self, datas):
        distribution = tfp.distributions.Categorical(probs = datas)
        return distribution.sample()
        
    def entropy(self, datas):
        distribution = tfp.distributions.Categorical(probs = datas)            
        return distribution.entropy()
      
    def logprob(self, datas, value_data):
        distribution = tfp.distributions.Categorical(probs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # probs set to action space
        return tf.expand_dims(distribution.log_prob(value_data), 1)

    def kl_divergence(self, datas1, datas2):
        distribution1 = tfp.distributions.Categorical(probs = datas1)
        distribution2 = tfp.distributions.Categorical(probs = datas2)

        return tf.expand_dims(tfp.distributions.kl_divergence(distribution1, distribution2), 1)


class PolicyFunction():
    def __init__(self, gamma = 0.99, lam = 0.95):
        self.gamma  = gamma
        self.lam    = lam

    def monte_carlo_discounted(self, rewards, done):
        running_add = 0
        returns     = []
        dones       = done      
        
        for step in reversed(range(len(rewards))):
            running_add = rewards[step] + (1.0 - dones[step]) * self.gamma * running_add
            returns.insert(0, running_add)
            
        return tf.stack(returns)
      
    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value           
        return q_values
      
    def generalized_advantage_estimation(self, values, rewards, next_values, dones):
        gae     = 0
        adv     = []     
        delta = tf.cast(rewards, dtype=tf.float32) + tf.cast((1 - dones), dtype=tf.float32) * tf.cast(self.gamma, dtype=tf.float32) * next_values - values         
        for step in reversed(range(len(rewards))):
            gae = delta[step] + tf.cast((1 - dones[step]), dtype=tf.float32) * tf.cast(self.gamma, dtype=tf.float32) * tf.cast(self.lam, dtype=tf.float32) * gae
            adv.insert(0, gae)
            
        return tf.stack(adv)

class TrulyPPO():
    def __init__(self, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma, lam):
        self.policy_kl_range    = policy_kl_range
        self.policy_params      = policy_params
        self.value_clip         = value_clip
        self.vf_loss_coef       = vf_loss_coef
        self.entropy_coef       = entropy_coef

        self.distributions      = Discrete()
        self.policy_function    = PolicyFunction(gamma, lam)

    # Loss for PPO  
    def compute_loss(self, action_probs, old_action_probs, values, old_values, next_values, actions, rewards, dones):
        # Don't use old value in backpropagation
        Old_values          = tf.stop_gradient(old_values)
        Old_action_probs    = tf.stop_gradient(old_action_probs)

        # Getting general advantages estimator
        Advantages      = self.policy_function.generalized_advantage_estimation(values, rewards, next_values, dones)
        Returns         = tf.stop_gradient(Advantages + values)
        Advantages      = tf.stop_gradient((Advantages - tf.math.reduce_mean(Advantages)) / (tf.math.reduce_std(Advantages) + 1e-6))

        # Finding the ratio (pi_theta / pi_theta__old):        
        logprobs        = self.distributions.logprob(action_probs, actions)
        Old_logprobs    = tf.stop_gradient(self.distributions.logprob(Old_action_probs, actions))
        ratios          = tf.math.exp(logprobs - Old_logprobs) # ratios = old_logprobs / logprobs

        # Finding KL Divergence                
        Kl              = self.distributions.kl_divergence(Old_action_probs, action_probs)

        # Combining TR-PPO with Rollback (Truly PPO)
        pg_loss         = tf.where(
                tf.logical_and(Kl >= self.policy_kl_range, ratios > 1),
                ratios * Advantages - self.policy_params * Kl,
                ratios * Advantages
        )
        pg_loss         = tf.math.reduce_mean(pg_loss)

        # Getting entropy from the action probability
        dist_entropy    = tf.math.reduce_mean(self.distributions.entropy(action_probs))

        # Getting critic loss by using Clipped critic value
        vpredclipped    = old_values + tf.clip_by_value(values - Old_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
        vf_losses1      = tf.math.square(Returns - values) * 0.5 # Mean Squared Error
        vf_losses2      = tf.math.square(Returns - vpredclipped) * 0.5 # Mean Squared Error
        critic_loss     = tf.math.reduce_mean(tf.math.maximum(vf_losses1, vf_losses2))           

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss 
        loss            = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss

class JointAux():
    def __init__(self):
        self.distributions  = Discrete()

    def compute_loss(self, action_probs, old_action_probs, values, Returns):
        # Don't use old value in backpropagation
        Old_action_probs    = tf.stop_gradient(old_action_probs)

        # Finding KL Divergence                
        Kl              = tf.math.reduce_mean(self.distributions.kl_divergence(Old_action_probs, action_probs))
        aux_loss        = tf.math.reduce_mean(tf.math.square(Returns - values) * 0.5)

        return aux_loss + Kl

class Agent():  
    def __init__(self, model, is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                 batchsize, n_update, PPO_epochs, gamma, lam, learning_rate):        
        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batchsize          = batchsize
        self.n_update           = n_update
        self.PPO_epochs         = PPO_epochs
        self.is_training_mode   = is_training_mode


        self.policy             = model
        self.policy_old         = model


        self.policy_memory      = PolicyMemory()
        self.policy_loss        = TrulyPPO(policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma, lam)

        self.aux_memory         = AuxMemory()
        self.aux_loss           = JointAux()
         
        self.optimizer          = Adam(learning_rate = learning_rate)
        self.distributions      = Discrete()        

    def save_eps(self, state, action, reward, done, next_state):
        self.policy_memory.save_eps(state, action, reward, done, next_state)

    def act(self, state):
        # check if state is NoneType
        
        return_state = False
        if type(state) == tuple:
            return_state = True
            state, next_state = state
            state = np.expand_dims(np.expand_dims(state, 0), 0)
            state = (state, next_state)
            action_probs = self.policy.call(state)

        else:
            state = np.expand_dims(np.expand_dims(state, 0), 0)
            action_probs = self.policy.call(state)

        if isinstance(action_probs, tuple):
            action_probs, next_state = action_probs
            return_state = True
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            #action = np.argmax(action_probs[0, 0])
            action  = self.distributions.sample(action_probs) 
        else:
            action = np.argmax(action_probs)

        if return_state:
            return tf.squeeze(action), next_state
        else:
            return tf.squeeze(action)

    # Get loss and Do backpropagation
    @tf.function
    def training_ppo(self, states, actions, rewards, dones, next_states):
        with tf.GradientTape() as tape:
            action_probs = self.policy.call(states)
            if isinstance(action_probs, tuple):
                action_probs, next_states = action_probs
            values = self.policy.get_value()
            old_action_probs  = self.policy_old.call(states)
            if isinstance(old_action_probs, tuple):
                old_action_probs, _  = old_action_probs
            old_values = self.policy_old.get_value()
            _ = self.policy.call(next_states)
            next_values = self.policy.get_value()

            if isinstance(action_probs, tuple):
                action_probs, next_states = action_probs
            if isinstance(old_action_probs, tuple):
                old_action_probs, _  = old_action_probs

            loss = self.policy_loss.compute_loss(action_probs, old_action_probs, values, old_values, next_values, actions, rewards, dones)

        gradients = tape.gradient(loss, self.policy.trainable_variables)        
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

    @tf.function
    def training_aux(self, states):
        _ = self.policy.call(states)
        Returns = tf.stop_gradient(self.policy.get_value())

        with tf.GradientTape() as tape:
            action_probs = self.policy.call(states)
            if isinstance(action_probs, tuple):
                action_probs, _ = action_probs
            values = self.policy.get_value()
            old_action_probs = self.policy_old.call(states)
            if isinstance(old_action_probs, tuple):
                old_action_probs, _ = old_action_probs

            joint_loss = self.aux_loss.compute_loss(action_probs, old_action_probs, values, Returns)

        gradients = tape.gradient(joint_loss, self.policy.trainable_variables)        
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

    # Update the model
    def update_ppo(self):
        for _ in range(self.PPO_epochs):
            states, actions, rewards, dones, next_states = self.policy_memory.get_all()
            for i in range((self.batchsize+self.n_update)//self.batchsize):
                start_idx = i * self.batchsize
                end_idx = (i + 1) * self.batchsize
                batch_states = np.array(states[start_idx:end_idx])
                batch_actions = actions[start_idx:end_idx]
                batch_rewards = np.array(rewards[start_idx:end_idx])
                batch_dones = np.array(dones[start_idx:end_idx])
                batch_next_states = np.array(next_states[start_idx:end_idx])
                if batch_states.shape[0] == 0 or batch_next_states.shape[0] == 0:
                    continue
                self.training_ppo(batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states)


        # Clear the memory
        states, _, _, _, _ = self.policy_memory.get_all()
        self.aux_memory.save_all(states)
        self.policy_memory.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.set_weights(self.policy.get_weights())

    def update_aux(self):
        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs): 
            for states in self.aux_memory.get_all_tensor().batch(self.batchsize):
                self.training_aux(states)

        # Clear the memory
        self.aux_memory.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.set_weights(self.policy.get_weights())

    def save_weights(self):
        self.policy.save_weights('policy_ppg_latest.ckpt')
        
    def load_weights(self):
        self.policy.load_weights('trained_model.ckpt')

class Runner():
    def __init__(self, env, agent, render, training_mode, n_update, n_aux_update):
        self.env = env
        self.agent = agent
        self.render = render
        self.training_mode = training_mode
        self.n_update = n_update
        self.n_aux_update = n_aux_update

        self.t_updates = 0
        self.t_aux_updates = 0

    def run_episode(self, episode):
        ############################################
        obs = self.env.reset()
        obs = np.moveaxis(obs, 0, -1)
        done = False
        total_reward = 0
        eps_time = 0
        next_state = self.agent.policy.get_initial_states()
        ############################################ 
        for i in range(1, 5000): # Adjust to how long you want the episode maximum to be
            if i == 1:
                action_probs = self.agent.act(obs)
                if isinstance(action_probs, tuple):
                    action_probs, next_state = action_probs
            else:
                obs = (obs, next_state)
                action_probs = self.agent.act(obs)
                if isinstance(action_probs, tuple):
                    action_probs, next_state = action_probs
            action = action_probs[0,0].argmax()
            print(f"action_probs = {action_probs}")
            print(f"action = {action}")
            if action_probs > 10:
                print(f"action_probs = {action_probs}") # Should never happen, but sometimes does...
                action_probs = 9
            
            next_obs, reward, done, _ = self.env.step(action)

            #next_obs = np.concatenate((next_obs, zero_array), axis=-1)
            eps_time += 1 
            self.t_updates += 1
            total_reward += reward
            
            if self.training_mode and i != 1:
                # save the experience as lists in the memory
                if isinstance(obs, tuple):
                    obs_t = np.expand_dims(obs[0], 0)
                else:
                    obs_t = np.expand_dims(obs, 0)
                if isinstance(next_obs, tuple):
                    obs_tp = np.expand_dims(next_obs[0], 0)
                else:
                    obs_tp = np.expand_dims(next_obs, 0)
                self.agent.policy_memory.save_eps(obs_t, action_probs, reward, int(done == True), obs_tp) 
                
            obs = next_obs
                    
            if self.render:
                self.env.render()
            
            if self.training_mode and self.n_update is not None and self.t_updates == self.n_update:
                self.agent.update_ppo()
                self.t_updates = 0
                self.t_aux_updates += 1

                if self.t_aux_updates == self.n_aux_update:
                    self.agent.update_aux()
                    self.t_aux_updates = 0

            if done: 
                model.save(f"../saved_models/PPG_expert{episode}", save_format="tf")
                break                
        
        if self.training_mode and self.n_update is None:
            self.agent.update_ppo()
            self.t_aux_updates += 1

            if self.t_aux_updates == self.n_aux_update:
                self.agent.update_aux()
                self.t_aux_updates = 0
        
        return total_reward, eps_time

def plot(datas):
    print('----------')

    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()

    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))

def run_PPG(env, epochs=10, render=False):
    ############## Hyperparameters ##############
    load_weights        = False # If you want to load the agent, set this to True, (Temp done in the code)
    save_weights        = True # If you want to save the agent, set this to True
    training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold    = 1750 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    using_google_drive  = False

    n_update            = 1 # How many episode before you update the Policy. Recommended set to 128 for Discrete
    n_plot_batch        = 10 # How many episode you want to plot the result
    n_episode           = 50 # How many episode you want to run
    n_saved             = 3 # How many episode to run before saving the weights

    policy_kl_range     = 0.0008 # Set to 0.0008 for Discrete
    policy_params       = 20 # Set to 20 for Discrete
    value_clip          = 825 # How many value will be clipped. Recommended set to the highest or lowest possible reward
    entropy_coef        = 0.05 # How much randomness of action you will get
    vf_loss_coef        = 1.0 # Just set to 1
    batchsize           = 4 # How many batch per update. size of batch = n_update / batchsize. Rocommended set to 4 for Discrete
    PPO_epochs          = epochs # How many epoch per update
    n_aux_update        = 4

    
    gamma               = 0.99 # Just set to 0.99
    lam                 = 0.7 # Just set to 0.95
    learning_rate       = 0.0001 # Same as BC
    ############################################# 
    try:
        model           = tf.keras.models.load_model("../saved_models/sub_optimal_expert.keras")
        print("Model loaded")
    except:
        print("No nodel found")
        
    agent               = Agent(model, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                            batchsize, n_update, PPO_epochs, gamma, lam, learning_rate)  

    runner              = Runner(env, agent, render, training_mode, n_update, n_aux_update)
    #############################################     
    if using_google_drive:
        render = False
        from google.colab import drive
        drive.mount('/test')

    if load_weights:
        agent.load_weights()
        print('Weight Loaded')

    rewards             = []   
    batch_rewards       = []
    batch_solved_reward = []

    times               = []
    batch_times         = []

    for i_episode in range(1, n_episode + 1):

        total_reward, time = runner.run_episode(i_episode)

        print(f'Episode {i_episode} \t t_reward: {total_reward} \t time: {time} \t ')
        batch_rewards.append(int(total_reward))
        batch_times.append(time)        

        if save_weights:
            if i_episode % n_saved == 0:
                agent.save_weights() 
                print('weights saved')

        if reward_threshold:
            if len(batch_solved_reward) == 100:            
                if np.mean(batch_solved_reward) >= reward_threshold:
                    print(f'You solved task after {len(rewards)} episode')
                    break

                else:
                    del batch_solved_reward[0]
                    batch_solved_reward.append(total_reward)

            else:
                batch_solved_reward.append(total_reward)

        if i_episode % n_plot_batch == 0 and i_episode != 0:
            # Plot the reward, times for every n_plot_batch
            plot(batch_rewards)
            plot(batch_times)

            for reward in batch_rewards:
                rewards.append(reward)

            for time in batch_times:
                times.append(time)

            batch_rewards   = []
            batch_times     = []

            print('========== Cummulative ==========')
            # Plot the reward, times for every episode
            plot(rewards)
            plot(times)

    print('========== Final ==========')
    # Plot the reward, times for every episode
    model.save("../saved_models/expert_model.keras")
    
    for reward in batch_rewards:
        rewards.append(reward)

    for time in batch_times:
        times.append(time)

    plot(rewards)
    plot(times)
    print('Model saved as expert_model.keras')
    return agent.policy
