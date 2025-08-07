import gym
from gym import spaces
import numpy as np

class SimulatedEnv(gym.Env):
    """
    A simulated environment for the LOKI-G project.
    This environment mimics the action and observation spaces of the RobotArmEnv
    but does not require any physical hardware.
    """
    def __init__(self):
        super(SimulatedEnv, self).__init__()

        # Define action space: 10 discrete actions
        self.action_space = spaces.Discrete(10)

        # Define observation space: 3 frames of 320x240 grayscale images
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8)

        # Current position of the simulated agent
        self.current_position = np.array([0.0, 0.0, 500.0])
        self.target_position = np.array([100.0, 100.0, 600.0])

        # A simple buffer to hold the last 3 frames
        self.frames = np.zeros((3, 240, 320), dtype=np.uint8)

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        # Move the agent based on the action
        self._update_position(action)

        # Calculate reward based on distance to target
        distance_to_target = np.linalg.norm(self.current_position - self.target_position)
        reward = -distance_to_target

        # Check if the episode is done (e.g., agent is close enough to target)
        done = distance_to_target < 20.0

        if done:
            reward = 100.0 # Bonus for reaching the target

        # Generate a new observation
        observation = self._get_observation()

        return observation, reward, done, {}

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        # Reset the agent's position
        self.current_position = np.array([0.0, 0.0, 500.0])
        # Return the initial observation
        return self._get_observation()

    def _update_position(self, action):
        """
        Update the agent's position based on the action.
        """
        magnitude = 15.0

        if action == 0:  # Move +x
            self.current_position[0] += magnitude
        elif action == 1:  # Move -x
            self.current_position[0] -= magnitude
        elif action == 2:  # Move -y
            self.current_position[1] -= magnitude
        elif action == 3:  # Move +y
            self.current_position[1] += magnitude
        elif action == 4:  # Move +z
            self.current_position[2] += magnitude
        elif action == 5:  # Move -z
            self.current_position[2] -= magnitude
        # Actions 6, 7, 8, 9 do nothing in this simple simulation

    def _get_observation(self):
        """
        Generate a new observation.
        For this simple simulation, we'll just create some noise.
        A more advanced simulation would render the environment.
        """
        # Create a new "image" frame with random noise
        new_frame = np.random.randint(0, 256, (240, 320), dtype=np.uint8)

        # Shift old frames and add the new one
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = new_frame

        return self.frames

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass

# Instantiate the environment and expose its methods at the module level.
# This is an unusual pattern, but it seems to be what the existing code expects.
env = SimulatedEnv()
reset = env.reset
step = env.step
close = env.close
