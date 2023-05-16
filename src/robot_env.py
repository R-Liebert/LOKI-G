import gym
from gym import spaces
import rs5
import numpy as np
import cv2
import threading
import sys

class FrameBuffer:
    def __init__(self, size, width, height):
        self.buffer = np.zeros((size, height, width), dtype=np.uint8)
        self.index = 0
    def add_frame(self, frame):
        self.buffer[self.index] = frame
        self.index = (self.index + 1) % len(self.buffer)
    def get_frames(self):
        if self.index == 0:
            # If the buffer is full, return all frames
            return self.buffer
        else:
            # Concatenate the two slices of the buffer separately
            first_slice = self.buffer[self.index:]
            second_slice = self.buffer[:self.index]
            # Concatenate the resulting arrays along the first dimension
            return np.concatenate((first_slice, second_slice), axis=0)

class RobotArmEnv(gym.Env):
    def __init__(self):
        super(RobotArmEnv, self).__init__()

        # Define action space
        self.action_space = spaces.Discrete(10)

        # Define observation space
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8)

        self._connect_to_robot()
        self.connected = False

        self.cap = cv2.VideoCapture(0)
        self.width = 320
        self.height = 240
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.buffer_size = int(self.fps*5)
        self.buffer = FrameBuffer(self.buffer_size, self.width, self.height)
        self.current_position = (0, 0, 500)

        self.action_history = []
        self.std_thresh = 0.4 # Threshold for standard deviation of action history

        self.emergency_stop = False

        self.capture_thread = threading.Thread(target=self._capture_thread)
        self.capture_thread.start()

    def _connect_to_robot(self):
        userinput = int(input("Enter 1 to connect to the robot: "))
        if userinput == 1:
            rs5.connect_to_robot()
            self.connected = True
            print("Connected to the robot")
        else:
            print("Exiting Python program")
            exit()
        return

    def _capture_thread(self):
        while True:
                ret, frame = self.cap.read()
                if ret:
                    # convert frame to greyscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.resize(frame, (320, 240))
                    self.buffer.add_frame(frame)


    def step(self, action):
        x, y, z = self._action_to_position(action)
        robtarget = rs5.create_Robtarget(x, y, z)
        rs5.robot.set_rapid_variable('WPW', 1)
        rs5.send_Robtarget(robtarget)
        rs5.robot.set_rapid_variable("python_ready", "TRUE")
        rs5.robot.wait_for_rapid()
        if self.action_history == []:
            self.action_history = [action]*5

        self.action_history.append(action)
        self.action_history = self.action_history[-5:]

        observation = np.stack(self.buffer.get_frames()[-3:])
        observation = np.moveaxis(observation, 0, -1)

        reward = self._calculate_reward(action)
        done = self._check_if_done()
        if done and not self.emergency_stop: reward = reward + 300

        return observation, reward, done, {}

    def _action_to_position(self, action):
        x, y, z = self.current_position
        magnitude = 15

        if action == 0:  # Original direction 2
            x = x + magnitude

        elif action == 1:  # Original direction 8
            x = x - magnitude

        elif action == 3:  # Original direction 6
            y = y + magnitude

        elif action == 2:  # Original direction 4
            y = y - magnitude

        elif action == 4:  # Original direction 7
            z = z + magnitude

        elif action == 5:  # Original direction 9
            z = z - magnitude
        
        else:
            x, y, z = self.current_position

        self.current_position = (x, y, z)
        return x, y, z


    def _calculate_reward(self, action):
        # Decrease the reward by 1 every step
        reward = 10
        key = 0
        # Define the function to listen for key press
        try:
            key = int(input("1: Small bonus, 2: Small penalty, 3: Stop, 4: Big reward:  "))
        except:
            pass
        
        if key == 1:
            reward = reward + 20
        elif key == 2:
            reward = reward - 20
        elif key == 3:
            self.emergency_stop = True
            reward = reward - 500
        elif key == 4:
            reward = reward + 100
        else:
            pass


        std = np.std(self.action_history)
        reward = reward + 3 if std < self.std_thresh else reward


        return reward


    def _check_if_done(self):
        done = False
        key = 0
        # Define the function to listen for key press
        try:
            key = int(input("Press 3 if done: "))
        except:
            pass

        if key == 3:
            done = True
        else:
            pass
            
        return done

    def reset(self):
        # Reset the environment to its initial state
        self._connect_to_robot()

        vect = np.stack(self.buffer.get_frames()[-3:])
        vect = np.moveaxis(vect, 0, -1)
        self.current_position = 0, 0, 500
        x, y, z = self.current_position
        robtarget = rs5.create_Robtarget(x, y, z)
        rs5.robot.set_rapid_variable('WPW', 1)
        rs5.send_Robtarget(robtarget)
        rs5.robot.set_rapid_variable("python_ready", "TRUE")
        rs5.robot.wait_for_rapid()

        return vect

    def close(self):
        self.cap.release()
        self.connected = False
