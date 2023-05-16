import rs5
import numpy as np
import time
import datetime
import cv2
import threading
import os
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

# Ask for input from the user to connect to the Robot
userinput = int(input("Enter 1 to connect to the robot: "))
if userinput == 1:
    rs5.connect_to_robot()
    print("Connected to the robot")
else:
    print("Exiting Python program")
    exit()

action_vector = []
observation_vector = []

cap = cv2.VideoCapture(0)

x = 0
y = 0
z = 500

# Generate a unique filename based on the current date and time
filename = datetime.datetime.now().strftime("%d_%H-%M-%S")

print('Press "3" if done')
print("X: 8 & 2 \nY: 4 & 6 \nZ: 7 & 9")

width = 320
height = 240
fps = int(cap.get(cv2.CAP_PROP_FPS))
buffer_size = int(fps*5)

buffer =FrameBuffer(buffer_size, width, height)

def capture_thread():
    while True:
        ret, frame = cap.read()
        if ret:
            # convert frame to greyscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (320, 240))
            buffer.add_frame(frame)

capture_thread = threading.Thread(target=capture_thread)
capture_thread.start()
            
while True:
    done = False

    #event = keyboard.read_event()
    #direction = int(event.name)
    try:
        direction = int(input("Enter direction: "))
    except:
        continue

    observation_vector.append(np.stack(buffer.get_frames()[-3:]))

    magnitude = 15

    if direction == 2:
        x = x + magnitude
        robtarget = rs5.create_Robtarget(x, y, z)

    elif direction == 8:
        x = x - magnitude
        robtarget = rs5.create_Robtarget(x, y, z)

    elif direction == 6:
        y = y + magnitude
        robtarget = rs5.create_Robtarget(x, y, z)

    elif direction == 4:
        y = y - magnitude
        robtarget = rs5.create_Robtarget(x, y, z)
    
    elif direction == 7:
        z = z + magnitude
        robtarget = rs5.create_Robtarget(x, y, z)

    elif direction == 9:
        z = z - magnitude
        robtarget = rs5.create_Robtarget(x, y, z)

    elif direction == 3:

        cap.release()
        direction = {'obs': observation_vector, 'action': action_vector}
        np.savez("example"+filename, direction)
        print("Exiting Python program")
        sys.exit()
        os.exit()
        break

    else:
        continue

    action_vector.append(direction)

    rs5.robot.set_rapid_variable('WPW', 1)
    rs5.send_Robtarget(robtarget)
    rs5.robot.set_rapid_variable("python_ready", "TRUE")
    rs5.robot.wait_for_rapid()


