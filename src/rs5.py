from urllib import response
from rwsuis import RWS
import time
from requests.auth import HTTPDigestAuth

# Connect to the robot
def connect_to_robot():
    
    global robot 
    base_url = "http://152.94.0.39"
    username = "Default User"
    password = "robotics"
    robot = RWS.RWS(base_url, username, password)
    robot.request_mastership()
    robot.start_RAPID()

def turn_Off_Robot():
    robot.motors_off()
    robot.set_rapid_variable("shutdown_flag", "TRUE")

def create_Robtarget(x, y, z):
    return [x, y, z] 

def send_Robtarget(new_robtarget):
    robot.set_robtarget_translation("puck_robtarget", new_robtarget)

def wait_for_rapid(var='ready_flag'):
    robot.wait_for_rapid()

