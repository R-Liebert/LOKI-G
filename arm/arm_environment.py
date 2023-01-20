import time


class ArmEnvironment():
    def __init__(self):
        from picamera import PiCamera
        
        self.camera = PiCamera()
        time.sleep(2)
        self.camera.resolution = (3280, 2464)
        self.camera.vflip = True

        pass
    def step(self, action):
        pass
    def reset(self):
        pass
