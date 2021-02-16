from mss import mss
import numpy as np
import cv2
import time


class Recorder:
    # constructor
    def __init__(self):
        self.fps = 60  # define fps for recording
        self.screen_shot = None
        self.monitor = None
        # save for memory conservation
        with mss() as screen_shot:
            self.screen_shot = screen_shot
            self.monitor = screen_shot.monitors[1]

    # return image as array
    # image can be shown by using cv2.imshow directly
    def record(self):
        return np.array(self.screen_shot.grab(self.monitor))

    def resize_image(self, raw_img):
        height = raw_img.shape[0]
        width = raw_img.shape[1]
        return cv2.resize(raw_img, (int(width/4), int(height/4)))

    def show_as_video(self):
        while True:
            raw_img = self.record()
            img = self.resize_image(raw_img)
            cv2.imshow("preview", img)
            # if ESC is pressed, break and destroy window
            if cv2.waitKey(1) == 27:
                break
            time.sleep(1 / self.fps)  # wait to avoid over running
        cv2.destroyAllWindows()


if __name__ == '__main__':
    recorder = Recorder()
    recorder.show_as_video()
