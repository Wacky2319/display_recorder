from mss import mss
import numpy as np
import cv2
import time
import pprint


class Recorder:
    # constructor
    def __init__(self):
        self.fps = 30  # define fps for recording
        self.screen_shot = None
        self.monitor = None
        self.face_img_map = {}
        self.face_number = 0
        # save for memory conservation
        with mss() as screen_shot:
            self.screen_shot = screen_shot
            self.monitor = screen_shot.monitors[1]
        # for Cascade-classifier
        self.path_cascade = "./etc/cascades/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(self.path_cascade)

    # return image as array
    # image can be shown by using "cv2.imshow" directly
    def record(self):
        return np.array(self.screen_shot.grab(self.monitor))

    # to reduce memory comsumpution and keep overall view
    def resize_image(self, raw_img):
        height = raw_img.shape[0]
        width = raw_img.shape[1]
        return cv2.resize(raw_img, (96, 96))

    # clip faces, using Haar-like feature
    def detect_face(self, raw_img):
        # input_size of the model based on AffectNet is: 3 x 96 x 96
        faces = self.face_cascade.detectMultiScale(raw_img, minSize=(256, 256))
        # if faces are not detected, return
        self.face_number = len(faces)
        if self.face_number == 0:
            return
        # process for each face
        for i, face in enumerate(faces):
            x, y, w, h = face
            self.face_img_map["face_img_" +
                              str(i)] = raw_img[y: y + h, x: x + w]

    def show_as_video(self):
        while True:
            # common process: get image, delete image,
            # resize image, and detect image
            # get image and show it
            raw_img = self.record()
            # cv2.imshow("preview_raw", raw_img)
            # detect faces
            self.detect_face(raw_img)
            # delete unused set of map
            img_map_number = len(self.face_img_map)
            for i in range(img_map_number - self.face_number):
                target_number = self.face_number + i
                del self.face_img_map["face_img_" + str(target_number)]
                cv2.destroyWindow("preview_" + str(target_number))
            # process for each faces
            for i, img_key in enumerate(self.face_img_map):
                img = self.resize_image(self.face_img_map[img_key])
                #  if you want to preview each face image, activate
                cv2.imshow("preview_" + str(i), img)

            # show overview image
            print(self.face_img_map.keys())
            # if ESC is pressed, break and destroy window
            if cv2.waitKey(1) == 27:
                break
            time.sleep(1 / self.fps)  # wait to avoid over running
        cv2.destroyAllWindows()


if __name__ == '__main__':
    recorder = Recorder()
    recorder.show_as_video()
