import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
import os
import time

fps = 20
counter = 0

while True:
    img = np.asarray(ImageGrab.grab(), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    height = img.shape[0]
    width = img.shape[1]
    img = cv2.resize(img, (int(width/4), int(height/4)))
    cv2.imshow("screen", img)
    # press ESC to exit
    if cv2.waitKey(1) == 27:
        break
    # time.sleep(1/fps)

cv2.destroyAllWindows()
