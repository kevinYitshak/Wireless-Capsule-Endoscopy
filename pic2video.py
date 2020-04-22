#pic2video.py

import cv2
import os
import random 
from numba import jit

image_folder = 'C:/Users/kevin/Documents/IISc/WCE/Video/'
video_name = 'WCE_video.avi'

images = [img for img in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, img))]
# print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width, height))

# @jit(nopython=True)
def img2video():
    
    for _ in range(len(images)):
        image = random.choice(images)
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    img2video()

