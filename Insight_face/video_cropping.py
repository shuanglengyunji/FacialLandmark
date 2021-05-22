import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
import time

if __name__ == '__main__':

    # open result json 
    f = open('output/mouth_result.json')
    data = json.load(f)

    # frame counter 
    image_count = 0

    # get video 
    cap = cv2.VideoCapture("../video.mp4")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rate = int(cap.get(cv2.CAP_PROP_FPS))

    # output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter('./temp/output.mp4', fourcc, rate, (200, 100))

    while True:  
        # read image 
        success, image = cap.read()
        if not success:
            break
        
        print("frame ", image_count)

        # get predict results 
        preds = data[str(image_count)]
        preds = sorted(preds, key=lambda pred: np.array(pred)[:,0].mean(), reverse=True)

        # shift and rotate 
        if len(preds):
            pred = preds[0]
            
            vertical = np.array(pred)[[1, 8, 10, 19]]
            centre = tuple(vertical.mean(0).astype(int))

            horizontal = np.array(pred)[[0, 13, 14, 2, 10, 8, 18, 5, 17, 9]]
            horizontal[:, 1] = height - horizontal[:,1]
            component = PCA(n_components=1).fit(horizontal).components_[0]
            theta = np.arctan2(component[1], component[0])
            if theta < -np.pi/2:
                theta = theta + np.pi
            elif theta > np.pi/2:
                theta = theta - np.pi
            
            M = np.float32([[1, 0, width/2 - centre[0]], [0, 1, height/2 - centre[1]]])
            image = cv2.warpAffine(image, M, (width, height))

            M = cv2.getRotationMatrix2D((width/2, height/2), -theta/np.pi*180.0, 1.0)
            image = cv2.warpAffine(image, M, (1280, 720))
            
            # cv2.imwrite("test.jpg", image)
            # exit()

        # crop image 
        image = image[360-50:360+50, 640-100:640+100, :]

        # write frame
        out.write(image)
        
        # accumulate counter 
        image_count = image_count + 1

    out.release()