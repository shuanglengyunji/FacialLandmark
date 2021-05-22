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
    out = cv2.VideoWriter('./temp/output.mp4', fourcc, rate, (width, height))

    while True:  
        # read image 
        success, image = cap.read()
        if not success:
            break
        
        print("frame ", image_count)

        # get predict results 
        preds = data[str(image_count)]

        preds = sorted(preds, key=lambda pred: np.array(pred)[:,0].mean(), reverse=True)

        # # plot on image 
        if len(preds):
            pred = preds[0]
            
            # calculate 
            vertical = np.array(pred)[[1, 8, 10, 19]]
            vertical[:, 1] = height - vertical[:, 1]
            horizontal = np.array(pred)[[0, 13, 14, 2, 10, 8, 18, 5, 17, 9]]
            horizontal[:, 1] = height - horizontal[:,1]

            middle = vertical.mean(0)
            component = PCA(n_components=1).fit(horizontal).components_[0]
            theta = np.arctan2(component[1], component[0])
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            corners = (R @ np.array([[100, 50], [100, -50], [-100, 50], [-100, -50]]).T).T
            corners[:,0] = corners[:,0] + middle[0]
            corners[:,1] = corners[:,1] + middle[1]

            # plt.figure()
            # plt.plot(middle[0], middle[1], 'k.')
            # plt.plot(corners[:, 0], corners[:, 1], 'r.')
            # plt.xlim([0, width])
            # plt.ylim([0, height])
            # plt.savefig('plot.png')
            # plt.close()

            # plot middle point 
            middle_point = np.array([middle[0], height - middle[1]]).astype(int)
            cv2.circle(image, tuple(middle_point), 6, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)

            # plot corner point 
            corner_points = corners
            corner_points[:, 1] = height - corner_points[:, 1]
            corner_points = corner_points.astype(int)
            for p in corner_points:
                cv2.circle(image, tuple(p), 6, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)

            # plot all points 
            for p in np.round(pred).astype(int):
                cv2.circle(image, tuple(p), 3, (255, 0, 0), cv2.FILLED, cv2.LINE_AA)

            # cv2.imwrite("test.jpg", image)

        # write frame
        out.write(image)
        
        # accumulate counter 
        image_count = image_count + 1

    out.release()