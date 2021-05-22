import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
import time

def get_box(pred, size):

    width, height = size

    vertical = np.array(pred)[[1, 8, 10, 19]]
    centre = vertical.mean(0).astype(float).tolist()

    horizontal = np.array(pred)[[0, 13, 14, 2, 10, 8, 18, 5, 17, 9]]
    horizontal[:, 1] = height - horizontal[:,1]
    component = PCA(n_components=1).fit(horizontal).components_[0]
    theta = np.arctan2(component[1], component[0])
    if theta < -np.pi/2:
        theta = theta + np.pi
    elif theta > np.pi/2:
        theta = theta - np.pi
    
    out = dict()
    out['centre'] = centre
    out['theta'] = theta

    return out

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

    out = dict()

    while True:  
        # read image 
        success, image = cap.read()
        if not success:
            break
        
        print("frame ", image_count)

        # get predict results 
        preds = data[str(image_count)]

        # get box
        preds = sorted(preds, key=lambda pred: np.array(pred)[:,0].mean(), reverse=True)
        if len(preds):
            pred = preds[0]
            box = get_box(pred, (width, height))
        else:
            box = None
        
        # save box location to output dict 
        out[image_count] = box

        # accumulate counter 
        image_count = image_count + 1

    f = open('./output/box.json', 'w')
    json.dump(out, f)
