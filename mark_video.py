import glob
import cv2
import numpy as np
import json
import time

if __name__ == '__main__':

    # get result json list 
    results = glob.glob('./output/*.json')
    indexes = [int(''.join(filter(str.isdigit, result))) for result in results]
    result_list = list(zip(indexes, results))
    result_list.sort(key=lambda tup: tup[0]) 

    # get video 
    cap = cv2.VideoCapture("../video.mp4")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rate = int(cap.get(cv2.CAP_PROP_FPS))

    image_count = 0

    # output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter('./temp/output.mp4', fourcc, rate, (width, height))

    for result in result_list:
        f = open(result[1])
        data = json.load(f)

        print("Result ", result[0])
        
        while True:  
            # read image 
            success, image = cap.read()
            if not success:
                break
            
            # get predict results 
            preds = data[str(image_count)]

            # plot on image 
            for count, pred in enumerate(preds):    
                pred = np.round(pred[52:72]).astype(np.int64)     # extract mouth area
                for pr in pred:
                    p = tuple(pr)
                    cv2.circle(image, p, 3, (255, 0, 0), cv2.FILLED, cv2.LINE_AA)

            # write frame
            out.write(image)

            # accumulate counter 
            image_count = image_count + 1

            # go to next result 
            if image_count > result[0]:
                break

    out.release()