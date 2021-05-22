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

    out = dict()
    for result in result_list:
        print("Reading: ", result)
        f = open(result[1])
        data = json.load(f)
        for item in data.items():
            id = item[0]
            mouth = list(map(lambda x : x[52:72], item[1]))
            out[id] = mouth

    f = open('./output/mouth_result.json', 'w')
    json.dump(out, f)

        