import cv2

cap = cv2.VideoCapture("../video.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rate = cap.get(cv2.CAP_PROP_FPS )
print( length )
print( rate )
print( length / rate / 60)

count = 0
success = True
while success:  
  success, image = cap.read()
  count += 1