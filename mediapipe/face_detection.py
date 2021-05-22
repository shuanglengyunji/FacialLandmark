import cv2
import progressbar
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# get video 
cap = cv2.VideoCapture("../../video.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rate = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('../../output.mp4', fourcc, rate, (width, height))

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.9, 
    min_tracking_confidence=0.9,
    max_num_faces=1,
    ) as face_mesh:

    for i in progressbar.progressbar(range(length)):
    # while True:
        success, image = cap.read()
        if not success:
            print("Video processing complete")
            break

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                    )
        # cv2.imshow('MediaPipe FaceMesh', image)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break
        out.write(image)

cap.release()
