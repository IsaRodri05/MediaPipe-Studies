import cv2
import mediapipe as mp

#Create the landmarks and connections
drawing = mp.solutions.drawing_utils
#Create the hands model
hands_picture = mp.solutions.hands

#Configuration of the model
with hands_picture.Hands(
    static_image_mode=True,
    max_num_hands=3,
    min_detection_confidence=0.5) as hands:

    #Read the video: connection with the file
    video = cv2.VideoCapture('pictures/video_2.mp4')

    while(video.isOpened()):
        #frame is the video, this has the image
        ret, frame = video.read()

        if not ret:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        video_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(video_rgb)

        #If you find at least one hand
        if results.multi_hand_landmarks is not None:

            #Show the landmarks and connections
            for index, landmarks in enumerate(results.multi_hand_landmarks):
                #Draw the landmarks and connections in the image 
                drawing.draw_landmarks(frame, landmarks, hands_picture.HAND_CONNECTIONS,
                                       drawing.DrawingSpec(color=(255, 133, 248), thickness=4, circle_radius=2),
                                       drawing.DrawingSpec(color=(255, 203, 88), thickness=2))
                
                #Accessing landmarks coordinates
                x1 = int(landmarks.landmark[hands_picture.HandLandmark.THUMB_TIP].x * width)
                y1 = int(landmarks.landmark[hands_picture.HandLandmark.THUMB_TIP].y * height)

                x2 = int(landmarks.landmark[hands_picture.HandLandmark.INDEX_FINGER_TIP].x * width)
                y2 = int(landmarks.landmark[hands_picture.HandLandmark.INDEX_FINGER_TIP].y * height)

                x3 = int(landmarks.landmark[hands_picture.HandLandmark.MIDDLE_FINGER_TIP].x * width)
                y3 = int(landmarks.landmark[hands_picture.HandLandmark.MIDDLE_FINGER_TIP].y * height)

                x4 = int(landmarks.landmark[hands_picture.HandLandmark.RING_FINGER_TIP].x * width)
                y4 = int(landmarks.landmark[hands_picture.HandLandmark.RING_FINGER_TIP].y * height)

                x5 = int(landmarks.landmark[hands_picture.HandLandmark.PINKY_TIP].x * width)
                y5 = int(landmarks.landmark[hands_picture.HandLandmark.PINKY_TIP].y * height)

                print(results.multi_handedness[index].classification[0].label)
                #Euclidean distance between the thumb tip and the index finger tip
                dist_thumb_index = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                print("thumb and finger:", dist_thumb_index)

                #Euclidean distance between the index tip and the middle finger tip
                dist_index_middle = ((x2 - x3)**2 + (y2 - y3)**2)**0.5
                print("index and middle:", dist_index_middle)

                #Euclidean distance between the middle tip and the ring finger tip
                dist_middle_ring = ((x3 - x4)**2 + (y3 - y4)**2)**0.5
                print("middle and ring:", dist_middle_ring)

                #Euclidean distance between the ring tip and the pinky finger tip
                dist_ring_pinky = ((x4 - x5)**2 + (y4 - y5)**2)**0.5
                print("ring and pinky:", dist_ring_pinky)

                print("")


        video_show = cv2.flip(frame, 1)
        cv2.imshow("Video", video_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()