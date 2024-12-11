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
    video = cv2.VideoCapture('pictures/video_1.mp4')

    while(video.isOpened()):
        #frame is the video, this has the image
        ret, frame = video.read()

        if not ret:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        video_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(video_rgb)

        #Recocgnize the side of the hand: handedness
        #print("Handedness:", results.multi_handedness)

        #Show the 21 landmarks
        #print("Landmarks:", results.multi_hand_landmarks)

        #If you find at least one hand
        if results.multi_hand_landmarks is not None:

            #Show the landmarks and connections
            for landmarks in results.multi_hand_landmarks:
                #Draw the landmarks and connections in the image 
                drawing.draw_landmarks(frame, landmarks, hands_picture.HAND_CONNECTIONS,
                                       drawing.DrawingSpec(color=(255, 133, 248), thickness=4, circle_radius=2),
                                       drawing.DrawingSpec(color=(255, 203, 88), thickness=2))


        video_show = cv2.flip(frame, 1)
        cv2.imshow("Video", video_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()