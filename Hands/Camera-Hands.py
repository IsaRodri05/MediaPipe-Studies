import cv2
import mediapipe as mp

drawing  = mp.solutions.drawing_utils
hands_model = mp.solutions.hands

#Activate the camera
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Configuration of the model
with hands_model.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.5) as hands:

    while True:
        #Verify if the camera is working
        #ret is a boolean that returns True if the frame is read correctly
        #frame is the image that is captured
        ret, frame = capture.read()

        if ret is False:
            break
        
        #Take the height, width and channels of the frame and RGB
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Process the image
        results = hands.process(frame_rgb)

        #Recocgnize the side of the hand: handedness
        #print("Handedness:", results.multi_handedness)

        if results.multi_hand_landmarks is not None:

        #Show the landmarks and connections
            for landmarks in results.multi_hand_landmarks:
                #Draw the landmarks and connections in the image 
                drawing.draw_landmarks(frame, landmarks, hands_model.HAND_CONNECTIONS,
                                        drawing.DrawingSpec(color=(255, 133, 248), thickness=4, circle_radius=2),
                                        drawing.DrawingSpec(color=(88, 203, 255), thickness=2))

        #Show the frame
        cv2.imshow("Frame", frame)
        #Enable the esc button
        if cv2.waitKey(1) & 0xFF == 27:
            break

capture.release()
cv2.destroyAllWindows()