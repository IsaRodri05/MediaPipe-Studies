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

    #Read the image
    image = cv2.imread("pictures/picture_5.jpg")
    height, width, _ = image.shape
    image = cv2.flip(image, 1)

    #Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    #Recocgnize the side of the hand: handedness
    print("Handedness:", results.multi_handedness)

    #Show the 21 landmarks
    print("Landmarks:", results.multi_hand_landmarks)

    #If you find at least one hand
    if results.multi_hand_landmarks is not None:
        #Index of the finger tips
        indexs = [12, 16, 20]

        #Show the landmarks and connections
        for landmarks in results.multi_hand_landmarks:
            #Draw the landmarks and connections in the image 
            drawing.draw_landmarks(image, landmarks, hands_picture.HAND_CONNECTIONS,
                                   drawing.DrawingSpec(color=(255, 133, 248), thickness=4, circle_radius=2),
                                   drawing.DrawingSpec(color=(88, 203, 255), thickness=2))
            
            #FIRST WAY TO ACCESS THE LANDMARKS
            #Accessing some landmarks according with their names. We need to multiply by the width and height of the image and convert to int
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

            print("Thumb tip x and y:", x1, y1)
            print("Index finger tip x and y:", x2, y2)
            print("Middle finger tip x and y:", x3, y3)
            print("Ring finger tip x and y:", x4, y4)
            print("Pinky tip x and y:", x5, y5)
            print("")

            cv2.circle(image, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
            cv2.circle(image, (x2, y2), 4, (255, 0, 0), cv2.FILLED)

            #SECOND WAY TO ACCESS THE LANDMARKS
            for (i, points) in enumerate(landmarks.landmark):
                if i in indexs:
                    x = int(points.x * width)
                    y = int(points.y * height)
                    cv2.circle(image, (x, y), 4, (255, 255, 0), cv2.FILLED)

    image = cv2.flip(image, 1)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()