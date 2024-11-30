import cv2
import mediapipe as mp

#Create the landmarks and connections
drawing = mp.solutions.drawing_utils
#Create the face model
face_model = mp.solutions.face_detection

#Configuration of the model
with face_model.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
    #Read the image
    image = cv2.imread("pictures/picture_3.jpg")
    height, width, _ = image.shape

    #Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    #Show the results of the face detection
    print("Face detection:", results.detections)

    #Draw the landmarks and connections
    if results.detections is not None:
        #Making the list of each face
        for detection in results.detections:
            #Create the 6 referece points and the square
            drawing.draw_detection(image, detection,
                                   drawing.DrawingSpec(color=(248, 133, 255), thickness=6, circle_radius=2),
                                   drawing.DrawingSpec(color=(246, 195, 83), thickness=6))
            #Get the bounding box
            #xmin and ymin are the coordinates of the upper left corner
            xmin = int(detection.location_data.relative_bounding_box.xmin * width)
            ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                        #width_box and height_box are the width and height of the bounding box
            width_box = int(detection.location_data.relative_bounding_box.width * width)
            height_box = int(detection.location_data.relative_bounding_box.height * height)
            
            #Draw the bounding box
            cv2.rectangle(image, (xmin, ymin), (xmin + width_box, ymin + height_box), (246, 83, 142), 2)

            #Get the keypoint nose
            x_nose = int(detection.location_data.relative_keypoints[2].x * width)
            y_nose = int(face_model.get_key_point(detection, face_model.FaceKeyPoint.NOSE_TIP).y * height)
            cv2.circle(image, (x_nose, y_nose), 5, (246, 83, 142), cv2.FILLED)

    #Show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
