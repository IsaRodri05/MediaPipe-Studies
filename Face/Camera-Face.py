import cv2
import mediapipe as mp

drawing = mp.solutions.drawing_utils
face_model = mp.solutions.face_detection

#Activate the camera
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Configuration of the model
with face_model.FaceDetection(
    min_detection_confidence=0.5) as face_detection:

    while True:
        ret, frame = capture.read()
        if ret is False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(frame_rgb)

        if results.detections is not None:
            for detection in results.detections:
                drawing.draw_detection(frame, detection,
                                       drawing.DrawingSpec(color=(248, 133, 255), thickness=6, circle_radius=2),
                                       drawing.DrawingSpec(color=(246, 195, 83), thickness=6))
                
                xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                width_box = int(detection.location_data.relative_bounding_box.width * width)
                height_box = int(detection.location_data.relative_bounding_box.height * height)
                cv2.rectangle(frame, (xmin, ymin), (xmin + width_box, ymin + height_box), (246, 83, 142), 2)

                x_nose = int(detection.location_data.relative_keypoints[2].x * width)
                y_nose = int(face_model.get_key_point(detection, face_model.FaceKeyPoint.NOSE_TIP).y * height)
                cv2.circle(frame, (x_nose, y_nose), 5, (246, 83, 142), cv2.FILLED)   
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

capture.release()
cv2.destroyAllWindows()