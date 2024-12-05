import cv2
import mediapipe as mp

f_mesh = mp.solutions.face_mesh
drawing = mp.solutions.drawing_utils
connections_c = mp.solutions.face_mesh.FACEMESH_CONTOURS 

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with f_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5) as face_mesh:

    while True:
        ret, frame = capture.read()

        if ret is False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks is not None:
            for face_lm in results.multi_face_landmarks:
                drawing.draw_landmarks(frame, face_lm, 
                                       connections_c,
                                       drawing.DrawingSpec(color=(255, 133, 248), thickness=4, circle_radius=2),
                                       drawing.DrawingSpec(color=(142, 83, 246), thickness=2))
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

capture.release()
cv2.destroyAllWindows()