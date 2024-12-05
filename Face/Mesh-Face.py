import cv2
import mediapipe as mp

f_mesh = mp.solutions.face_mesh
drawing = mp.solutions.drawing_utils
connections_c = mp.solutions.face_mesh.FACEMESH_CONTOURS #Connect the face, mouth, eyes, eyesbrow contour
connections_t = mp.solutions.face_mesh.FACEMESH_TESSELATION #Connect all the landmarks
connections_i = mp.solutions.face_mesh.FACEMESH_IRISES #Connect the eyes    

#Use with images
with f_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    min_detection_confidence=0.5) as face_mesh:

    image = cv2.imread("pictures/picture_4.jpg")
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    #index for some landmarks
    indexes = [4, 133, 362]

    print("FaceLandmarks: ", results.multi_face_landmarks)

    if results.multi_face_landmarks is not None:
        for face_lm in results.multi_face_landmarks:
            drawing.draw_landmarks(image, face_lm, 
                                   connections_c,
                                   drawing.DrawingSpec(color=(255, 133, 248), thickness=4, circle_radius=2),
                                   drawing.DrawingSpec(color=(142, 83, 246), thickness=2))
            for index in indexes:
                x_lm = int(face_lm.landmark[index].x * width)
                y_lm = int(face_lm.landmark[index].y * height)
                cv2.circle(image, (x_lm, y_lm), 5, (246, 83, 142), cv2.FILLED)

            
    cv2.imshow("Image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()