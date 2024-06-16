import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Kamera
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Zamiana BGR w RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        # Detekcja twarzy
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Obliczenie punktu centralnego (czoło)
                cx, cy = x + w // 2, y + h // 26

                # Narysowany celownik
                cv2.drawMarker(image, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, 
                               markerSize=100, thickness=2)
                radius = max(w, h) // 6
                cv2.circle(image, (cx, cy), radius, (0, 0, 255), 2)

        # Pokaż obraz
        cv2.imshow('Celowanie', image)
        # Wyłącz program przy pomocy Esc
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
