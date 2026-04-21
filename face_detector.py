import cv2
from collections import deque
from fer import FER
import mediapipe as mp

class FaceDetector:
    def __init__(self, emotion_fps=10, emotion_pad=0.12):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.emotion_detector = FER(mtcnn=False)

        self.history = deque(maxlen=2)
        self.frame_counter = 0
        self.EMOTION_FPS = emotion_fps
        self.EMOTION_PAD = emotion_pad

    def _promediar_probs(self):
        if not self.history:
            return None
        keys = self.history[0].keys()
        return {k: sum(d[k] for d in self.history) / len(self.history) for k in keys}

    def _map_emotion(self, probs):
        if not probs:
            return None
        if probs.get('surprise', 0) >= 0.30:
            return "SORPRENDIDO"
        if probs.get('sad', 0) >= 0.35:
            return "TRISTE"
        if probs.get('happy', 0) >= 0.35:
            return "FELIZ"
        return "NATURAL"
    
    def _detectar_sideeye(self, face_landmarks):
        LEFT_EYE_OUTER = 33
        LEFT_EYE_INNER = 133
        RIGHT_EYE_OUTER = 362
        RIGHT_EYE_INNER = 263
        LEFT_PUPIL = 468
        RIGHT_PUPIL = 473

        OjoIzqExter = face_landmarks.landmark[LEFT_EYE_OUTER]
        OjoIzqInter = face_landmarks.landmark[LEFT_EYE_INNER]
        OjoDerExter = face_landmarks.landmark[RIGHT_EYE_OUTER]
        OjoDerIntern = face_landmarks.landmark[RIGHT_EYE_INNER]
        PupilaIzq = face_landmarks.landmark[LEFT_PUPIL]
        PipilaDer = face_landmarks.landmark[RIGHT_PUPIL]

        left_ratio = (PupilaIzq.x - OjoIzqExter.x) / (OjoIzqInter.x - OjoIzqExter.x)
        right_ratio = (PipilaDer.x - OjoDerExter.x) / (OjoDerIntern.x - OjoDerExter.x)

        gaze_ratio = (left_ratio + right_ratio) / 2.0

        if gaze_ratio > 0.60:
            return "SIDEEYE"
        else:
            return None

    def detectar_rostro_y_emocion(self, frame, frame_rgb):
        height, width = frame.shape[:2]
        self.frame_counter += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(100, 100))
        face_box = faces[0] if len(faces) == 1 else None

        # --- Detección de barbilla con FaceMesh ---
        face_result = self.face_mesh.process(frame_rgb)
        chin_x = chin_y = None
        if face_result.multi_face_landmarks:
            chin = face_result.multi_face_landmarks[0].landmark[152]
            chin_x, chin_y = chin.x, chin.y
            sideeye = self._detectar_sideeye(face_result.multi_face_landmarks[0])
        else:
            sideeye = None
        # --- Emoción ---
        emotion = None
        prom_emocion = None

        if face_box is not None and (self.frame_counter % self.EMOTION_FPS == 0):
            x, y, w, h = face_box
            x_pad, y_pad = int(w * self.EMOTION_PAD), int(h * self.EMOTION_PAD)
            x1, y1 = max(0, x - x_pad), max(0, y - y_pad)
            x2, y2 = min(width, x + w + x_pad), min(height, y + h + y_pad)
            face_roi = frame_rgb[y1:y2, x1:x2]

            if face_roi.size > 0:
                detections = self.emotion_detector.detect_emotions(face_roi)
                if detections:
                    probs = detections[0]["emotions"]
                    self.history.append(probs)
                    prom_emocion = self._promediar_probs()
                    emotion = self._map_emotion(prom_emocion)
        elif self.history:
            prom_emocion = self._promediar_probs()
            emotion = self._map_emotion(prom_emocion)

        return chin_x, chin_y, emotion, face_box, prom_emocion, sideeye
