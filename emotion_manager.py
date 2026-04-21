import cv2
from hand_detector import HandDetector
from face_detector import FaceDetector

class EmotionManager:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.face_detector = FaceDetector()
        self.last_result = None

    def update(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detección de rostro
        chin_x, chin_y, emotion, face_box, probs, sideeye  = self.face_detector.detectar_rostro_y_emocion(frame, frame_rgb)

        # Detección de gestos
        gesture,  hand_landmarks = self.hand_detector.detectar_gesto(frame_rgb, chin_x, chin_y)

        if gesture:
            return {"nombre": gesture, "face_box": face_box, "probs": probs, "hand_landmarks": hand_landmarks}
        elif sideeye:
            return {"nombre": sideeye, "face_box": face_box, "probs": probs, "hand_landmarks": hand_landmarks}
        elif emotion:
            return { "nombre": emotion, "face_box": face_box, "probs": probs, "hand_landmarks": hand_landmarks}
        
        return {"nombre": None, "face_box": face_box, "probs": probs, "hand_landmarks": hand_landmarks}
