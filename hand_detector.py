import mediapipe as mp
import math

class HandDetector:
    def __init__(self, EPS=0.015, CHIN_THRESHOLD=0.05):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.CHIN_THRESHOLD = CHIN_THRESHOLD

    def distancia_norm(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def detectar_gesto(self, frame_rgb, chin_x=None, chin_y=None):

        result = self.hands.process(frame_rgb)
        detected_gesture = None
        landmarks_to_draw = result.multi_hand_landmarks

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                all_y = [lm.y for lm in hand_landmarks.landmark]
                all_x = [lm.x for lm in hand_landmarks.landmark]
                thumb_tip_y = hand_landmarks.landmark[4].y
                thumb_tip_x = hand_landmarks.landmark[4].x

                # LIKE
                if thumb_tip_y <= min(all_y):
                    detected_gesture = "LIKE"
                    break
                # DISLIKE
                elif thumb_tip_y >= max(all_y):
                    detected_gesture = "DISLIKE"
                    break
                # LADO
                elif thumb_tip_x <= min(all_x):
                    detected_gesture = "COSTADO"
                    break
                # PENSANDO
                elif chin_x is not None:
                    dist = self.distancia_norm(thumb_tip_x, thumb_tip_y, chin_x, chin_y)
                    if dist < self.CHIN_THRESHOLD:
                        detected_gesture = "PENSANDO"
                        break

        return detected_gesture, landmarks_to_draw
