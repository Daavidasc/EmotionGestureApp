import os
import cv2
import mediapipe as mp
from emotion_manager import EmotionManager
import numpy as np

# ✅ Cargar imagen original sin alterar tamaño
def load_image(name):
    path = os.path.join("assets", f"{name.lower()}.png")
    if os.path.exists(path):
        img = cv2.imread(path)
        return img
    return np.zeros((150, 150, 3), dtype=np.uint8)

def main():
    cap = cv2.VideoCapture(0)
    manager = EmotionManager()
    mp_drawing = mp.solutions.drawing_utils
    icon = np.zeros((150, 150, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # espejo horizontal
        frame = cv2.flip(frame, 1)
        result = manager.update(frame)

        # DIBUJAR ROSTRO
        if result["face_box"] is not None:
            x, y, w, h = result["face_box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # DIBUJAR MANOS
        if result["hand_landmarks"]:
            for hand_landmarks in result["hand_landmarks"]:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
                )


        if result["nombre"]:
            nombre = result["nombre"]
            cv2.putText(frame, f"Gesto: {result['nombre']}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            icon = load_image(nombre)

        if result["probs"]:
            start_y = 100
            cv2.putText(frame, "EMOCIONES:", (10, start_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for i, (emo, val) in enumerate(result["probs"].items()):
                cv2.putText(frame, f"{emo}: {val:.2f}",
                            (10, start_y + 20*(i+1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        h_frame = frame.shape[0]
        h_icon, w_icon = icon.shape[:2]

        if h_icon > 0 and w_icon > 0:
            scale = h_frame / h_icon
            new_w = int(w_icon * scale)
            new_h = h_frame

            # usa el método original (sin cambiar interpolador)
            icon_resized = cv2.resize(icon, (new_w, new_h))

            # unir ambos frames horizontalmente
            combined = np.hstack((frame, icon_resized))
        else:
            combined = frame  # por si no hay icono cargado

        cv2.imshow("Detector de Gestos y Emociones", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
