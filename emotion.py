import cv2
import time
from deepface import DeepFace

model = DeepFace.build_model("Emotion")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_colors = {
    'angry': (0, 0, 255),    # Red
    'disgust': (0, 255, 0),  # Green
    'fear': (255, 0, 0),     # Blue
    'happy': (255, 255, 0),  # Yellow
    'sad': (0, 255, 255),    # Cyan
    'surprise': (255, 0, 255),  # Magenta
    'neutral': (128, 128, 128)  # Gray
}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

start_time = time.time()
frame_count = 0
resized_face = None

while True:
    try:
        ret, frame = cap.read()
        if not ret or frame.shape[0] == 0 or frame.shape[1] == 0:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)
            preds = model.predict(reshaped_face)[0]
            emotion_idx = preds.argmax()
            emotion = emotion_labels[emotion_idx]

            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_colors[emotion], 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_colors[emotion], 2)

        # Display the frame
        cv2.imshow('Real-time Emotion Detection', frame)

        # Display the Face ROI separately
        if resized_face is not None:
            cv2.imshow('Face ROI', resized_face)

        # FPS calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

cap.release()
cv2.destroyAllWindows()