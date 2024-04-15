from playsound import playsound
from pytube import YouTube
import cv2
import time
import numpy as np
from keras.models import load_model


# Define a dictionary of mood-based song recommendations with YouTube links
mood_songs = {
    'Angry': ['https://www.youtube.com/watch?v=VAJK04HOLd0&pp=ygURYW5ncnkgaGluZGkgc29uZ3M%3D'],
    'Happy': ['https://www.youtube.com/watch?v=z3t4Ylb_e30&list=TLPQMTUwMzIwMjSjVGMiP4eq7w&index=1'],
    'Sad': ['https://www.youtube.com/watch?v=ktPD6TMovxs&pp=ygUHaHVtbmF2YQ%3D%3D'],
    'Surprise': ['https://www.youtube.com/watch?v=MJDIDTb2Zwk&list=RDMJDIDTb2Zwk&start_radio=1'],
    'Neutral': ['https://www.youtube.com/watch?v=3PKlWvQVm2A&pp=ygULaGluZGkgc29uZ3M%3D'],
    'Fear': ['https://www.youtube.com/watch?v=Gl5wBrNuxx4&pp=ygUVZmVhciBnZW5yZSBoaW5kaSBzb25n'],
    'Disgust': ['https://www.youtube.com/watch?v=ZETfvb4hrnw&pp=ygUYZGlzZ3VzdCBnZW5yZSBoaW5kaSBzb25n']
}

# Function to play a YouTube video
def play_youtube(video_url):
    yt = YouTube(video_url)
    stream = yt.streams.get_audio_only()
    audio_file = stream.download(output_path='/Users/sarvajeethuk/Downloads/IR/Project/Face_IR/Face_Emotion_Detector/Songs')  # Download the audio
    
    # Play the downloaded audio file
    playsound(audio_file)

# Load your own trained model
your_model = load_model("/Users/sarvajeethuk/Downloads/IR/Project/Face_IR/Face_Emotion_Detector/Model")

emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Disgust']
emotion_colors = {
    'Angry': (0, 0, 255),    # Red
    'Fear': (255, 0, 0),     # Blue
    'Happy': (255, 255, 0),  # Yellow
    'Sad': (0, 255, 255),    # Cyan
    'Surprise': (255, 0, 255),  # Magenta
    'Neutral': (128, 128, 128)  # Gray
}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

start_time = time.time()
emotion_summary = []
resized_face = None  # Initialize resized_face outside the loop

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
            
            # Convert to 3 channels (grayscale to RGB)
            resized_face = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2RGB)
            
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(1, 48, 48, 3)
            preds = your_model.predict(reshaped_face)
            emotion_idx = preds.argmax()
            emotion = emotion_labels[emotion_idx]

            # Append the predicted emotion to the summary list
            emotion_summary.append(emotion)
            
            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_colors[emotion], 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_colors[emotion], 2)

        # Display the frame
        cv2.imshow('Real-time Emotion Detection', frame)

        # Display the Face ROI separately
        if resized_face is not None:
            cv2.imshow('Face ROI', resized_face)

        # Calculate elapsed time and check if 5 seconds have passed
        elapsed_time = time.time() - start_time
        if elapsed_time >= 5:
            if emotion_summary:
                # Analyze the collected emotions and provide a summary
                emotion_summary_counts = {label: emotion_summary.count(label) for label in set(emotion_summary)}
                dominant_emotion = max(emotion_summary_counts, key=emotion_summary_counts.get)
                print("Emotion Summary (last 5 seconds):")
                for label, count in emotion_summary_counts.items():
                    print(f"{label}: {count}")

                print(f"Dominant Emotion: {dominant_emotion}")

                # Print recommended songs based on the dominant emotion
                print("Recommended songs based on your mood:")
                if dominant_emotion in mood_songs:
                    for song_url in mood_songs[dominant_emotion]:
                        play_youtube(song_url)
                else:
                    print("No recommendations available for this mood.")

            else:
                print("No faces detected in the last 5 seconds.")

            # Reset variables for the next 5-second window
            start_time = time.time()
            emotion_summary = []

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break
cap.release()
cv2.destroyAllWindows()