import random
import webbrowser
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import base64
import cv2
import time
import numpy as np
import requests
from keras.models import load_model

# Load your own trained model
your_model = load_model("/Users/sarvajeethuk/Downloads/IR/Project/Face_IR/Face_Emotion_Detector/Model.h5")

emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Fear']
client_id = 'e98bde463dc04b159ac2edcf437130f4'
client_secret = '767cffb3d51347738f7a191cc3e53cf8'
redirect_uri = "http://localhost:8080/callback/"


# Function to authenticate with Spotify and obtain an access token
def get_access_token(client_id, client_secret):
    url = 'https://accounts.spotify.com/api/token'
    headers = {'Authorization': 'Basic ' + base64.b64encode((client_id + ':' + client_secret).encode()).decode()}
    payload = {'grant_type': 'client_credentials'}
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        print("Error in obtaining access token:", response.text)
        return None
# Initialize Spotipy with OAuth2 authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,client_secret=client_secret,redirect_uri=redirect_uri,scope="user-library-read"))

# Function to suggest songs based on user's saved tracks and detected emotion
def suggest_songs(emotion):
    # Get user's saved tracks
    saved_tracks = sp.current_user_saved_tracks()
    if saved_tracks:
        random.shuffle(saved_tracks['items'])
        # Filter tracks based on detected emotion
        for track in saved_tracks['items']:
            track_name = track['track']['name']
            artist_name = track['track']['artists'][0]['name']
            track_id = track['track']['id']
            
            # Fetch track features (including audio features like valence, energy, etc.)
            track_features = sp.audio_features(track_id)
            if track_features:
                valence = track_features[0]['valence']
                # Based on the emotion, filter tracks with suitable valence (a measure of positiveness)
                if emotion == 'Happy' and valence > 0.5:
                    print(f"Suggested {emotion} song: {track_name} by {artist_name}")
                    top_track_uri = track['track']['uri']
                    webbrowser.open(top_track_uri)
                    break  # Play only the top suggested song
                elif emotion == 'Sad' and valence < 0.5:
                    print(f"Suggested {emotion} song: {track_name} by {artist_name}")
                    top_track_uri = track['track']['uri']
                    webbrowser.open(top_track_uri)
                    break  # Play only the top suggested song
                elif emotion == 'Angry' and valence < 0.5:
                    print(f"Suggested {emotion} song: {track_name} by {artist_name}")
                    top_track_uri = track['track']['uri']
                    webbrowser.open(top_track_uri)
                    break  # Play only the top suggested song
                elif emotion == 'Surprise' and valence > 0.5:
                    print(f"Suggested {emotion} song: {track_name} by {artist_name}")
                    top_track_uri = track['track']['uri']
                    webbrowser.open(top_track_uri)
                    break  # Play only the top suggested song
                elif emotion == 'Fear' and valence < 0.5:
                    print(f"Suggested {emotion} song: {track_name} by {artist_name}")
                    top_track_uri = track['track']['uri']
                    webbrowser.open(top_track_uri)
                    break  # Play only the top suggested song
                elif emotion == 'Neutral':
                    print(f"Suggested {emotion} song: {track_name} by {artist_name}")
                    top_track_uri = track['track']['uri']
                    webbrowser.open(top_track_uri)
                    break  # Play only the top suggested song
                # Add conditions for other emotions if needed
    else:
        print("No saved tracks found.")

emotion_colors = {
    'Angry': (0, 0, 255),    # Red
    'Fear': (255, 0, 0),     # Blue
    'Happy': (255, 255, 0),  # Yellow
    'Sad': (0, 255, 255),    # Cyan
    'Surprise': (255, 0, 255),  # Magenta
    'Neutral': (128, 128, 128)  # Gray
}
def detect_emotion():
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
    
            elapsed_time = time.time() - start_time
            if elapsed_time >= 5:
                if emotion_summary:
                    emotion_summary_counts = {label: emotion_summary.count(label) for label in set(emotion_summary)}
                    dominant_emotion = max(emotion_summary_counts, key=emotion_summary_counts.get)
                    print("Emotion Summary (last 5 seconds):")
                    for label, count in emotion_summary_counts.items():
                        print(f"{label}: {count}")
                    print(f"Dominant Emotion: {dominant_emotion}")
    
                    suggest_songs(dominant_emotion)
                    break
                else:
                    print("No faces detected in the last 5 seconds.")
                start_time = time.time()
                emotion_summary = []
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        except Exception as e:
            print(f"Error: {e}")
            break
    
    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotion()
