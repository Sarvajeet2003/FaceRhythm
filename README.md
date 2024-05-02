**Real-time Emotion-based Music Recommendation System**

This project aims to build a real-time emotion-based music recommendation system using facial expression detection. The system detects the user's emotions through their webcam, suggests songs from Spotify based on their current mood, and plays the top suggested song.

**Features**

- **Real-time Emotion Detection**: Utilizes Haar cascades for face detection and a trained convolutional neural network (CNN) for emotion recognition to detect the user's emotions in real-time.
- **Spotify Integration**: Authenticates with Spotify API to access the user's saved tracks and suggest songs based on their detected emotion.
- **Automatic Song Suggestion**: Filters the user's saved tracks based on the detected emotion and suggests a song with a matching valance.
- **User-friendly Interface**: Provides real-time feedback on the detected emotion and the suggested song, enhancing user experience.

**Installation**

1. Clone the repository:<https://github.com/Sarvajeet2003/FaceRhythm>
2. Install the required dependencies:

**Usage**

1. Run the main scripts:
2. python3 model.py
3. python3 spotify.py
4. Allow access to your webcam when prompted.
5. The program will detect your facial expressions in real-time and suggest a song based on your current mood.

**Configuration**

- Ensure that you have a Spotify developer account and obtain your client ID and client secret.
- Update the client\_id and client\_secret variables in the main.py file with your Spotify credentials.

**Files and Functionality Model.py**

- **Functionality:** Contains code for loading and utilizing the trained emotion detection model.
- **Description:** This file defines functions to load the pre-trained convolutional neural network (CNN) model for emotion recognition. It provides methods to preprocess images, predict emotions from facial expressions, and map predictions to corresponding emotion labels.

**Spotify.py**

- **Functionality:** Handles authentication with the Spotify APIand suggests songs based on the user's detected emotion.
- **Description:** This file encapsulates functionalities related to Spotify integration. It includes functions to authenticate with the Spotify APIusing client credentials flow,obtain an access token, and interact with the Spotify APIto retrieve the

  user's saved tracks. Additionally,it provides a method to suggest songs based on the detected emotion by filtering the user's saved tracks and selecting a song

  with a matching valance.

**Credits**

- Facial expression detection model based on FER2013 dataset trained using Keras.
- Spotify integration using[ Spotipy](https://spotipy.readthedocs.io/en/2.19.0/).
- Emotion labels and color mappings inspired by[ Emotion Recognition GitHub repository](https://github.com/priya-dwivedi/face_and_emotion_detection).
