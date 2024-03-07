import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

# Define the list of emotion labels
emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Disgust']

# Load and preprocess the dataset
def load_dataset(dataset_path):
    data = []
    labels = []

    for label, emotion in enumerate(emotion_labels):
        expression_folder = f"{dataset_path}/{emotion}/"
        for filename in os.listdir(expression_folder):
            img_path = os.path.join(expression_folder, filename)
            img = load_img(img_path, target_size=(48, 48))
            img_array = img_to_array(img)
            data.append(img_array)
            labels.append(label)

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)

    return data, labels

# Load the dataset
dataset_path = "/Users/sarvajeethuk/Downloads/IR/Project/Face_IR/Face_Emotion_Detector/images"
data, labels = load_dataset(dataset_path)

# Convert labels to one-hot encoding
labels_one_hot = to_categorical(labels, num_classes=len(emotion_labels))

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(emotion_labels), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate the model on the test set
accuracy = model.evaluate(X_val, y_val)[1]
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save("/Users/sarvajeethuk/Downloads/IR/Project/Face_IR/Face_Emotion_Detector/Model")