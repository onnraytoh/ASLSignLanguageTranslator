# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:45:38 2024

@author: admin
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import tensorflow as tf
from time import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Paths
data_dir = 'C:/Users/admin/OneDrive - Sunway Education Group/Desktop/TohOnnRay_20043303/archive/asl_alphabet_train/asl_alphabet_train'
test_dir = 'C:/Users/admin/OneDrive - Sunway Education Group/Desktop/TohOnnRay_20043303/archive/asl_alphabet_test/asl_alphabet_test'
model_save_path = 'C:/Users/admin/OneDrive - Sunway Education Group/Desktop/TohOnnRay_20043303/asl_model.h5'
labels_path = 'C:/Users/admin/OneDrive - Sunway Education Group/Desktop/TohOnnRay_20043303/labels.txt'

# Parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = 20
confidence_threshold = 0.85

# Model Training Function
def train_and_save_model():
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=15,
        brightness_range=(0.8, 1.2),
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    class_labels = list(train_gen.class_indices.keys())
    with open(labels_path, "w") as f:
        for label in class_labels:
            f.write(f"{label}\n")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = True

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation="relu"),
        BatchNormalization(),
        tf.keras.layers.Dense(len(class_labels), activation="softmax"),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights
    )
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

# Test Data Loading Function
def load_test_data(test_dir, img_height, img_width):
    subdirs = [name for name in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, name))]
    if not subdirs:
        print("Error: Test directory structure is invalid.")
        return None

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode="categorical",
        shuffle=False
    )
    return test_gen

# Model Evaluation Function
def evaluate_model(model, test_gen, class_labels):
    if test_gen is None:
        print("Test data not loaded properly.")
        return

    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.85)
mp_drawing = mp.solutions.drawing_utils

# PyQt5 Application
class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Detection")
        self.setGeometry(100, 100, 800, 600)

        if not os.path.exists(model_save_path):
            print("Model not found. Training a new model...")
            train_and_save_model()

        self.model = load_model(model_save_path)
        with open(labels_path, "r") as f:
            self.class_labels = [line.strip() for line in f]

        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.last_predicted_label = None
        self.last_prediction_time = 0
        self.prediction_cooldown = 1.0

        self.video_label = QLabel()
        self.confidence_label = QLabel("Confidence: 0%")
        self.start_button = QPushButton("Start Video Feed")
        self.stop_button = QPushButton("Stop Video Feed")
        self.test_button = QPushButton("Test Model")
        self.visualize_button = QPushButton("Visualize Softmax Output")
        self.quit_button = QPushButton("Quit")  # New Quit button

        self.start_button.clicked.connect(self.start_video_feed)
        self.stop_button.clicked.connect(self.stop_video_feed)
        self.test_button.clicked.connect(self.test_model)
        self.visualize_button.clicked.connect(self.visualize_softmax_output)
        self.quit_button.clicked.connect(self.quit_application)  # Connect quit button to the method

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.test_button)
        layout.addWidget(self.visualize_button)
        layout.addWidget(self.quit_button)  # Add quit button to the layout

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_video_feed(self):
        self.camera = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_video_feed(self):
        # Stop the timer
        self.timer.stop()

        # Release the camera if it's opened
        if self.camera.isOpened():
            self.camera.release()

        # Clear the video display label
        self.video_label.clear()

    def test_model(self):
        test_gen = load_test_data(test_dir, img_height, img_width)
        if test_gen:
            evaluate_model(self.model, test_gen, self.class_labels)

    def process_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w) - 20
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 20
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w) + 20
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h) + 20

                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                hand_roi = rgb_frame[y_min:y_max, x_min:x_max]
                hand_roi_resized = cv2.resize(hand_roi, (img_height, img_width))  # Resize to 224x224
                self.hand_roi_resized = hand_roi_resized  # Store resized ROI

                # Normalize image
                hand_roi_normalized = hand_roi_resized / 255.0
                hand_roi_expanded = np.expand_dims(hand_roi_normalized, axis=0)

                predictions = self.model.predict(hand_roi_expanded)
                confidence = np.max(predictions)
                predicted_label = self.class_labels[np.argmax(predictions)]

                if confidence >= confidence_threshold:
                    current_time = time()
                    if predicted_label != self.last_predicted_label or current_time - self.last_prediction_time > self.prediction_cooldown:
                        self.last_predicted_label = predicted_label
                        self.last_prediction_time = current_time

                confidence_percentage = int(confidence * 100)
                self.confidence_label.setText(f"Confidence: {confidence_percentage}%")
                cv2.putText(frame, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def visualize_softmax_output(self):
        # Method to plot the softmax output
        if self.last_predicted_label is None:
            print("No prediction available to visualize.")
            return

        if hasattr(self, 'hand_roi_resized'):
            hand_roi_expanded = np.expand_dims(self.hand_roi_resized / 255.0, axis=0)
            predictions = self.model.predict(hand_roi_expanded)

            # Plotting the softmax output
            confidence = predictions[0]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(self.class_labels, confidence)
            ax.set_xlabel('Class Labels')
            ax.set_ylabel('Confidence')
            ax.set_title('Softmax Output Probabilities')

            # Show the plot in a new window
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
        else:
            print("Hand ROI is not available for prediction.")

    def quit_application(self):
        print("Exiting application...")
        self.close()  # Close the application window

if __name__ == '__main__':
    app = QApplication([])
    window = SignLanguageApp()
    window.show()
    app.exec_()






