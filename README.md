# Digit Recognizer – Handwritten Digit Classification

A convolutional neural network (CNN) implemented in TensorFlow/Keras for classifying handwritten digits using the MNIST dataset. It also supports prediction from custom grayscale images.

## Features

\- Trained on the MNIST dataset with 60,000 training and 10,000 test images  
\- Achieves high accuracy using CNN layers, dropout, and ReLU activations  
\- Supports prediction on custom uploaded digit images  
\- Preprocessing pipeline included using OpenCV for custom input  
\- Saves the trained model as 'digit_recognition_model.h5'  

## Dataset

\- Source: MNIST (loaded from `tensorflow.keras.datasets.mnist`)  
\- Input shape: 28x28 grayscale images  
\- Classes: 10 (digits 0–9)  

## Model Architecture

\- Input reshaped to 28x28x1  
\- Conv2D: 32 filters, kernel size (3,3), activation='relu'  
\- MaxPooling2D  
\- Flatten  
\- Dense: 128 neurons, activation='relu'  
\- Dropout: 0.2  
\- Dense: 10 neurons, activation='softmax'  

## How to Run

1. Ensure Python, TensorFlow, NumPy, and OpenCV are installed  
