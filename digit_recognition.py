


import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np

# Define the preprocess_custom_image function
def preprocess_custom_image(image_path):
    custom_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    custom_image = cv2.resize(custom_image, (28, 28))
    custom_image = custom_image / 255.0
    custom_image = custom_image.reshape(1, 28, 28, 1)  # Add an extra dimension for grayscale
    return custom_image

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),  # Reshape for convolutional layers
    layers.Conv2D(32, (3,3), activation='relu'),  # Convolutional layer with ReLU activation
    layers.MaxPooling2D((2,2)),  # Max pooling layer
    layers.Flatten(),  
    layers.Dense(128, activation='relu'),  
    layers.Dropout(0.2),  
    layers.Dense(10, activation='softmax')  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Define the path to your custom image
custom_image_path = r'C:\Users\Dell 7410\Desktop\digit recognition\WhatsApp Image 2023-09-08 at 8.28.39 PM (1).jpeg'

# Preprocess the custom image
processed_image = preprocess_custom_image(custom_image_path)

# Use the model to make predictions on the custom image
predictions = model.predict(processed_image)

# Get the predicted digit
predicted_digit = np.argmax(predictions)

print(f'The predicted digit is: {predicted_digit}')

# Save the model
model.save('digit_recognition_model.h5')
