import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np

# Load the grayscale image
img_gray = load_img('bw_image.jpg', color_mode='grayscale')
img_gray = img_gray.resize((256,256))  # Resize the image to the desired size
img_gray = img_to_array(img_gray) / 255.0  # Convert the image to a numpy array and normalize

# Add a new dimension to the array to make it compatible with the input shape of the model
img_gray = np.expand_dims(img_gray, axis=0)

# Load the pre-trained model
model = Sequential([
    InputLayer(input_shape=(None, None, 1)),
    Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    Conv2D(128, (3,3), activation='relu', padding='same', strides=2),
    Conv2D(256, (3,3), activation='relu', padding='same'),
    Conv2D(256, (3,3), activation='relu', padding='same', strides=2),
    Conv2D(512, (3,3), activation='relu', padding='same'),
    Conv2D(512, (3,3), activation='relu', padding='same'),
    Conv2D(256, (3,3), activation='relu', padding='same'),
    UpSampling2D((2,2)),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    UpSampling2D((2,2)),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    Conv2D(2, (3, 3), activation='tanh', padding='same'),
    UpSampling2D((2, 2))
])

model.compile(optimizer='adam', loss='mse')

# Load the pre-trained weights
model.load_weights('colorization_weights.h5')

# Colorize the grayscale image
img_colorized = model.predict(img_gray)
img_colorized = (img_colorized * 128 + 128).clip(0, 255).astype('uint8')
img_colorized = array_to_img(img_colorized[0])

# Save the colorized image
img_colorized.save('colorized_image.jpg')
