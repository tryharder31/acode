from PIL import Image
import numpy as np

# Load the image
image = Image.open('path_to_your_image.jpg')

# Convert the image to a NumPy array
image_array = np.array(image)
from keras.preprocessing.image import img_to_array, load_img

# Resize the image to the size your CNN expects (e.g., 32x32)
image = load_img('path_to_your_image.jpg', target_size=(32, 32))

# Convert the image to a NumPy array
image_array = img_to_array(image)

# Scale the image pixels to be between 0 and 1
image_array /= 255.0
images_list = [image_array1, image_array2, image_array3, ...]  # List of your image arrays
images_batch = np.array(images_list)  # Convert list of arrays to a 4D batch array
predictions = cnn_model.predict(images_batch)
