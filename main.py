import numpy as np
import tensorflow as tf
#git commit -m "a" && git add . && git push
import tensorflow as tf

# Check if GPU is available
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# Create two matrices
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

# Perform matrix multiplication
product = tf.matmul(matrix1, matrix2)

print(product)
