#git commit -m "a" && git add . && git push
print('hi')
output_file = "output.txt"
f = open(output_file, "w")
print('hi',file=f)
import numpy as np
import tensorflow as tf

# Check if GPU is available
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()),file=f)
else:
    print("Please install GPU version of TF",file=f)

# Create two matrices
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

# Perform matrix multiplication
product = tf.matmul(matrix1, matrix2)

print(product,file=f)
