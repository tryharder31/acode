#git commit -m "a" && git add . && git push


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# It's a good practice to scale the data for neural networks training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a MLPRegressor model
model = MLPRegressor(hidden_layer_sizes=(64,64,64), activation='relu', solver='adam', max_iter=500, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Use the model to make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Training set: MSE = {}, R2 = {}".format(mse_train, r2_train))
print("Test set: MSE = {}, R2 = {}".format(mse_test, r2_test))


quit()


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
