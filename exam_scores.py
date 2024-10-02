import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras import optimizers

data = np.array([[1.4, 20.], [1.8, 30.], [2.1, 37.], [2.2, 45.],
[2.3,26.], [2.9, 86.], [3.5, 67.], [3.6, 100.],
[4.2, 72.], [4.6, 82.], [4.7, 99.]])

[X_train,Y_train] = data.transpose()

model = Sequential()
model.add(Dense(1, activation='linear', input_dim=1))
SGD = optimizers.SGD(lr=0.03) #lr is the learning rate
model.compile(loss='mean_squared_error',optimizer=SGD)

#this command does the minimization
model.fit(X_train, Y_train, epochs=100, verbose=1)

#displays some info, note there are only 2 fitting parameters
model.summary()

# the predict function on the next line assumes an
# array of inputs so we need to put 2.7 inside an array
x = np.array([2.7])

#... and predicted_y is likewise an array of predicted outputs
predicted_y = model.predict(x)

print(f'For x={x:f} the predicted y is: {predicted_y:f}')
