# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# defining a single layer of type Seq => Dense (fully connected)
model = Sequential([Dense(units=1, input_shape=[1])])
# optimizer = stochastic gradient descent, predicting how accurate guess was
model.compile(optimizer='sgd', loss='mean_squared_error')

# y=2x-1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# learning step
model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
