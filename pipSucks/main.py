import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# layer to hold what is learned
learned = Dense(units=1, input_shape=[1])
# defining a single layer of type Seq => type Dense (fully connected)
model = Sequential(learned)
# optimizer = stochastic gradient descent, predicting how accurate guess was
model.compile(optimizer='sgd', loss='mean_squared_error')

# y=2x-1 where weight is 2 and bias is -1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# learning x500 times
model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
# relationships between X & Y
print("Here is what I learned: {}".format(learned.get_weights()))
