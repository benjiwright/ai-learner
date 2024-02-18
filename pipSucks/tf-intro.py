import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def lesson():
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

    print_predict(model, predict=10)

    # relationships between X & Y
    (x_array, y_array) = learned.get_weights()

    print(f"Here is what I learned:\n"
          f"x_val = {x_array[0]}, type = {type(x_array[0])} \n "
          f"y_val = {y_array[0]}, type = {type(y_array[0])}"
          )


def print_predict(model, predict):
    print(f"When X is {predict}, the model predicts the value to be {model.predict([predict])}")
    print_actual(predict)


def print_actual(x):
    y = 2 * x - 1
    print(f"The actual value for X at {x} is {y}")


lesson()
