import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# 70,000 greyscale clothing images. 60k training img & 10k test img.
data = tf.keras.datasets.fashion_mnist

# load_data
# img: 28x28 pixel arrays
# lbl: 0-9 aka categories of clothes
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# each pixel is 0-255 grayscale. This will normalize the data. (get values closer)
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    # not a layer. Flattens 2D input image array into 1D line
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # units=neurons. Too many units=overfitting. Too few, may not have enough parameters to learn
    # activation=fx used by neuron. rectified linear unit=only pass to next layer if positive
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    # output layer. 10 neurons, one for each class of clothing. Softmax given us highest value
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

model.compile(
    # adam is improvement over sgd previously used
    optimizer='adam',
    # each image maps to category, so this loss fx is good
    loss='sparse_categorical_crossentropy',
    # report back how well the network is learning
    metrics=['accuracy']
)

# fit img to label
model.fit(training_images, training_labels, epochs=5)

# pass 10k img to trained model to evaluate
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

# values of the 10 output neurons with confidence
# [8.1111517e-07
# 2.8020648e-07
# 1.9261265e-07
# 1.7469000e-08
# 2.9035716e-08
#  3.0743305e-03 => .0003% confident img at idx 0 is label 6
#  8.5821466e-07
#  2.6676910e-02
#  1.0086280e-05
#  9.7023654e-01] => 97% confident img at idx 0 is label 9
print(classifications[0])

# label for clothing at idx 0 is 9
print(test_labels[0])


########################################################################################################################
# Having fun with output
########################################################################################################################

def image_details(classification, idx):
    # possible labels
    labels_array = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # Sort the classifications array in descending order and get the indices of the top two values
    top_indices = np.argsort(classification)[::-1][:2]

    # Get the top two maximum values in the classifications array
    top_values = classification[top_indices]

    # Get the corresponding labels
    predicted_labels = [labels_array[i] for i in top_indices]

    print(f"With {top_values[0] * 100:.2f}% confidence, the image maps to label: {predicted_labels[0]}\n"
          f"My second best guess is with {top_values[1] * 100:.2f}% confidence, "
          f"the image maps to label: {predicted_labels[1]}\n")

    # Display the image at index
    plt.imshow(test_images[idx], cmap='gray')
    plt.axis('off')
    plt.show()


image_details(classifications[0], 0)

# Loop 5 times to make 5 random guesses
for _ in range(5):
    # Get a random index within the range of the array
    random_index = random.randint(0, classifications.shape[0] - 1)

    # Call image_details function to display image details
    image_details(classifications[random_index], random_index)

