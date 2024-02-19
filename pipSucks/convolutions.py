import os
import tensorflow as tf
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This hides info and warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


data = tf.keras.datasets.fashion_mnist  # so much clothes!

((training_images, training_labels),
 (test_images, test_labels)) = data.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    # 64 randomly initialized convolutions.
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Custom callback for timing
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.total_time = time.time()

    def on_train_end(self, logs={}):
        self.total_time = time.time() - self.total_time
        print(f"Total training time: {self.total_time:.2f} seconds")

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time.time() - self.epoch_time_start
        print(f"Epoch {epoch} time: {epoch_time:.2f} seconds")



# Create an instance of the custom callback
time_callback = TimeHistory()


model.fit(training_images, training_labels, epochs=50, callbacks=[time_callback])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

# num_randomly_initialized_convolutions = 64
