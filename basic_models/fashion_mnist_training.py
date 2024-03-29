import tensorflow as tf
from os import path, getcwd, chdir


def train_mnist():

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') is not None and (logs.get('acc') > 0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0

    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation= tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    
    # model fitting
    history = model.fit(
                        x_train, y_train, epochs=5, callbacks=[callbacks])
    # model fitting
    return history.epoch, history.history['acc'][-1]


# train with a single convolutional layer

def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.


    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') is not None and (logs.get('acc') > 0.998):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    training_images=training_images.reshape(60000, 28, 28, 1)
    training_images=training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images=test_images/255.0


    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    # model fitting
    history = model.fit(
                        training_images, training_labels, epochs=10, callbacks=[callbacks]
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]


if __name__ == '__main__':
    train_mnist()
    #train_mnist_conv()