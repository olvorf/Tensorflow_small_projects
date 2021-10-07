import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

#path = f"{getcwd()}/../tmp2/happy-or-sad.zip"

#path = "C:\Users\olsiv\Desktop\Coursera\Tensorflow_developer\happy-or-sad.zip"
#path = f"{getcwd()}/happy-or-sad.zip"

#zip_ref = zipfile.ZipFile(path, 'r')
#zip_ref.extractall("/tmp/h-or-s")
#zip_ref.close()


local_zip = './C1/happy-or-sad.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./C1/happy-or-sad')
zip_ref.close()

def train_happy_sad_model():

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') is not None and (logs.get('acc') > DESIRED_ACCURACY):
                print("\n Reached 99.9% accuracy so cancelling the training")
                self.model.stop_training = True
 

    callbacks = myCallback()

    	# This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        # Your Code Here
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',
                    optimizer=RMSprop(learning_rate=0.001),
                    metrics=['acc'])

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
                        './C1/happy-or-sad',
                        target_size = (150,150),
                        batch_size = 10,
                        class_mode = 'binary')
    
    history = model.fit(
                            train_generator,
                            steps_per_epoch = 8,
                            epochs = 15,
                            verbose = 1,
                            callbacks = [callbacks])
    
    # model fitting
    return history.history['acc'][-1]

if __name__ == '__main__':
    train_happy_sad_model()