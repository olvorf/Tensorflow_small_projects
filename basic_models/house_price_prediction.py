import tensorflow as tf
import numpy as np
from tensorflow import keras

def house_model(y_new):
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)
    ys = np.array([100,150,200,250,300,350,450,500,550,600,650,700,750], dtype= float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) 
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs,ys, epochs=4000)
    return (model.predict(y_new)[0]+1)//100

if __name__ == '__main__':
    prediction = house_model([7.0])
    print(prediction)
