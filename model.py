import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Reshape, LSTM, LeakyReLU

# Need to declare it every time, otherwise it gets erased from memory
def cldnn():
    #FILL THIS IN WITH MODEL ARCHITECTURE
    model = Sequential()
    model.add(Conv2D(256,(1,3),activation="relu",input_shape =(2,39,1)))
    model.add(Conv2D(256,(2,3),activation=None))
    model.add(LeakyReLU(0.3))
    model.add(Conv2D(256,(1,3),activation=None))
    model.add(LeakyReLU(0.3))
    model.add(Dropout(0.20))
    model.add(Conv2D(80,(1,3),activation="relu"))
    model.add(Reshape((31,80)))
    model.add(Flatten())
    model.add(Dense(128,activation="relu",kernel_initializer="normal"))
    model.add(Dense(2,activation="sigmoid"))
    model.summary()
    
    return model