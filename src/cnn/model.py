from keras.models import Model
from keras.layers import Input, Reshape, Dense, Conv1D, Flatten, LeakyReLU, MaxPool1D, Dropout, BatchNormalization, Softmax, Concatenate
from keras.initializers import RandomNormal
import numpy as np
import tensorflow as tf
from keras import optimizers

def CNN_Model(num_classes, input_size):
    inputs = Input(shape=(input_size,1))
    initializer = RandomNormal(mean=0.0, stddev=np.sqrt(0.05))

    opt = optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08
    )
    
    x = Conv1D(filters=16, kernel_size=21, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(2, 2)(x)
    
    x = Conv1D(filters=32, kernel_size=11, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(2, 2)(x)
    
    x = Conv1D(filters=64, kernel_size=5, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(2, 2)(x)
    
    x = Flatten()(x)
    
    x = Dense(units=2048, activation='tanh', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(rate = 0.5)(x)
    
    x = Dense(units=num_classes, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    outputs = Softmax(dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Raman_CNN")
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# other attempt
def CNN_Model1(num_classes, input_size):
    inputs = Input(shape=(input_size,1))
    initializer = RandomNormal(mean=0.0, stddev=np.sqrt(0.05))

    opt = optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08
    )
    
    x = Conv1D(filters=16, kernel_size=21, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(2, 2)(x)
    
    x = Conv1D(filters=32, kernel_size=11, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(2, 2)(x)
    
    x = Conv1D(filters=64, kernel_size=5, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(2, 2)(x)
    
    x = Flatten()(x)

    x = Dense(units=2048, activation='tanh', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Reshape((2048, 1))(x)
    x = MaxPool1D(2, 2)(x)
    x = Flatten()(x)
    x = Dropout(rate=0.5)(x)
    
    x = Dense(units=num_classes, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    outputs = Softmax(dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Raman_CNN")
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# other other attempt
def CNN_Model2(num_classes, input_size):
    inputs = Input(shape=(input_size, 1))
    initializer = RandomNormal(mean=0.0, stddev=np.sqrt(0.05))
    opt = optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08
    )

    # conv block 1
    x1 = Conv1D(filters=16, kernel_size=21, kernel_initializer=initializer)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)
    x1 = MaxPool1D(2, 2)(x1)

    # conv block 2
    x2 = Conv1D(filters=32, kernel_size=11, kernel_initializer=initializer)(x1)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)
    x2 = MaxPool1D(2, 2)(x2)

    # conv block 3
    x3 = Conv1D(filters=64, kernel_size=5, kernel_initializer=initializer)(x2)
    x3 = BatchNormalization()(x3)
    x3 = LeakyReLU()(x3)
    x3 = MaxPool1D(2, 2)(x3)

    # flatten each feature map and concatenate
    f1 = Flatten()(x1)
    f2 = Flatten()(x2)
    f3 = Flatten()(x3)
    x = Concatenate()([f1, f2, f3])

    # dense block 1
    x = Dense(units=2048, activation='tanh', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Reshape((2048, 1))(x)
    x = MaxPool1D(2, 2)(x)
    x = Flatten()(x)
    x = Dropout(rate=0.5)(x)

    # dense block 2
    x = Dense(units=num_classes, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    outputs = Softmax(dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs, name="Raman_CNN")
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model