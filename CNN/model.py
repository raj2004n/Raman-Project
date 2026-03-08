from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, LeakyReLU, MaxPool1D, Concatenate, Dropout, BatchNormalization, Softmax, InputLayer
from tensorflow.keras.initializers import RandomNormal

import tensorflow as tf
from tensorflow.keras import layers, models, Input

initializer = RandomNormal(mean=0.0, stddev=0.05)

def CNN_model1(num_classes, leaky_alpha=0.1, dropout_rate=0.5):
    inputs = Input(shape=(913,1))

    # ── Branch 1: 16 filters, large kernel (21) ──────────────────────────────
    b1 = layers.Conv1D(16, kernel_size=21, padding="same", use_bias=False, name="conv1_16")(inputs)
    b1 = layers.BatchNormalization(name="bn1")(b1)
    b1 = layers.LeakyReLU(negative_slope=leaky_alpha, name="lrelu1")(b1)
    b1 = layers.MaxPooling1D(pool_size=2, strides=2, name="pool1")(b1)

    # ── Branch 2: 32 filters, medium kernel (11) ─────────────────────────────
    b2 = layers.Conv1D(32, kernel_size=11, padding="same", use_bias=False, name="conv2_32")(inputs)
    b2 = layers.BatchNormalization(name="bn2")(b2)
    b2 = layers.LeakyReLU(negative_slope=leaky_alpha, name="lrelu2")(b2)
    b2 = layers.MaxPooling1D(pool_size=2, strides=2, name="pool2")(b2)

    # ── Branch 3: 64 filters, small kernel (5) ───────────────────────────────
    b3 = layers.Conv1D(64, kernel_size=5, padding="same", use_bias=False, name="conv3_64")(inputs)
    b3 = layers.BatchNormalization(name="bn3")(b3)
    b3 = layers.LeakyReLU(negative_slope=leaky_alpha, name="lrelu3")(b3)
    b3 = layers.MaxPooling1D(pool_size=2, strides=2, name="pool3")(b3)

    # ── Flatten each branch then concatenate ─────────────────────────────────
    f1 = layers.Flatten(name="flat1")(b1)
    f2 = layers.Flatten(name="flat2")(b2)
    f3 = layers.Flatten(name="flat3")(b3)

    concat = layers.Concatenate(name="concat")([f1, f2, f3])

    # ── Dense block: 2048 → BN → Tanh → Pooling (Dense reduction) → Dropout ──
    # The diagram shows Pooling(2,2) after Tanh inside the dense block;
    # this is interpreted as halving the feature dimension via a Dense layer.
    x = layers.Dense(2048, use_bias=False, name="dense_2048")(concat)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("tanh", name="tanh")(x)
    # "Pooling(2,2)" on a 1-D dense vector → reduce to 1024 units
    x = layers.Dense(1024, use_bias=False, name="dense_pool")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # ── Output head: Dense(#classes) → BN → Softmax ──────────────────────────
    x = layers.Dense(num_classes, use_bias=False, name="dense_classes")(x)
    x = layers.BatchNormalization(name="bn_out")(x)
    outputs = layers.Activation("softmax", name="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Raman_CNN")
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def CNN_model(num_classes):
    inputs = Input(shape=(913,1))
    
    x = Conv1D(filters=16, kernel_size=21, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(pool_size=2)(x)
    
    x = Conv1D(filters=32, kernel_size=11, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(pool_size=2)(x)
    
    x = Conv1D(filters=64, kernel_size=5, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(pool_size=2)(x)
    
    x = Flatten()(x)
    
    x = Dense(units=2048, activation='tanh', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(rate = 0.5)(x)
    
    x = Dense(units=num_classes, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    outputs = Softmax(dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Raman_CNN")
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model