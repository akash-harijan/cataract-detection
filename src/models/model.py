from tensorflow import keras
import tensorflow as tf

def create_model(img_size=(160, 160, 3)):
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        input_shape=img_size,
        include_top=False,
    )

    base_model.trainable = False

    inputs = keras.Input(shape=img_size)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.summary()

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model
