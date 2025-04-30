from tensorflow.keras import layers, models, applications, regularizers, optimizers
import tensorflow as tf

def build_optimized_frr_cnn(input_dimensions=(48, 48, 1), total_classes=7):
    # Load VGG16 base with ImageNet weights
    vgg16_base = applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(48, 48, 3)
    )
    vgg16_base.trainable = True  # Fine-tune all layers

    # Construct the full model
    expression_model = models.Sequential([
        layers.Input(shape=input_dimensions),
        layers.Conv2D(3, (3, 3), padding='same'),  # Convert grayscale to 3 channels

        vgg16_base,
        layers.BatchNormalization(),
        layers.Flatten(),

        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(total_classes, activation='softmax')
    ])

    expression_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    return expression_model
