import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import psutil
import gc


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)



BASE_DIR = '/kaggle/input/pokemonclassification/PokemonData'
IMG_SIZE = (160, 160)
BATCH_SIZE = 16

def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        BASE_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        BASE_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator

# Create the balanced model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        # First Block
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second Block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third Block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fourth Block
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Dense Layers
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, train_generator, validation_generator, epochs=50):

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    best_val_acc = 0
    patience = 0
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Training loop
        train_losses = []
        train_accuracies = []
        train_generator.reset()

        for step in range(len(train_generator)):
            x_batch, y_batch = next(train_generator)

            # Training step
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss_value = loss_fn(y_batch, logits)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Calculate accuracy manually
            pred = tf.argmax(logits, axis=1)
            true = tf.argmax(y_batch, axis=1)
            acc = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

            train_losses.append(float(loss_value))
            train_accuracies.append(float(acc))

            # Clear memory
            del x_batch, y_batch, logits, grads, pred, true
            if step % 20 == 0:
                gc.collect()
                print(f"Step {step}")
        # Validation loop
        val_losses = []
        val_accuracies = []
        validation_generator.reset()

        for step in range(len(validation_generator)):
            x_val, y_val = next(validation_generator)
            val_logits = model(x_val, training=False)
            val_loss = loss_fn(y_val, val_logits)

            # Calculate validation accuracy manually
            val_pred = tf.argmax(val_logits, axis=1)
            val_true = tf.argmax(y_val, axis=1)
            val_acc = tf.reduce_mean(tf.cast(tf.equal(val_pred, val_true), tf.float32))

            val_losses.append(float(val_loss))
            val_accuracies.append(float(val_acc))

            del x_val, y_val, val_logits, val_pred, val_true

        # Calculate epoch metrics
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = np.mean(train_accuracies)
        epoch_val_loss = np.mean(val_losses)
        epoch_val_acc = np.mean(val_accuracies)

        history['accuracy'].append(float(epoch_train_acc))
        history['val_accuracy'].append(float(epoch_val_acc))
        history['loss'].append(float(epoch_train_loss))
        history['val_loss'].append(float(epoch_val_loss))

        print(f"Training acc: {epoch_train_acc:.4f}")
        print(f"Validation acc: {epoch_val_acc:.4f}")

        # Save best model and check early stopping
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            model.save('best_model.keras')
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print("Early stopping triggered")
                break

        # Clear memory
        gc.collect()
        tf.keras.backend.clear_session()

    return history



if __name__ == "__main__":
    try:
        # Create generators
        train_generator, validation_generator = create_generators()
        num_classes = len(train_generator.class_indices)
        print(f"Number of classes: {num_classes}")

        # Create and compile model
        model = create_model((IMG_SIZE[0], IMG_SIZE[1], 3), num_classes)
        model.summary()

        # Train model
        history = train_model_efficiently(model, train_generator, validation_generator)

        # Save final model and results
        model.save('final_model.keras')

        with open('training_history.json', 'w') as f:
            json.dump(history, f)



        print(f"\nTraining completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

    finally:
        # Final cleanup
        gc.collect()
        tf.keras.backend.clear_session()