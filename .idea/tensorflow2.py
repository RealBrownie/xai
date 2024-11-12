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

# Print GPU info and start time
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def print_memory_usage():
    print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

# Set paths and parameters
BASE_DIR = '/kaggle/input/pokemonclassification/PokemonData'
IMG_SIZE = (160, 160)  # Balanced size
BATCH_SIZE = 32

print_memory_usage()

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    validation_split=0.2
)

# Set up generators
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

num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

print(f"\nDataset summary:")
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print_memory_usage()

# Create the balanced model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        # First Block
        layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Second Block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Third Block
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Fourth Block
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Dense Layers
        layers.Flatten(),
        layers.Dense(1024),
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

# Create and compile model
print("\nCreating model...")
model = create_model((IMG_SIZE[0], IMG_SIZE[1], 3), num_classes)
model.summary()
print_memory_usage()

# Compile with learning rate schedule
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Set up TensorBoard
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# Train the model
print(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            'pokemon_model_{epoch:02d}_{val_accuracy:.3f}.keras',
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        tensorboard_callback
    ]
)

# Save full model and weights
print("\nSaving model and weights...")
model.save('pokemon_classifier_final.keras')
model.save_weights('pokemon_weights_final.keras')

# Save training history and parameters
params = {
    'class_names': class_names,
    'img_size': IMG_SIZE,
    'num_classes': num_classes,
    'training_history': {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
}

with open('model_params.json', 'w') as f:
    json.dump(params, f)

# Plot results
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_plots.png')
plt.show()

# Function to predict and visualize results
def show_predictions(generator, num_images=5):
    # Get a batch of images
    images, labels = next(generator)
    predictions = model.predict(images)
    plt.figure(figsize=(15, 3))
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        plt.axis('off')
        true_class = class_names[np.argmax(labels[i])]
        pred_class = class_names[np.argmax(predictions[i])]
        color = 'green' if true_class == pred_class else 'red'
        plt.title(f'True: {true_class}\nPred: {pred_class}', color=color)
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.show()

# Show some predictions
print("\nGenerating prediction examples...")
validation_generator.reset()
show_predictions(validation_generator)

print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_memory_usage()

# Final saved files summary
print("\nSaved files:")
print("1. pokemon_classifier_final.keras - Complete model")
print("2. pokemon_weights_final.keras - Model weights")
print("3. model_params.json - Class names and training history")
print("4. training_plots.png - Accuracy and loss plots")
print("5. prediction_examples.png - Example predictions")
print("\nYou can find these files in the Output section of your Kaggle notebook.")

# Calculate and print total training time
start_time = params.get('training_start_time', None)
if start_time:
    end_time = datetime.now()
    training_duration = end_time - datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    print(f"\nTotal training time: {training_duration}")