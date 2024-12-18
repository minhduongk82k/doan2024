import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

def create_model(input_shape, num_classes):
    """
    Create a CNN model.
    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        num_classes (int): Number of output classes.
    Returns:
        Sequential: Compiled CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir, model_save_path, input_size=(48, 48), batch_size=32, epochs=10):

    train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2) 

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    print("Class indices:", train_generator.class_indices)

    model = create_model(input_shape=(input_size[0], input_size[1], 3), num_classes=2)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        verbose=1
    )

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    data_dir = "../data/data_train" 
    model_save_path = "../saved_model/emotion_recognition_model.h5"
    train_model(data_dir, model_save_path)
