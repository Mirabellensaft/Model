import mlflow
import mlflow.tensorflow
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 80
img_width = 60
batch_size = 32
epochs = 10

# Set MLflow tracking URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")

# Start MLflow run
with mlflow.start_run():

    # Define and compile your TensorFlow model
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(108, activation='softmax')
])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Load and preprocess image data
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        'train/',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        'val/',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    
    # Extract class names
    class_names = {v: k for k, v in train_generator.class_indices.items()}
    with open('class_names.json', 'w') as json_file:
        json.dump(class_names, json_file)

    # Train the model
    model.fit(train_generator, validation_data=val_generator, epochs=epochs)

    # Evaluate the model
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'test/',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    loss, accuracy = model.evaluate(test_generator)

    # Log parameters, metrics, and model
    mlflow.log_params({
        "epochs": epochs,
        "optimizer": "adam"
    })
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("loss", loss)
    mlflow.tensorflow.log_model(model, "model")


    model.save("model/model.h5")