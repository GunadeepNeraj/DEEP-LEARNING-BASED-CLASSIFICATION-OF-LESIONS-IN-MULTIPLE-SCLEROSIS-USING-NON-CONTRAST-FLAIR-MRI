import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, DenseNet121
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, LSTM, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Constants ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

# --- Data Paths (Make sure they exist and are populated) ---
TRAIN_DIR = "/content/drive/MyDrive/mri_data/train"
VAL_DIR = "/content/drive/MyDrive/mri_data/val"
TEST_DIR = "/content/drive/MyDrive/mri_data/test"

# --- Data Generators with Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- Feature Extraction Models ---
def build_feature_extractor(base_model_class, input_shape=(IMG_SIZE, IMG_SIZE, 3), name="feature_model"):
    base_model = base_model_class(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    return Model(inputs, x, name=name)

# InceptionV3
inception_model = build_feature_extractor(InceptionV3, name="InceptionV3_features")

# DenseNet121
densenet_model = build_feature_extractor(DenseNet121, name="DenseNet121_features")

# --- Dual CNN Feature Generator ---
def generate_features(generator, model1, model2):
    features1 = model1.predict(generator, verbose=1)
    features2 = model2.predict(generator, verbose=1)
    return np.concatenate([features1, features2], axis=1), generator.classes

# Extract and combine features
X_train, y_train = generate_features(train_gen, inception_model, densenet_model)
X_val, y_val = generate_features(val_gen, inception_model, densenet_model)
X_test, y_test = generate_features(test_gen, inception_model, densenet_model)

# --- Reshape for LSTM ---
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# --- Build LSTM Classifier ---
input_layer = Input(shape=(1, X_train.shape[2]))
x = LSTM(100, return_sequences=False, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output_layer = Dense(2, activation='softmax')(x)

lstm_model = Model(input_layer, output_layer)
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# --- Train Model ---
history = lstm_model.fit(X_train, y_train,
                         validation_data=(X_val, y_val),
                         epochs=EPOCHS,
                         batch_size=BATCH_SIZE)

# --- Evaluate Model ---
loss, acc = lstm_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# --- Save Model ---
lstm_model.save('h12.h5')
print("Model saved as h12.h5")
