# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from utils.helpers import change_tensor_to_expected_shape
import numpy as np

def build_cnn(input_shape=(33,33,4)):
  """
    input_shape : (Height, WIdth, Channels), Keras expects the channels to be at the end
  """
  model = Sequential([
    Conv2D(64, (3,3), activation ='relu', input_shape=input_shape, padding='same'),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((3,3), strides=(2,2)),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D((3,3), strides=(2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(5, activation='softmax')
  ])

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model

def train_model(patches_input_path, labels_input_path):
  patches = np.load(patches_input_path, allow_pickle=True)
  labels = np.load(labels_input_path, allow_pickle=True)
  patches = change_tensor_to_expected_shape(patches)
  print("patches size ====== ", patches.shape)

  model = build_cnn()
  model.summary()

  model.fit(patches, labels, epochs=10, batch_size=128, validation_split=0.1, shuffle=True, verbose=True)
  return model 

def save_model(model, output_path):
  model.save(output_path)

def save_model_weights_only(model, output_path):
  model.save_weights(output_path)
  # output_path = output_path + ".h5"
  # model.save_weights(output_path)