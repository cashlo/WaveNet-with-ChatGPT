import argparse
import os

import tensorflow as tf
import numpy as np
import librosa

from model import WaveNetModel

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--data_dir', type=str, default='data', help='Directory where the data is stored')
args = parser.parse_args()

# Load and preprocess the data
def load_and_preprocess_data(filename):
  # Load the audio file as a waveform
  waveform, sr = librosa.load(filename)

  # Convert the waveform to a tensor
  waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)

  return waveform_tensor

def create_dataset(data_dir):
  # List all the audio files in the data directory
  filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav') or f.endswith('.mp3')]

  # Create a dataset object that loads and preprocesses the audio files
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset.map(load_and_preprocess_data)
  dataset = dataset.batch(args.batch_size)
  dataset = dataset.repeat(args.epochs)

  # Split the dataset into training and validation sets
  train_size = int(0.8 * len(filenames))
  val_size = len(filenames) - train_size
  train_dataset, val_dataset = dataset.skip(val_size).take(train_size), dataset.take(val_size)

  return train_dataset, val_dataset

# Define the model
model = WaveNetModel()

# Compile the model
loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
model.compile(optimizer=optimizer, loss=loss)

train_dataset, val_dataset = create_dataset(args.data_dir)

# Train the model
model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset)

# Evaluate the model on the validation set
val_loss = model.evaluate(val_dataset)
print(f'Validation loss: {val_loss:.4f}')

# Save the model
model.save('wavenet.h5')