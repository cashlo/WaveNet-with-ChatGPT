import argparse
import os

import tensorflow as tf
import numpy as np
import librosa

from model import WaveNetModel
from collections import Counter

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
  print(f"loading {filename}...")
  waveform, sr = librosa.load(filename)

  waveform = waveform[2000:2010]

  # Convert the waveform to a mu-law encoding
  waveform_mu_law = librosa.mu_compress(waveform)

  print(Counter(waveform_mu_law))


  # Convert the mu-law encoding to a tensor
  waveform_mu_law_tensor = tf.convert_to_tensor(waveform_mu_law, dtype=tf.uint8)

  # Reshape the tensor to fit the input of the Conv1D layer
  waveform_mu_law_tensor = tf.reshape(waveform_mu_law_tensor, (-1, 1))

  return waveform_mu_law_tensor

def create_dataset(data_dir):
  # List all the audio files in the data directory
  filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav') or f.endswith('.mp3')]
  audio_tensors = [load_and_preprocess_data(f) for f in filenames]

  # Find the maximum length of the audio tensors
  max_length = max([t.shape[0] for t in audio_tensors])
  
  # Pad the shorter tensors with zeros
  print(audio_tensors)
  audio_tensors = [tf.pad(t, [[0,max_length - t.shape[0]], [0, 0]]) for t in audio_tensors]

  # Create the output tensors
  output_tensors = [tf.slice(t, [1, 0], [t.shape[0]-1, t.shape[1]]) for t in audio_tensors]
  output_tensors = [tf.pad(t, [[0,1], [0, 0]]) for t in output_tensors]


  # Create a dataset object that loads and preprocesses the audio files
  input_dataset = tf.data.Dataset.from_tensor_slices(audio_tensors)
  output_dataset = tf.data.Dataset.from_tensor_slices(output_tensors)
  dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
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

print("Training set:")
for element in train_dataset: print(element)
print("Validation set:")
for element in val_dataset: print(element)

# Train the model
model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset)

# Evaluate the model on the validation set
val_loss = model.evaluate(val_dataset)
print(f'Validation loss: {val_loss:.4f}')

# Save the model
model.save_weights('weights.h5')