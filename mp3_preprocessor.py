import glob
import librosa
import numpy as np

def load_and_convert_mp3(filename):
  # Load the audio file as a waveform
  waveform, sr = librosa.load(filename)

  # Convert the waveform to a tensor
  tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)

  return tensor

# Find all MP3 files in the data directory
filenames = glob.glob('data/*.mp3')

# Load and convert the MP3 files
waveforms = [load_and_convert_mp3(filename) for filename in filenames]

# Stack the waveform tensors into a single tensor
waveform_tensor = tf.stack(waveforms)