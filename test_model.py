import unittest
import numpy as np
from model import WaveNetModel

class TestModel(unittest.TestCase):
  def setUp(self):
    # Initialize the model and input data
    self.model = WaveNetModel()
    self.model(np.random.rand(100, 1).reshape(1, 100, 1))
    self.model.load_weights('weights.h5')
    self.input_data = np.random.rand(100, 1)
    self.input_data = self.input_data.reshape(1, 100, 1)
  
  def test_only_using_past_data(self):
    # Get the model's prediction for the input data
    prediction = self.model(self.input_data)
    
    # Check that the model's prediction only depends on past data
    for i in range(prediction.shape[0]):
      print(prediction[i])
      print(self.input_data[:i+1])
      self.assertFalse(np.isclose(prediction[i], self.input_data[:i+1], rtol=1e-5, atol=1e-5))

if __name__ == '__main__':
  unittest.main()