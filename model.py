import tensorflow as tf

class WaveNetModel(tf.keras.Model):
  def __init__(self, num_filters=64, kernel_size=3):
    super(WaveNetModel, self).__init__()
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    
    self.conv_layers = []
    self.dense_layers = []
    for i in range(num_filters):
      conv_layer = tf.keras.layers.Conv1D(
          filters=num_filters, kernel_size=kernel_size,
          padding="same", activation="relu")
      self.conv_layers.append(conv_layer)
      dense_layer = tf.keras.layers.Dense(1)
      self.dense_layers.append(dense_layer)
      
  def call(self, inputs):
    x = inputs
    for i in range(self.num_filters):
      x = self.conv_layers[i](x)
      x = self.dense_layers[i](x)
    return x