from tensorflow import keras
from tensorflow.keras import metrics, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, LSTM
from tensorflow.keras.layers import Dense, Activation, Dropout

# Compiler and layer parameters
ACTIVATION = 'relu'
LOSS = keras.losses.categorical_crossentropy
OPTIMIZER = keras.optimizers.Adam(learning_rate=0.00001)
METRICS = ['accuracy']

def model(
      activation = ACTIVATION,
      loss = LOSS,
      optimizer = OPTIMIZER,
      metrics = METRICS
      ):
  """
  Recreates the architecture of the final model
  from the project and compiles it using the
  passed arguments.

  NOTE: This function is written in such a way
  that all layers minus the output layer
  use the same activation function, which was
  passed in as an argument.

  Parameters:
  -----------
  activation: layer activation function
  loss: model loss function
  optimizer: model optimizer
  metrics: metrics to optimize

  Returns:
  --------
  Compiled model object that can be fitted to
  data and used for making predictions
  """
  model = Sequential(name = 'CNN_LSTM_Hybrid')

  # 1st Convolutional Layer
  model.add(Sequential([
    Conv1D(128,10,
          activation=activation,
          kernel_regularizer=regularizers.l2(0.01),
          input_shape=(640, 7)),
    BatchNormalization(),
    MaxPooling1D(pool_size=4),
    Dropout(0.1)],
          name = 'Conv1'))

  # 2nd Convolutional Layer
  model.add(Sequential([
    Conv1D(256,5,
          activation=activation,
          kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    MaxPooling1D(pool_size=4),
    Dropout(0.1)],
          name = 'Conv2'))

  # 3rd Convolutional Layer
  model.add(Sequential([
    Conv1D(512,5,
          activation=activation,
          kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    MaxPooling1D(pool_size=4),
    Dropout(0.1)],
          name = 'Conv3'))

  # LSTM Layer
  model.add(Sequential([
    LSTM(350),
    BatchNormalization(),
    Dropout(0.1)],
          name = 'LSTM'))

  # Dense Layer
  model.add(Sequential([
    Dense(64, activation=activation, bias_regularizer=regularizers.l2(1e-4)),
    Dropout(0.2)],
          name = 'Dense1'))

  # Output Layer
  model.add(Sequential([
    Dense(8, activation='softmax')],
          name = 'Final'))

  # Compiler
  model.compile(loss =loss,
    optimizer=optimizer,
    metrics = metrics)

  return model

if __name__ == "__main__":
      pass