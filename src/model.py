from tensorflow import keras
from tensorflow.keras import metrics, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, LSTM
from tensorflow.keras.layers import Dense, Activation, Dropout

# Compiler parameters
loss = keras.losses.categorical_crossentropy
optimizer = keras.optimizers.Adam(learning_rate=0.00001)
metrics = ['accuracy']

def model():
  model = Sequential(name = 'CNN_LSTM_Hybrid')

  # 1st Convolutional Layer
  model.add(Sequential([
    Conv1D(128,10,
          activation='relu',
          kernel_regularizer=regularizers.l2(0.01),
          input_shape=(640, 7)),
    BatchNormalization(),
    MaxPooling1D(pool_size=4),
    Dropout(0.1)],
          name = 'Conv1'))

  # 2nd Convolutional Layer
  model.add(Sequential([
    Conv1D(256,5,
          activation='relu',
          kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    MaxPooling1D(pool_size=4),
    Dropout(0.1)],
          name = 'Conv2'))

  # 3rd Convolutional Layer
  model.add(Sequential([
    Conv1D(512,5,
          activation='relu',
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
    Dense(64, activation='relu', bias_regularizer=regularizers.l2(1e-4)),
    Dropout(0.2)],
          name = 'Dense1'))

  # Output Layer
  model.add(Sequential([
    Dense(8, activation='softmax')],
          name = 'Final'))

  # Compiler
  model2.compile(loss =loss,
    optimizer=optimizer,
    metrics = metrics)

  return model