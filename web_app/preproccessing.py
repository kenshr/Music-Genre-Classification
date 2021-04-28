import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from pydub import AudioSegment
import librosa
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

def audio_split_30(filepath):
  """
  Takes an mp3 file and returns a new
  mp3 file containing the first 30
  seconds of the original audio.

  Parameters:
  -----------
  filepath (str): string giving absolute
  or relative filepath to audio file

  Returns:
  --------
  Returns a 30 second mp3 taken from
  the original audio file
  """

  try:
    song = AudioSegment.from_mp3(filepath)
    return song[:30000]
  except:
     return 'Invalid file. Needs to be an mp3 file of 30 seconds or longer.'


def preprocess(filepath):
  # Create spectrogram
  y, sr = librosa.load(path)
  spect = librosa.feature.melspectrogram(y=y, sr=sr)
  spect = librosa.power_to_db(spect, ref=np.max)
  spect = spect.T
  spect = spect[:640,:]
  spect = librosa.core.db_to_power(spect)
  X = np.log(spect)

  # Flatten spectrogram into channels
  sub_bass = X[:, :, :2].mean(axis=2)
  bass = X[:, :, 2:6].mean(axis=2)
  lower_midrange = X[:, :, 6:12].mean(axis=2)
  midrange = X[:, :, 12:48].mean(axis=2)
  higher_midrange = X[:, :, 48:96].mean(axis=2)
  presence = X[:, :, 96:112].mean(axis=2)
  harmonics = X[:, :, 112:128].mean(axis=2)

  X_flat = np.dstack((sub_bass, bass, lower_midrange, midrange,
                      higher_midrange, presence, harmonics))

  return X_flat

def create_radar_chart(probas):
  fig = go.Figure(data=go.Scatterpolar(
    r=probas,
    theta=['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
          'Instrumental', 'International', 'Pop', 'Rock'],
    fill='toself'))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True
      ),
    ),
    showlegend=False
  )

  fig.show()