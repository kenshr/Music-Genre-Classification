import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from pydub import AudioSegment
import librosa
import warnings
warnings.filterwarnings('ignore')

# This class is currently built to primarily support working with the fma_small dataset but
# has the tools necessary to be refactored to handle all datasets. Will try to revisit this
# when I have more time.

class AudioManagement:
  def __init__(self, audio_dir = '../data/fma_small/', metadata_path = '../data/fma_metadata/tracks.csv', split_dir_name = 'split'):
    self.audio_dir = audio_dir
    self.metadata_path = metadata_path
    self.split_dir_name = split_dir_name
    self.split = False


  def make_spectrogram(self, track_id):
    """
    Takes a track_id, identifies the corresponding mp3's
    file path, and returns a spectrogram of the mp3.

    Parameters:
    -----------
    track_id (int): number representing a song's track id

    Returns:
    --------
    Returns a matrix of values which represent the
    spectrogram of the audio file that was passed in
    """
    # Format track id to match filename
    tid = '{:06d}'.format(track_id)
    # Create file path to mp3
    path = os.path.join(self.audio_dir, tid[:3], tid + '.mp3')
    # Generate spectrogram from mp3
    y, sr = librosa.load(path)
    spect = librosa.feature.melspectrogram(y=y, sr=sr)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T


  def create_feature_label_data(self, split):
    """
    Loads in metadata from the path specified
    when the class instance was created, iterates
    through all tracks for the specified split,
    and returns spectrograms and their corresponding
    genres as two separate arrays (feature, label).

    NOTE: The function naturally splits the data
    because the creator of the dataset pre-split
    the data, meaning that there's a column denoting
    which split each track belongs to.

    Parameters:
    -----------
    split (str): A string indicating which split
    (training, validation, or test) to generate
    feature and label data for

    Returns:
    --------
    Returns a feature array (X) and a label array (y)
    """
    genres = []
    X = np.empty((0, 640, 128))
    ct = 0

    # Read in metadata
    tracks = pd.read_csv(self.metadata_path, index_col=0, header=[0,1])
    # Check for incorrect splits
    if len(tracks[(tracks[('set','subset')] == 'small') & (tracks[('set','split')] == split)]) not in [800, 6400]:
      return 'Please provide a valid split: training, validation, or test'
    # Sort for small dataset and split specified in argument
    df = tracks[(tracks[('set', 'subset')] == 'small') & (tracks[('set', 'split')] == split)]

    genre_encoding = {'Electronic':0, 'Experimental':1, 'Folk':2, 'Hip-Hop':3,
                      'Instrumental':4, 'International':5, 'Pop':6, 'Rock':7}

    for index, row in df.iterrows():
      ct += 1
      try:
        track_id = int(index)
        genre = str(row[('track', 'genre_top')])
        spect = self.make_spectrogram(track_id)
        # Normalize for small shape differences from time discrepancies
        spect = spect[:640, :]
        X = np.append(X, [spect], axis=0)
        genres.append(genre_encoding[genre])
      except:
        print("Couldn't process: ", ct)
        continue
    y = np.array(genres)

    X = librosa.core.db_to_power(X)
    X = np.log(X)

    # shuffles data for model learning performance
    shuffle = np.random.permutation(len(X))
    X = X[shuffle]
    y = y[shuffle]

    return X, y

  @staticmethod
  def convert_to_1D(filepath):
    """
    Takes a specified filepath, loads in the data,
    splits the spectrograms into logical frequency
    bands, and flattens each band along its
    frequency (y-axis) via averaging. This converts
    a spectrogram image representing audio into 7
    1-dimensional channels which represent the same
    audio. The channels are then stacked into a matrix
    and outputted as a feature array along with their
    corresponding label array.

    Parameters:
    -----------
    filepath (str): the absolute filepath or relative
    filepath from this file to the desired data file

    Returns:
    --------
    Returns a flattened feature array (X) and a
    label array (y)
    """
    # Load in spectrograms and labels
    try:
      npzfile = np.load(filepath)
    except:
      return 'invalid file'
    X = npzfile['arr_0']
    y = npzfile['arr_1']
    # Split spectrogram into frequency bands and flatten them
    sub_bass = X[:, :, :2].mean(axis=2)
    bass = X[:, :, 2:6].mean(axis=2)
    lower_midrange = X[:, :, 6:12].mean(axis=2)
    midrange = X[:, :, 12:48].mean(axis=2)
    higher_midrange = X[:, :, 48:96].mean(axis=2)
    presence = X[:, :, 96:112].mean(axis=2)
    harmonics = X[:, :, 112:128].mean(axis=2)

    X_flat = np.dstack((sub_bass, bass, lower_midrange, midrange,
                        higher_midrange, presence, harmonics))

    return X_flat, y


  def duplicate_filestructure(self):
    """
    Recreates a given directory's entire file structure in
    a copy directory without files, which lives at the same
    level as the original directory. Also updates instance's
    audio directory to point at new directory.

    Parameters:
    -----------
    dir_path (str): filepath to directory that will be duplicated
    output_dir_name (str): name of new copy directory

    Returns:
    --------
    None
    """
    # Create new output path on same level as duplicated directory
    output_path = os.path.join(os.path.dirname(os.path.dirname(self.audio_dir)), self.split_dir_name)

    # Checks that output directory doesn't already exist
    if not os.path.isdir(output_path):
      # Filestructure recreation
      for root, dirs, files in os.walk(self.audio_dir):
        os.mkdir(os.path.join(output_path, root[len(self.audio_dir):]))

    # Recursive solution in case output file already exists
    else:
      self.split_dir_name = self.split_dir_name + '1'
      duplicate_filestructure(input_path, self.split_dir_name)


  def split_audio(self, segment_length=5):
    """
    Iterates through all mp3 files in the object's audio
    directory, splits the files into chunks of the
    specified length, stores the split files in a duplicated
    directory with the same positions as their parent mp3's.

    Parameters:
    -----------
    segment_length (int): length (seconds) to split mp3's into

    Returns:
    --------
    None
    """

    # Create empty copy directory
    self.duplicate_filestructure()

    # Iterate through audio directory and split mp3 files
    for root, dirs, files in os.walk(self.audio_dir):
      for file in files:
        if file.endswith('.mp3'):
          song = AudioSegment.from_mp3(os.path.join(root, file))
          save_path = root.split('test')[0] + self.split_dir_name + root.split('test')[1]
          # Save chunks into copy directory
          for i, chunk in enumerate(song[::(segment_length*1000)]):
            with open(f'{save_path}{i}-{file}', "wb") as f:
              chunk.export(f, format="mp3")

    self.split = True



if __name__=="__main__":
  # Create instance of class
  am = AudioManagement()

  # Initialize train, validation, test data
  X_train, y_train = am.create_feature_label_data(split='training')
  X_val, y_val = am.create_feature_label_data(split='validation')
  X_test, y_test = am.create_feature_label_data(split='test')
  # Save data
  np.savez('../data/training_data', X_train, y_train)
  np.savez('../data/validation_data', X_val, y_val)
  np.savez('../data/test_data', X_test, y_test)
  # Create 1D data
  X_train_1D, y_train_1D = am.convert_to_1D('../data/training_data.npz')
  X_val_1D, y_val_1D = am.convert_to_1D('../data/validation_data.npz')
  X_test_1D, y_test_1D = am.convert_to_1D('../data/test_data.npz')
  # Save 1D data
  np.savez('../data/1D_training_data', X_train, y_train)
  np.savez('../data/1D_validation_data', X_val, y_val)
  np.savez('../data/1D_test_data', X_test, y_test)