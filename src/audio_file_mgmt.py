import os

from pydub import AudioSegment
import librosa

# This class is currently built to primarily support working with the fma_small dataset but
# has the tools necessary to be refactored to handle all datasets. Will try to revisit this
# when I have more time.

class AudioManagement:
  def __init__(self, audio_dir = '../data/fma_small/', metadata_path = '../data/fma_metadata/tracks.csv', split_dir_name = 'split'):
    self.audio_dir = audio_dir
    self.metadata_path = metadata_path
    self.split_dir_name = split_dir_name
    self.split = False

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


  def make_spectrogram(self, track_id):
    """
    Takes a track_id, identifies the corresponding mp3's
    file path, and returns a spectrogram of the mp3.

    Parameters:
    -----------
    track_id (int):

    Returns:
    --------

    """
    # Format track id to match filename
    tid = '{:06d}'.format(track_id)
    # Create file path to mp3
    path = os.path.join(self.audio_dir, tid_str[:3], tid_str + '.mp3')
    # Generate spectrogram from mp3
    y, sr = librosa.load(path)
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T


  def create_array(df):
    genres = []
    X_spect = np.empty((0, 640, 128))
    count = 0
    #Code skips records in case of errors
    for index, row in df.iterrows():
      try:
        count += 1
        track_id = int(row['track_id'])
        genre = str(row[('track', 'genre_top')])
        spect = create_spectogram(track_id)
        # Normalize for small shape differences
        spect = spect[:640, :]
        X_spect = np.append(X_spect, [spect], axis=0)
        genres.append(dict_genres[genre])
        # if count % 100 == 0:
        #   print("Currently processing: ", count)
      except:
        print("Couldn't process: ", count)
        continue
    y_arr = np.array(genres)
    return X_spect, y_arr

  def array_


if __name__=="__main__":
    # am = AudioManagement()
    pass
