# Music-Genre-Classification

## Table of Contents
[Overview](#overview)<br/>
[So what is Audio Signal Processing?](#asp)<br/>
[Data Preparation](#data-prep)<br/>
[EDA](#EDA)<br/>
[Modeling](#modeling)<br/>
[Summary](#summary)<br/>
[References](#references)<br/>

## <a name="overview">Overview</a>

Music is everywhere. You can find yourself hearing all kinds of music as you go throughout your day - maybe some soft indie-rock at your local coffee shop, the latest pop songs over the radio, some electronic dance music in your fitness class. The options are endless. And even so, music continues to grow, adapt, and expand. New genres and subgenres continue to pop up and offer new music.

In this ever-expanding space, it's important to be able to classify songs to be able to group and categorize them in meaningful ways. As such, this project aims to tackle the popular problem of genre classification.

In a [previous project](https://github.com/kenshr/Music-Genre-Classification-with-Audio-Features) I addressed the same issue using audio features of songs, which were gathered from Spotify's web API, in order to determine their genres. While this project proved to be largely successful, I wanted to build on the same idea and make my models more adaptable. Rather than using audio features that need to be generated via a third party, this project will use deep learning models to perform genre classification on songs using raw audio.

My workflow is outlined below:

<p align="center">
<img src='imgs/workflow.png'>
</p>

The dataset<sup>[**1**](https://github.com/mdeff/fma)</sup> used for this project has 30 second clips of 8000 songs, equally balanced across 8 genres:
- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

In order to use this data to train a deep learning model, we will need to convert it into a format that the model can use. For this project, we will use audio signal processing tools and concepts to convert the audio into images (which can be used to train models).

## <a name="asp">So what is Audio Signal Processing?</a>

This section will serve as a primer to introduce and demystify some audio signal processing concepts that were employed in this project. If you are only interested in strictly data science, please feel free to skip to the [next section](#data-prep).

Audio, in essence, is the product of variations in air pressure over time. These changes are recorded at a specified sample rate, traditionally 44,100 samples per second (44.1 kHz), which gives us waveforms. As seen in the image<sup>[**2**](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)</sup> below, these waveforms offer large amounts of information within the span of even 1 second.

<p align="center">
<img src='imgs/waveform_visualization.gif'>
</p>

These waveforms can be stored in mp3's, wav files, and other audio file formats. This brings us to our starting point, with our data being stored in a .mp3 file format. The first issue to address is how to convert our data into a palatable format that can be input into deep learning models, as they cannot process raw audio data. In order to overcome this, I use mel spectrograms like the one shown below.

<p align="center">
<img src="imgs/sample_melspect.png">
</p>

Spectrograms are visual representations of a spectrum of frequencies from a signal as it varies over time.<sup>[**3**](https://en.wikipedia.org/wiki/Spectrogram#:~:text=A%20spectrogram%20is%20a%20visual,they%20may%20be%20called%20waterfalls.)</sup> Put simply, a spectrogram is a picture depicting audio visually. The raw audio is converted into a spectrogram using a mathematical technique called the Fourier Transform. The Fourier transform takes a small window of the audio and breaks its down into constituent sine and cosine waves of corresponding amplitude. Doing this will convert our audio into the "spectrum of frequencies" mentioned above. The package we're using applies a Fast Fourier Tranform (FFT) algorithm multiple times across our passed-in audio to generate the spectrogram. This process can be seen in the image<sup>[**4**](https://www.mathworks.com/help/dsp/ref/dsp.stft.html)</sup>  below.

<p align="center">
<img src="imgs/fft_visualization.png" width="450">
</p>

The "mel" component of the mel spectrogram refers to the mel scale. The mel scale is a method of scaling pitches to make equal distances in pitch sound equally distant to listeners. This is necessary because humans are innately better at hearing differences in low frequencies than differences in high frequencies. The formula<sup>[**5**](https://en.wikipedia.org/wiki/Mel_scale)</sup> to convert frequencies into mels is:

$$m=2595\log _{10}\left(1+{\frac {f}{700}}\right)$$

Therefore, after generating the spectrogram, the frequency (y-axis) is mel-scaled accordingly. This can be seen in the graph above as the frequency follows a logarithmic trend as it increases.

Coming back to the big picture, the main idea behind using mel spectrograms for our deep learning models is that they are in a visual format which CAN be processed through a neural network and they are scaled in such a way that the data is represented in a format similar to how a human would perceive the sound. With this knowledge, we can now safely go into how the data was preprocessed and prepared for modeling.

## <a name="data-prep">Data Preparation</a>
Because we are dealing with audio files, the data could not simply be loaded into a dataframe and manipulated. Instead, I began by first writing scripts to properly access audio files from their nested directories. Thankfully, the creator of this dataset was very deliberate with their file structure and naming conventions, which made accessing the correct tracks easier with the os<sup>[**6**](https://docs.python.org/3/library/os.html)</sup> python package. Accessing specific files was important to be able to link songs with their associated metadata (i.e. genre) located on a separate .csv file. Once these scripts were functional, I was able to start accessing directories of audio to convert songs into mel spectrograms.

The data was pre-trimmed into 30 second samples so I was able to convert the data directly into mel spectrograms using the Librosa<sup>[**7**](https://librosa.org/doc/latest/index.html)</sup> python package. I then took the matrices representing the spectrograms and stored them in an array with their corresponding genre. This was the final format that was used to store the feature and target data for modeling.

Later on in the project, I decided to experiment with 1D convolutional neural networks in order to better analyze long-term temporal elements of songs. To do this, I took the mel spectrogram matrices and broke them into sections based on their frequency ranges. My thought process was that I could isolate different frequency ranges in the spectrogram that correspond to different instruments used in a song and convert them to time-based 1-dimensional input channels. The table<sup>[**8**](https://www.cuidevices.com/blog/understanding-audio-frequency-range-in-audio-design)</sup> below shows the frequency band thresholds that I used when creating my input channels.

| Frequency Subset | Frequency Range (Hz) |                               Instruments                              |
|:----------------:|:--------------------:|:----------------------------------------------------------------------:|
|     Sub-Bass     |         16-60        | bass guitar, tuba, bass                                                |
|       Bass       |        60-250        | speaking/vocal range                                                   |
|  Lower Midrange  |        250-500       | brass instruments and wind instruments like the saxophone and clarinet |
|     Midrange     |       500-2000       | higher frequency instruments like the violin and piccolo               |
|  Higher Midrange |       2000-4000      | harmonic frequencies of lower midrange instruments like the trumpet    |
|     Presence     |       4000-6000      | harmonic frequencies of midrange instruments like violin and piccolo   |
|    Brilliance    |      6000-20000      | high-pitched sounds like whistles, cymbals, and high harmonics         |

<br/>
Once the spectrograms were split into frequency range channels, I "flattened" each of the channels by averaging their frequencies for small incremental time splits to thus yield 7 channels of 1-dimensional frequency input that represent the song.

## <a name="EDA">EDA</a>
After preparing the data for modeling, I took some time to explore the metadata associated with the 8,000 songs from the dataset. The metadata consists of 51 features corresponding to anything from the engineer(s) who worked on the song to the song's wikipedia page (if any) to the song's licensing. It is quite sparse with just under a third of the features having information for less than 3,000 songs.

I was interested in seeing how the songs were distributed based on when they were made. Using the "date_created" feature, I found that the earliest song in the dataset was made on 11/25/2008 while the last song was made on 3/24/2017. I then graphed the number of songs that came from each year per genre. 2017 was omitted from the visualization since it contained very few tracks.

<p align="center">
<img src="imgs/tracks_per_genre_per_year.png">
</p>

We can see that the 1000 songs per genre are fairly evenly distributed over the years from 2008 to 2016. The only major deviation is that almost half of the songs in the Instrumental category came from 2015. Hopefully, having such evenly distributed data over a 10 year period will make the model more robust and less susceptible to signal coming from changes in genre trends over time.

<p align="center">
<img src="imgs/avg_length_per_genre.png">
</p>

I also plotted the average song length per genre. There was a surprising amount of variation with the longest genre, International, being over 50 seconds longer than the shortest genre, Hip-Hop. For our purposes, this should not have any effect on our modeling since we're using 30 second samples for each song.

## <a name="modeling">Modeling</a>
I began modeling by establishing a baseline using a dummy classifier which made predictions at random. I elected to use accuracy as my main metric for gauging performance since we want the models to predict all genres equally well and there are no differing rewards/penalties associated with the accuracy for predicting any one genre. Given that we have 8 genres, it makes sense that the dummy classifier had an accuracy of 13% (roughly one in eight).

I made my first model starting with a  2D convolutional neural network (CNN) to process the mel spectrogram images. With this, I was able to achieve a max accuracy of 40% on the validation set.

After hitting a wall, I pivoted to the 1D convolutional neural network approach mentioned above in the data preparation section. This new configuration was able to push the needle forward to an accuracy of 43% on the test set. Another added benefit of this configuration was that training time on the neural network was cut down by 75% since it was processing much less data (1D vs 2D).

The final model configuration utilized the same 1D convolutional neural network and added an LSTM layer to the architecture to better capture long-term temporal trends in the audio. This way, the initial CNN layers could capture different feature elements of the song which would then be fed into the LSTM to discern longer patterns such as rhythm and cadence.

<p align="center">
<img src="imgs/CNN_LSTM_modelsummary.png">
</p>

After testing various neural network architectures and performing hyper-parameter tuning, the final model yielded a maximum accuracy of 49% on the validation set at 40 epochs.

<p align="center">
<img src="imgs/model_loss.png">
</p>

<p align="center">
<img src="imgs/model_accuracy.png">
</p>

## <a name="summary">Summary</a>

Overall, this project was a great success. We took 30 second clips of 8,000 songs spread evenly across 8 genres and used them to train deep learning models. Starting from a baseline of 12.5% accuracy, the trained CNN-LSTM model was able to improve performance by 36% with a maximum validation accuracy of 49%. During this process, I also learned a lot about (audio) signal processing, which is its own field of study altogether.

I firmly believe that this model, in its current architecture, can perform much better if it is given a larger dataset to train with. 8,000 data points is not much to work with, especially for training neural networks. Some next steps for this project would be using the larger, imbalanced datasets from FMA<sup>[**1**](https://github.com/mdeff/fma)</sup> to see how the model performs. I'd also like to toy around with partitioning the audio samples into smaller snippets to increase sample size, although that  may affect the model's ability to study long-term temporal effects in the audio.

## <a name="references">References</a>

1. [Dataset](https://github.com/mdeff/fma)
2. [Waveform Visualization](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
3. [Spectrogram Definition](https://en.wikipedia.org/wiki/Spectrogram#:~:text=A%20spectrogram%20is%20a%20visual,they%20may%20be%20called%20waterfalls.)
4. [Fast Fourier Transform Visualization](https://www.mathworks.com/help/dsp/ref/dsp.stft.html)
5. [Mel Scale Formula](https://en.wikipedia.org/wiki/Mel_scale)
6. [os Documentation](https://docs.python.org/3/library/os.html)
7. [Librosa Documentation](https://librosa.org/doc/latest/index.html)
8. [Frequency Range Thresholds](https://www.cuidevices.com/blog/understanding-audio-frequency-range-in-audio-design)