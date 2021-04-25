# Music-Genre-Classification
## Table of Contents
[Overview](#overview)<br/>
[Introduction](#intro)<br/>
[What is Audio Signal Processing?](#asp)<br/>
[Data Preparation](#data-prep)<br/>
[Exploratory Data Analysis](#EDA)<br/>
[Deep Learning Models](#deep)<br/>
[Summary](#summary)<br/>
[Future Ideas](#future)<br/>
[References](#references)<br/>

## <a name="overview">Overview</a>

Music is everywhere. You can find yourself hearing all kinds of music as you go throughout your day - maybe some soft indie-rock at your local coffee shop, the latest pop songs over the radio, some electronic dance music in your fitness class. The options are endless. And even so, music continues to grow, adapt, and expand. New genres and subgenres continue to pop up and offer new music.

In this ever-expanding space, it's important to be able to classify songs to be able to group and categorize them in meaningful ways. As such, this project aims to tackle the popular problem of genre classification.

In a [previous project](https://github.com/kenshr/Music-Genre-Classification-with-Audio-Features) I addressed the same issue using audio features of songs, which were gathered from Spotify's web API, in order to determine their genres. While this project proved to be largely successful, I wanted to build on the same idea and make my models more adaptable. Rather than using audio features that need to be generated via a third party, this project will use deep learning models to perform genre classification on songs using raw audio.

The dataset<sup>[1](https://github.com/mdeff/fma)</sup> I will be using for this project has 30 second clips of 8000 songs, equally balanced across 8 genres:
-




## <a name="asp">What is Audio Signal Processing?</a>

This section will serve as a primer to introduce some technical concepts specific to audio signal processing that were employed in this project. If you are mainly interested in the strictly data science aspects of this project, please feel free to skip to the [next section](#data-prep).

One of the first issues to address is how to properly feed the audio as input into our models. Audio, in essence, is the product of variations in air pressure over time. These changes are recorded at a specified sample rate traditionally 44,100 samples per second (44.1 kHz), and give us waveforms. As seen below, these waveforms are rich with information and can vary in shape depending on sample rate.

![<img src='imgs/waveform_visualization.gif'>](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)

These waveforms are what you and I as everyday listeners hear when we listen to mp3 and wav files.

a palatable data format that can be used as input into our deep learning models.

The first point to discuss is mel spectrograms. Spectrograms are visual representations of a spectrum of frequencies from a signal as it varies over time<sup>[1](https://en.wikipedia.org/wiki/Spectrogram#:~:text=A%20spectrogram%20is%20a%20visual,they%20may%20be%20called%20waterfalls.)</sup>. Put simply, a spectrogram is a picture depicting audio visually.

The "mel" component of the spectrogram refers to the mel scale,



## <a name="data-prep">Data Preparation</a>



## <a name="EDA">Exploratory Data Analysis</a>



## <a name="deep">Deep Learning Models</a>



## <a name="summary">Summary</a>



## <a name="future">Future Ideas</a>



## <a name="references">References</a>

1. [Dataset](https://github.com/mdeff/fma)
2. [What is a Spectrogram?](https://en.wikipedia.org/wiki/Spectrogram#:~:text=A%20spectrogram%20is%20a%20visual,they%20may%20be%20called%20waterfalls.)
2.