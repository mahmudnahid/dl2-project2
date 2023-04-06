# AASD4015 - Advanced Mathematical Concepts for Deep Learning

## Project: Generating Music with Deep Learning

#### Github Pages Link: https://mahmudnahid.github.io/dl2-project2/


### Team Members:
- Khandaker Nahid Mahmud (101427435)
- Siddhant Gite (101359755)

# Problem Statement: 

Generating long pieces of music is a challenging problem, as music contains structure at multiple timescales, from milisecond timings to motifs to phrases to repetition of entire sections. In this project we trained 2 models on the Bach chorales dataset to generate Bach-like music. This is an excercise problem from chapter 15 of the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron. The exercise is as follow:

Download the [Bach chorales](https://github.com/ageron/data/tree/main/jsb_chorales) dataset and unzip it. It is composed of 382 chorales composed by Johann Sebastian Bach. Each chorale is 100 to 640 time steps long, and each time step contains 4 integers, where each integer corresponds to a note's index on a piano (except for the value 0, which means that no note is played). Train a model—recurrent, convolutional, or both—that can predict the next time step (four notes), given a sequence of time steps from a chorale. Then use this model to generate Bach-like music, one note at a time: you can do this by giving the model the start of a chorale and asking it to predict the next time step, then appending these time steps to the input sequence and asking the model for the next note, and so on. 


But along with the CNN model as suggested, we implemented two models for generating music:

1. CNN & LSTM based Model
2. Transformer based Model

# Introduction

A musical piece often consists of recurring elements at various levels, from motifs to phrases to sections such as verse-chorus. To generate a coherent piece, a model needs to reference elements that came before, sometimes in the distant past, repeating, varying, and further developing them to create contrast and surprise. But before we delve into the technical implementation  let us understand the building blocks of music:

<img src="https://raw.githubusercontent.com/mahmudnahid/dl2-project2/main/images/piano.png" width="500" />


Labels of the notes are (in sharp, #, notation):

```
  C#   D#        F#   G#   A#
C    D    E    F    G    A    B ...
```

Labels of the notes are (in flat, $\flat$, notation):

```
  Db   Eb        Gb   Ab   Bb
C    D    E    F    G    A    B ...
```


The A in the 4th octave is typically tuned at 440 Hz

* A half step is the smallest increment you can make
* After twelve half-steps you're back to the same note, but it sounds exactly twice as high
* In standard piano tuning, the frequency is multiplied by ${}^{12}\sqrt{2} \approx 1.059$

Frequency of note is implemented as:

$$f = f_{\mathrm{A4}}\bigg( {}^{12}\sqrt{2} \bigg)^ N $$

where $N$ is the number of steps needed (can be negative) to move from A4 to the desired note.

**Scale:**
A scale is a selection of notes that fit well together.

**Chords:**
A chord is any harmonic set of pitches/frequencies consisting of multiple notes that are heard as if sounding simultaneously.

**Arpeggio:**
An arpeggio is a type of broken chord in which the notes that compose a chord are individually sounded in a progressive rising or descending order.



#### Reference: 
- https://www.youtube.com/watch?v=hXrpV2ffJRU&ab_channel=JustinGuitar
- https://github.com/marcelraas/music-generator/blob/master/presentation/2-music-generation.ipynb


Now that we know the building blocks of music, let us understand how we can generate music with deep learning. We take a language-modeling approach to training generative models for symbolic music. Hence we represent music as a sequence of discrete tokens, with the vocabulary determined by the dataset.

The JSB Chorale dataset consists of four-part scored choral music, which can be represented as a matrix where rows correspond to chords and columns to time discretized notes. The matrix’s entries are integers that denote which pitch is being played. Notes range from 36 (C1 = C on octave 1) to 81 (A5 = A on octave 5), plus 0 for silence:

<img src="https://raw.githubusercontent.com/mahmudnahid/dl2-project2/main/images/chorale.JPG" width="250" />

This is very similar to time-series data or word sequence data  in NLP. So we took a sequence to sequence modeling approach for generating the output note sequences. Each chorale will be a long sequence of notes (rather than chords), and we can just train a model that can predict the next note given all the previous notes. We will feed a window to the neural net, and it tries to predict that same window shifted one time step into the future.
    

# Dataset

### Bach chorales:

The dataset is composed of 382 chorales composed by Johann Sebastian Bach. Each chorale is 100 to 640 time steps long, and each time step contains 4 integers, where each integer corresponds to a note's index on a piano (except for the value 0, which means that no note is played).

The dataset is available here: https://github.com/ageron/data/tree/main/jsb_chorales


# Summary of Findings
In our experiment CNN+LSTM model and Transformer Model achieved accuracy score 0.815 and 0.812 respectively. Though the accuracy score of both is in the same range yet from the graphs we found that if the sequence is long then the Transformer model might loose long-term coherence, as shown in the following graphs:

<img src="https://raw.githubusercontent.com/mahmudnahid/dl2-project2/main/images/cnn_cold.JPG" width="500" />

Fig: Generated chorale by CNN+LSTM model 

<img src="https://raw.githubusercontent.com/mahmudnahid/dl2-project2/main/images/transformer_cold.JPG" width="500" />

Fig: Generated chorale by Transformer model 

While the Transformer allows us to capture self-reference through attention, it relies on absolute timing signals and thus has a hard time keeping track of regularity that is based on relative distances, event orderings, and periodicity. 


# Reference
- Chapter 15, [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron
- Chapter 12, [Deep Learning with Python](https://www.amazon.ca/Deep-Learning-Python-Francois-Chollet/dp/1617294438) by Francois Chollet
- https://www.youtube.com/watch?v=hXrpV2ffJRU&ab_channel=JustinGuitar
- https://github.com/marcelraas/music-generator/blob/master/presentation/2-music-generation.ipynb
- https://magenta.tensorflow.org/music-transformer

