# Character Level Langauge Model

Deep Learning Assignment2, 인공신경망과 딥러닝 과제2  
This Assigment is work on a neural network for character-level language modeling with the Shakespeare dataset. The language model is "many-to-many" recurrent neural networks.
<br></br>

## Contents
In this repository, I have contributed by modifying the code to run on CUDA. Additionally, I conducted a study comparing the performance of single-layer RNN and LSTM models with two-layer RNN and LSTM models, and documented the results in the result_file.

The results indicate that LSTM models outperform RNN models. Furthermore, two-layer LSTM models show better performance than single-layer LSTM models, as evidenced by the continuous decrease in validation loss.

The result_files has text files which contain generated charaters length over 100 length. Each text file's name has a number which indicates T(temperature parameter of Softmax function). Range of T is 1~5. T1 has least entropy and T5 has most entropy. 

<br></br>

## Softmax Temperature

![image](https://github.com/jewookwak/Language-Modeling/assets/65706899/58486d44-ef78-41a9-b1d8-45c4ab6fd830)

Temperature scaling (T) adds randomness and controls the unpredictability of the next word. It adjusts the entropy of the probability distribution used for sampling, determining how surprising or predictable the next word will be.    
Please check [here](https://medium.com/@harshit158/softmax-temperature-5492e4007f71), if you interested in softmax temperature.

<br></br>


## Character level langauge model

It has the same structure as the existing RNN model, but has one alphabetic character, not a token, as an input. At this time, characters include spaces, paragraph changes, etc.

The structure of the character-level language model is shown in the figure below.

![structure](https://user-images.githubusercontent.com/42087965/140010696-65ff1b41-8a6c-4716-b1a2-4d28c1c580bb.png)

It takes a specific starting token as input and outputs a character that is likely to appear later. The output character is a recursive model that goes back into the input of the model and predicts the next character.

Please check [here](https://towardsdatascience.com/character-level-language-model-1439f5dd87fe), if you interested in character level language model.

<br></br>

## Dataset

The Shakespear's Literature Collection was used as the experimental dataset. It takes the form of a play of "Speaker: Dialogue", and all of them are used as input data. Thus, the newly created text also took the form of a play script.

In addition, paragraphs change for every line, and this was also learned with input data.

The training data used in the experiment can be found [here](https://github.com/Kiminjo/Character-level-language-model/blob/main/data/shakespeare_train.txt).

<br></br>

## Software Requirements

- python >= 3.5
- pytorch
- numpy 
- matplotlib

<br></br>

## Key Files

- `dataset.py` : It takes text data as input and converts it into a batch of tensor.
- `model.py` : A model used for training and sentence generation was defined. Two basic RNN models and LSTM models were used in the experiment.
- `main.py` : The main file of this project. Train the our character langauge model. In addition, the error rate was visualized using matplotlib.
- `generate.py` : A new sentence is generated using the model learned in the main file.
