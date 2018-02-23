import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    #the length of the series
    n_elements = len(series)

    # containers for input/output pairs
    X = []
    y = []

    #append x and y values
    for i in range(n_elements - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1), return_sequences=False))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    #use hex to represent punctuations
    ser_number = [p.encode('utf-8').hex() for p in punctuation]
    #zip them
    dic = dict(zip(punctuation, ser_number))、
    #replace them
    for p in punctuation:
        text = text.replace(p, ' '+str(dic[p]))
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    n_elements = len(text)

    #append x and y values
    for i in range(0, n_elements - window_size, step_size):
        inputs.append(str(text[i:i+window_size]))
        outputs.append(text[i+window_size])

    return inputs,outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars), return_sequences=False))
    model.add(Dense(num_chars, activation='softmax'))
    return model
