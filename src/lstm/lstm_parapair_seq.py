#!/usr/bin/python3

import math, json, os, sys
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed

def load_data(train_data_file, test_data_file):
    train_data = np.load(train_data_file)
    test_data = np.load(test_data_file)
    Xtrain = train_data[:, 0:len(train_data[0]) - 1]
    Xtrain = Xtrain.reshape((Xtrain.shape[0], 1, Xtrain.shape[1]))
    ytrain = train_data[:, -1]
    ytrain = ytrain.reshape((ytrain.shape[0], 1, 1))
    Xtest = test_data[:, 0:len(test_data[0]) - 1]
    Xtest = Xtest.reshape((Xtest.shape[0], 1, Xtest.shape[1]))
    ytest = test_data[:, -1]
    ytest = ytest.reshape((ytest.shape[0], 1, 1))
    print("Data loaded")
    print("Train Positive samples: {}".format(np.count_nonzero(ytrain.flatten()))+
          "Negative samples: {}".format(ytrain.flatten().size - np.count_nonzero(ytrain.flatten())))
    print("Test Positive samples: {}".format(np.count_nonzero(ytest.flatten())) +
          "Negative samples: {}".format(ytest.flatten().size - np.count_nonzero(ytest.flatten())))
    return Xtrain, ytrain, Xtest, ytest

def train_model(Xtrain, ytrain, num_epoch, num_train_sample):
    model = Sequential()
    model.add(LSTM(10, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # train LSTM
    for epoch in range(num_epoch):
        model.fit(Xtrain, ytrain, epochs=1, batch_size=num_train_sample, verbose=2)
    return model

def eval_model(Xtrain, ytrain, Xtest, ytest, num_test_sample, model):
    # evaluate LSTM
    yhat = model.predict_classes(Xtest, verbose=0)
    train_eval = model.evaluate(Xtrain, ytrain)
    test_eval = model.evaluate(Xtest, ytest)
    print(train_eval)
    print(test_eval)
    for i in range(num_test_sample):
        print('Expected:', ytest[i].flatten(), 'Predicted', yhat[i].flatten())

def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    num_epochs = int(sys.argv[3])
    Xtrain, ytrain, Xtest, ytest = load_data(train_file, test_file)
    num_train_sample = Xtrain.shape[0]
    num_test_sample = Xtest.shape[0]
    m = train_model(Xtrain, ytrain, num_epochs, num_train_sample)
    eval_model(Xtrain, ytrain, Xtest, ytest, num_test_sample, m)

if __name__ == '__main__':
    main()