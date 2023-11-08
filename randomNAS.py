import randomvariables as rv
import randomgeneratemodel as rgm
from randomsearchspace import vocab_dict
import numpy as np
import pickle
import os

#location to log data

def randomsearch(X_train,y_train,X_test,y_test) -> None:
    """
    Generate the model, evaluate and log it.

    Parameters
    -------
    X_train
        Input training data.
    y_train
        Output of training data.
    X_test
        Input testing data.
    y_test
        Output of testing data.
    """
    # Dictionary to store all combinations to check later 
    samples = {}

    #intialize the search space
    vocab = vocab_dict()

    #iterate over the number of samples we want to generate randomly and evaluate
    for _ in range(rv.numsamples):
        # Pick combination randomly from vocab, and delete combination to avoid repetition
        id = np.random.randint(0, len(vocab))
        sample = vocab[id]
        del vocab[id]

        # Generate model
        model = rgm.randommodelgenerate(*sample, X_train.shape[1], rv.target_classes)

        # Train the model
        history = model.fit(
            X_train,
            y_train, 
            rv.batch_size, 
            rv.epochs, 
            validation_data =(X_test, y_test),
            callbacks=None, 
            verbose=0
        )

        # Store test accuracy for the combination
        if(len(history.history['val_accuracy'])==1):
            valacc = [history.history['val_accuracy'][0]]
        else:
            valacc = np.ma.average(history.history['val_accuracy'], weights = np.arange(1,len(history.history['val_accuracy'])+1),axis=-1)

        print(f"Architecture is {sample}")
        print(f"Validation Accuracy is {valacc}")  

        # Save data to samples
        samples[sample] = valacc
    try:
        with open(rv.nas_data_log, 'wb') as f:
            pickle.dump(samples, f)  
    except FileNotFoundError:
        os.mkdir('LOGS')
        with open(rv.nas_data_log, 'x') as f:
            pickle.dump(samples, f)
