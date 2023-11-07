from randomvariables import numsamples
from randomgeneratesequence import randomsequencegenerate
import randomgeneratemodel as rgm
from randomsearchspace import vocab_dict, encode_sequences
import numpy as np

#location to log data

def randomsearch(x_data,y_data,x_val,y_val):
    #generate the sequence and evalaute and log it

    #list to store data for each iteration
    data =[]
    return_data =[]

    #list to store all sequences to check later if any sequences were generated more than once
    samples = []

    #intialize the search space
    vocab = vocab_dict()

    #iterate over the number of samples we want to generate randomly and evaluate
    for i in range(numsamples):

        #generate sequence
        sequence = randomsequencegenerate(vocab,samples)

        #generate model
        if sequence != "NA":
            #print(np.shape(x_data[0]))
            model = rgm.randommodelgenerate(sequence,vocab, np.shape(x_data[0]))
            #train the model
            history = rgm.trainmodel(model,x_data,y_data,x_val,y_val)
            #store the val accuracy for the sequence
            valacc = rgm.valmodel(history)
            #append the results to data and sample
            samples.append(sequence)
            data.append((sequence,valacc))
            #print the data
            #print(data[i][0])
            print("Architecture is ", encode_sequences(data[i][0],vocab))
            print("Validation Accuracy is ", data[i][1])
            return_data.append([encode_sequences(data[i][0],vocab),data[i][1]])
        else:
            data.append("NA")
            print("Sequence repeated")

    return return_data