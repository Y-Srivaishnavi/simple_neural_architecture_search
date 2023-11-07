from randomsearchspace import decode_sequence
import randomvariables as rv
import tensorflow as tf
import numpy as np


def randommodelgenerate(sequence, vocab , mlp_input_shape):
    #print("input shape :",mlp_input_shape[0])
    #variable to track flatten layer
    track=0
    #decode sequence to get details of the layers (nodes,activation)
    layer_configs = decode_sequence(sequence, vocab)
    #print(layer_configs)
    #create sequential model
    model= tf.keras.models.Sequential()

    #flatten the input layer for mutli-dimensional inputs >2
    if(len(mlp_input_shape)>1):
        #print("1")
        model.add(tf.keras.layers.Flatten(name='flatten', input_shape=mlp_input_shape))
        #for each element in the layer_configs
        for i, layer_conf in enumerate(layer_configs):
            #add a model layer (Dense or dropout)
            if layer_conf == 'dropout' :
                model.add(tf.keras.layers.Dropout(rv.mlp_dropout, name='dropout'))
            else:
                if(layer_configs[1] not in ("relu", "softmax","sigmoid")):
                    if(track==0):
                        model.add(tf.keras.layers.Flatten())
                        model.add(tf.keras.layers.Dense(units=layer_conf[0], activation = "relu"))
                        track=1
                    else:
                        model.add(tf.keras.layers.Dense(units=layer_conf[0], activation = "relu"))

                else:
                    model.add(tf.keras.layers.Conv1D(layer_conf[0],layer_conf[1],activation="relu"))
    
    else:
        #for 2d inputs
        for i,layer_conf in enumerate(layer_configs):
            #print("2")
            # print("i: ", i )
            # print("layer_conf: ", layer_conf)

            #add input layer
            if i==0:
                if(layer_conf[1]=="relu"):
                    model.add(tf.keras.layers.Dense(units=layer_conf[0],activation="relu",input_shape=mlp_input_shape))
                    track = 1
                else:
                    model.add(tf.keras.layers.Conv1D(layer_conf[0],layer_conf[1],activation="relu",input_shape = (mlp_input_shape[0],1)))
                    firstlayer="conv1d"

            #add subsequent layers
            elif layer_conf == 'dropout':
                model.add(tf.keras.layers.Dropout(rv.mlp_dropout, name='dropout'))
            else:
                if(layer_conf[1] in ("relu","sigmoid","softmax")):
                    if(track==0):
                        model.add(tf.keras.layers.Flatten())
                        model.add(tf.keras.layers.Dense(units=layer_conf[0], activation = "relu"))
                        track=1
                    else:
                        model.add(tf.keras.layers.Dense(units=layer_conf[0], activation = "relu"))

                else:
                    model.add(tf.keras.layers.Conv1D(layer_conf[0],layer_conf[1],activation="relu"))
    #print(model.summary())
    
    optim = tf.keras.optimizers.Adam(lr=rv.mlp_lr)
    model.compile(loss=rv.mlp_loss_func,optimizer=optim,metrics=rv.metrics)

    return model

#function to train the model
def trainmodel(model,x_data,y_data,x_val,y_val,callbacks=None):
    return model.fit(x_data,y_data, rv.batch_size, rv.epochs, validation_data =(x_val,y_val),callbacks= callbacks, verbose=0)

#function to return validation accuracy
def valmodel(history):
    if(len(history.history['val_accuracy'])==1):
        return [history.history['val_accuracy'][0]]
    else:
        return np.ma.average(history.history['val_accuracy'], weights = np.arange(1,len(history.history['val_accuracy'])+1),axis=-1)
                

    
        