############ Search Space ################
# set number of classes to be predicted by model
target_classes = 2  

############ NAS ################
# set the number of neural network models to be generated during the search process.
numsamples = 50

############ GENERATE SEQUENCE ################
# Set maximum number of layers in a sequence
max_len = 3

############ GENERATE MODEL ################
# Set the learning rate, loss function, evaluation metric, number of epochs and dropout probability for the Multilayer Perceptron (MLP)
mlp_lr = 0.01
mlp_loss_func = 'categorical_crossentropy'
metrics = ['accuracy']
batch_size = 256
epochs = 150
mlp_dropout = 0.5

############ LOG -RANDOMRUN.PY ################
# File pointer to store logs
nas_data_log = 'LOGS/generated.pkl'
