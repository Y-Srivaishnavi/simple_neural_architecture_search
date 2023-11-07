############ Search Space ################
target_classes = 2


############ NAS ################
numsamples = 50


############ GENERATE SEQUENCE ################
max_len = 3


############ GENERATE MODEL ################
mlp_lr = 0.01
mlp_loss_func = 'categorical_crossentropy'
metrics = ['accuracy']
batch_size = 256
epochs = 150
mlp_dropout = 0.5


############ LOG -RANDOMRUN.PY ################
nas_data_log = 'LOGS/generated.pkl'
