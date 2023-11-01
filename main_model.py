import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy

# Define a function to evaluate a given neural network architecture
def evaluate(architecture):
    # Create a new Keras model instance using the given architecture
    model = Sequential()
    model.add(Dense(units=architecture['hidden_layer_sizes'][0], input_shape=(input_dim,)))
    model.add(Dense(units=architecture['hidden_layer_sizes'][1], activation='relu'))
    model.add(Dense(units=architecture['hidden_layer_sizes'][2], activation='relu'))
    model.add(Dense(units=output_dim, activation='softmax'))
    
    # Compile the model with a suitable optimizer and loss function
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model on the training data
    model.fit(X_train, to_categorical(y_train), epochs=50, batch_size=32, validation_data=(X_test, to_categorical(y_test)))
    
    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(X_test, to_categorical(y_test))
    
    # Return the evaluation metric (in this case, accuracy)
    return test_acc

# Define a list of possible neural network architectures
architectures = [
    {'hidden_layer_sizes': [20, 20, 20]},
    {'hidden_layer_sizes': [20, 30, 20]},
    {'hidden_layer_sizes': [30, 20, 20]},
    {'hidden_layer_sizes': [30, 30, 20]},
    {'hidden_layer_sizes': [30, 30, 30]}
]

# Perform neural architecture search using a random search strategy
random_search = RandomizedSearchCV(estimator=evaluate, param_grid=architectures, n_iter=10, cv=5, scoring='accuracy')
random_search.fit(X, y)

# Print the best neural network architecture found by the search
print('Best architecture:', random_search.best_params_)
