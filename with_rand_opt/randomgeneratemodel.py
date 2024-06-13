import tensorflow as tf
from randomvariables import metrics

def randommodelgenerate(nodes: tuple, activation: str, input_shape: int, target: int):
    """
    Generates triple-layered neural network from given parameters.

    Parameters
    -------
    nodes: int
        Number of nodes in hidden layer
    activation: str
        Activation function to used in model
    input_shape: tuple
        Shape of input layer
    target: int
        Number of target classes

    Returns
    -------
    Sequential
        Tensorflow model with given parameters
    
    Notes
    -----
    For simplicity, dropout size is always 0.5
    """
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(None, input_shape)),
        tf.keras.layers.Dense(nodes[0], activation=activation),
        tf.keras.layers.Dense(nodes[1], activation=activation),
        tf.keras.layers.Dense(1, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    return model                

    
        
