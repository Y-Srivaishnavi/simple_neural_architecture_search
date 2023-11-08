import tensorflow as tf

def randommodelgenerate(nodes: int, activation: str, input_shape: tuple, target: int):
    """
    Generates single-layered neural network from given parameters.

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
        model.add(tf.keras.Input(shape=input_shape)),
        tf.keras.layers.Dense(nodes, activation=activation),
        tf.keras.layers.Dense(target, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model                

    
        
