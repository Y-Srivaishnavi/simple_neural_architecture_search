from randomvariables import target_classes

def vocab_dict():
    """
    Creates a dictionary of possible layers for a neural networks, using pre-defined kernel and filter sizes.
    
    Returns
    -------
    dict
        A search-space dictionary containing unique combinations of (filter size, kernel size, number of nodes, activation function)
    
    Notes
    -----
    For simplicity, activation function is always relu, strides and padding are one, and dropout size is always 0.5
    """
    # Define search space of parameters
    filter_size = [3,5,9,11,13,15,17,19,21,23,25,27,29,31]
    kernel_size = [3,5,9,11,13,15,17,19,21,23,25,27,29,31]
    nodes = [4,8,10,12,16,19,25,50,150,200,300,500,1]
    activations = ['relu', 'sigmoid', 'softmax']

    # Create a dictionary out of IDs and parameters
    vocab = {}
    id = 100
    
    # Create Cartesian product of all filter and kernel sizes, and node combinations, and assign UID to each tuple
    for filter in filter_size:
        for kernel in kernel_size:
            for node in nodes:
                id += 1
                if node != 1:
                    activation = activations[0]
                else:
                    activation = activations[1] if target_classes == 2 else activations[2]
                      
                vocab[id] = (filter, kernel, node, activation)
    
    return vocab

def encode_sequences(sequence, vocab):
    """
    Function to encode an architecture sequence.
    """
    values = list(vocab.values())
    encoded_sequences = []
    for value in sequence:
        encoded_sequences.append(values[value-1])
    return encoded_sequences

def decode_sequence(sequence, vocab):
    """
    Function to decode an architecture sequence.
    """
    keys=list(vocab.keys())
    values = list(vocab.values())
    decoded_sequence = []
    for key in sequence:
        decoded_sequence.append(values[keys.index(key)])
    return decoded_sequence
