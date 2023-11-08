def vocab_dict() -> list:
    """
    Generates search space of hidden layers and activation function combinations.
    
    Returns
    -------
    list
        A search-space list containing unique combinations of (number of nodes, activation function)
    
    Notes
    -----
    For simplicity, dropout size is always 0.5
    """
    # Define search space of  for image data, taking activation function = "relu"
    # filter_size = [3,5,9,11,13,15,17,19,21,23,25,27,29,31]
    # kernel_size = [3,5,9,11,13,15,17,19,21,23,25,27,29,31]
    # nodes = [4,8,10,12,16,19,25,50,150,200,300,500,1]

    nodes = [4,8]
    activations = ['linear','relu']

    # Create a dictionary out of IDs and parameters
    vocab = []
    
    # Create Cartesian product of all node combinations, and assign UID to each tuple
    for node in nodes:
        for activation in activations:                  
            vocab.append((node, activation))
    
    return vocab

