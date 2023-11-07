from randomvariables import max_len
import tensorflow as tf
import numpy as np


def randomsequencegenerate(vocab, samples):
    final_layer_id = len(vocab)
    dropout_id = final_layer_id - 1
    vocab_idx = [0]+list(vocab.keys())
    
    
    print("GENERATING ARCHITECTURE SEQUENCE...")
    print('------------------------------------------------------')
    # initialise the empty list for architecture sequence
    seed = []
    # while len of generated sequence is less than maximum architecture length
    while len(seed) < max_len:
            # pad sequence for correctly shaped input for controller
            sequence = tf.keras.utils.pad_sequences([seed], maxlen=max_len - 1, padding='post')
            sequence = sequence.reshape(1, 1, max_len - 1)
            #sampling from uniform distribution
            # print(vocab_idx)
            # print(np.random.choice(vocab_idx))
            next = np.random.choice(vocab_idx)
            

            if next ==  dropout_id and len(seed) == 0 :
                continue
            
            if next == final_layer_id and len(seed) == 0:
                continue
            
            if next == final_layer_id:
                seed.append(next)
                break
            
            if len(seed) == max_len - 1 :
                seed.append(final_layer_id)
                break
            

            #add condition to not allow conv layers after dense layers
            if(len(seed) >=1):
                if (next < 197 and seed[-1]>=197):
                    continue
                    
                else:
                    if not next == 0: 
                        seed.append(next)
            
            if len(seed) == max_len - 1 :
                seed.append(final_layer_id)
                break
            
            
            
            if not next == 0: 
                seed.append(next)
        #print("seed: ", seed)
            

        # check if the generated sequence has been generated before.
        # if not, add it to the sequence data. 
    if seed not in samples:
        return seed
    else:
        return "NA"

