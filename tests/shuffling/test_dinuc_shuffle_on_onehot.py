from deeplift.dinuc_shuffle  import dinuc_shuffle
from dragonn.utils import get_sequence_strings
import random
import numpy as np

import wget
url="http://mitra.stanford.edu/kundaje/projects/dragonn/deep_lift_input_classification_spi1.npy"
wget.download(url)
deep_lift_input_classification_spi1=np.load("deep_lift_input_classification_spi1.npy")
print(deep_lift_input_classification_spi1.shape)
deep_lift_input_classification_spi1_strings=get_sequence_strings(deep_lift_input_classification_spi1)

for i in range(len(deep_lift_input_classification_spi1)): 
    random.seed(1234)
    shuffled_strings=dinuc_shuffle(deep_lift_input_classification_spi1_strings[i])
    random.seed(1234)
    shuffled_array=dinuc_shuffle(deep_lift_input_classification_spi1[i].squeeze())
    #decode the array
    shuffled_array=''.join(get_sequence_strings(np.expand_dims(np.expand_dims(shuffled_array,axis=1),axis=1)))
    #make sure shuffling the string and numpy array gave same shuffle output
    if (shuffled_strings != shuffled_array):
        print("FAILED!")
print("TEST PASSED!") 
