import unittest
from deeplift.dinuc_shuffle import dinuc_shuffle
from itertools import permutations
from collections import defaultdict
import random

def dinuc_count(seq):
    count = defaultdict(lambda: 0) 
    for i in range(len(seq)-2):
        count[seq[i:i+2]] += 1
    return count

class TestDinucShuffle(unittest.TestCase):

    def test_dinuc_shuffle(self):
        for i in range(1000):
            random_sequence = "".join([['A','C','G','T'][int(random.random()*4)]
                                    for i in range(200)])
            shuffled_seq = dinuc_shuffle(random_sequence)
            print("sequences")
            print(random_sequence)
            print(shuffled_seq)
            orig_count = dinuc_count(random_sequence)
            shuffled_count = dinuc_count(shuffled_seq)
            print("counts")
            print(orig_count)
            print(shuffled_count)
            assert len(orig_count.keys())==len(shuffled_count.keys())
            for key in orig_count:
                assert orig_count[key]==shuffled_count[key]
