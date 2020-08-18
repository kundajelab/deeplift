from __future__ import division, print_function
import numpy as np

def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)


def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")


def one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]


def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """
    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if not rng:
        rng = np.random.RandomState()
   
    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token
 
    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)
       
        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]


if __name__ == "__main__":
    from datetime import datetime

    def bench(
        seq_len=1000, num_seqs=500, num_shufs=10, seed=1234, one_hot=False,
        vectorize=True
    ):
        rng = np.random.RandomState(seed)
        times = []
    
        if one_hot:
            seqs = [
                tokens_to_one_hot(rng.choice(4, seq_len), 4)
                for _ in range(num_seqs)
            ]
        else:
            seqs = [
                "".join(rng.choice(["A", "C", "T", "G"], seq_len))
                for _ in range(num_seqs)
            ]
    
        total_start = datetime.now()
        results = []
        for seq in seqs:
            start = datetime.now()
            results.append(dinuc_shuffle(seq, num_shufs, rng))
            end = datetime.now()
            delta = (end - start).total_seconds() * 1000
            times.append(delta)
        total_end = datetime.now()

        print("Total time: %.2fs" % ((total_end - total_start).total_seconds()))
        print("Average time for each sequence: %.2fms" % np.mean(times))
        return results

    def dinuc_content(seq):
        # Strings only
        counts = {}
        for i in range(len(seq) - 1):
            try:
                counts[seq[i:i + 2]] += 1
            except KeyError:
                counts[seq[i:i + 2]] = 1
        return counts

    def one_hot_to_dna(one_hot):
        return "".join(
            np.array(["A", "C", "G", "T"])[one_hot_to_tokens(one_hot)]
        )

    def dna_to_one_hot(dna):
        return np.identity(4)[
            np.unique(string_to_char_array(dna), return_inverse=True)[1]
        ]

    def test_dinuc_content(seq_len=1001, num_shufs=5, seed=1234, one_hot=False):
        rng = np.random.RandomState(seed)
  
        orig = "".join(rng.choice(["A", "C", "T", "G"], seq_len))
        if one_hot: 
            orig_one_hot = dna_to_one_hot(orig)
            shufs = [
                one_hot_to_dna(one_hot) for one_hot in
                dinuc_shuffle(orig_one_hot, num_shufs, rng)
            ]
        else:
            shufs = dinuc_shuffle(orig, num_shufs, rng)
 
        # Get percent match matrix
        matches = np.zeros((num_shufs + 1, num_shufs + 1))
        char_arrays = [string_to_char_array(s) for s in [orig] + shufs]

        for i in range(num_shufs + 1):
            for j in range(i + 1, num_shufs + 1):
                matches[i, j] = np.sum(char_arrays[i] == char_arrays[j])
        matches = matches / seq_len * 100
     
        names = ["Orig"] + ["Shuf%d" % i for i in range(1, num_shufs + 1)]
        print("% nucleotide matches")
        print("\t" + "\t".join(names))
        for i in range(num_shufs + 1):
            print(names[i], end="\t")
            if i:
                print("\t".join(["-"] * i), end="\t")
            print("0", end="\t")
            print("\t".join(["%.3f" % x for x in matches[i, i + 1:]]))

        # Get nucleotide contents
        nuc_content = lambda s: \
            dict(zip(*np.unique(list(s), return_counts=True)))
        orig_nuc_cont = nuc_content(orig)
        shuf_nuc_conts = [nuc_content(shuf) for shuf in shufs]

        print("\nNucleotide counts")
        print("Nuc\t" + "\t".join(names))
        format_str = "%s\t" + "\t".join(["%d"] * len(names))
        for nuc in sorted(orig_nuc_cont.keys()):
            contents = [nuc, orig_nuc_cont[nuc]] + \
                [shuf_dict[nuc] for shuf_dict in shuf_nuc_conts]
            print(format_str % tuple(contents))
        
        # Get dinucleotide contents
        orig_dinuc_cont = dinuc_content(orig)
        shuf_dinuc_conts = [dinuc_content(shuf) for shuf in shufs]

        print("\nDinucleotide counts")
        print("Dinuc\t" + "\t".join(names))
        format_str = "%s\t" + "\t".join(["%d"] * len(names))
        for dinuc in sorted(orig_dinuc_cont.keys()):
            contents = [dinuc, orig_dinuc_cont[dinuc]] + \
                [shuf_dict[dinuc] for shuf_dict in shuf_dinuc_conts]
            print(format_str % tuple(contents))

    print("Testing correctness of dinucleotide shuffling")
    test_dinuc_content(one_hot=True, seed=None)

    print("\nShuffling 500 sequences of length 1000, 10 shuffles each...")
    results = bench(one_hot=True, vectorize=True)
