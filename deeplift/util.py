from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import os
import os.path
import numpy as np
from collections import namedtuple
from collections import OrderedDict
import json
import deeplift

NEAR_ZERO_THRESHOLD = 10**(-7)


def compile_func(inputs, outputs):
    import deeplift.backend as B
    return B.function(inputs, outputs)


def enum(**enums):
    class Enum(object):
        pass
    to_return = Enum;
    for key,val in enums.items():
        if hasattr(val, '__call__'): 
            setattr(to_return, key, staticmethod(val))
        else:
            setattr(to_return,key,val);
    to_return.vals = [x for x in enums.values()];
    to_return.the_dict = enums
    return to_return;


def assert_is_type(instance, the_class, instance_var_name):
    return assert_type(instance, the_class, instance_var_name, True)


def assert_is_not_type(instance, the_class, instance_var_name):
    return assert_type(instance, the_class, instance_var_name, False)


def assert_type(instance, the_class, instance_var_name, is_type_result):
    assert (is_type(instance, the_class) == is_type_result),\
            instance_var_name+" should be an instance of "\
            +the_class.__name__+" but is "+str(instance.__class__) 
    return True


def is_type(instance, the_class):
    return superclass_in_base_classes(instance.__class__.__bases__, the_class)


def superclass_in_base_classes(base_classes, the_class):
    """
        recursively determine if the_class is among or is a superclass of
         one of the classes in base_classes. The comparison is done by
         name so that even if reload is called on a module, this still
         works.
    """
    for base_class in base_classes:
        if base_class.__name__ == the_class.__name__:
            return True 
        else:
            #if the base class in turn has base classes of its own
            if len(base_class.__bases__)!=1 or\
                base_class.__bases__[0].__name__ != 'object':
                #check them. If there's a hit, return True
                if (superclass_in_base_classes(
                    base_classes=base_class.__bases__,
                    the_class=the_class)):
                    return True
    #if 'True' was not returned in the code above, that means we don't
    #have a superclass
    return False


def run_function_in_batches(func,
                            input_data_list,
                            learning_phase=None,
                            batch_size=10,
                            progress_update=1000,
                            multimodal_output=False):
    #func has a return value such that the first index is the
    #batch. This function will run func in batches on the inputData
    #and will extend the result into one big list.
    #if multimodal_output=True, func has a return value such that first
    #index is the mode and second index is the batch
    assert isinstance(input_data_list, list), "input_data_list must be a list"
    #input_datas is an array of the different input_data modes.
    to_return = [];
    i = 0;
    while i < len(input_data_list[0]):
        if (progress_update is not None):
            if (i%progress_update == 0):
                print("Done",i)
        func_output = func(*([x[i:i+batch_size] for x in input_data_list]
                                +([] if learning_phase is
                                   None else [learning_phase])
                        ))
        if (multimodal_output):
            assert isinstance(func_output, list),\
             "multimodal_output=True yet function return value is not a list"
            if (len(to_return)==0):
                to_return = [[] for x in func_output]
            for to_extend, batch_results in zip(to_return, func_output):
                to_extend.extend(batch_results)
        else:
            to_return.extend(func_output)
        i += batch_size;
    return to_return


def mean_normalise_weights_for_sequence_convolution(weights,
                                                    bias,
                                                    axis_of_normalisation,
                                                    dim_ordering):
    print("Normalising weights for one-hot encoded sequence convolution")
    print("axis of normalisation is: "+str(axis_of_normalisation))
    print("Weight shape on that axis is: "
          +str(weights.shape[axis_of_normalisation]))
    mean_weights_at_positions=np.mean(weights,axis=axis_of_normalisation)
    if (dim_ordering=='th'):
        print("Theano dimension ordering; output channel axis is first one "
              "which has a length of "+str(weights.shape[0]))
        #sum across remaining dimensions except output channel which is first
        new_bias = bias + np.sum(np.sum(mean_weights_at_positions,
                                    axis=1),axis=1)
    elif (dim_ordering=='tf'):
        print("Tensorflow dimension ordering; output channel axis is last one "
              "which has a length of "+str(weights.shape[-1]))
        #sum across remaining dimensions except output channel which is last
        new_bias = bias + np.sum(np.sum(mean_weights_at_positions,
                                    axis=0),axis=0)
    else:
        raise RuntimeError("Unsupported dim ordering "+str(dim_ordering))
    mean_weights_at_positions = np.expand_dims(
                                 mean_weights_at_positions,
                                 axis_of_normalisation)
    renormalised_weights=weights-mean_weights_at_positions
    return renormalised_weights, new_bias


def mean_normalise_rnn_weights(weights, bias):
    assert len(weights.shape)==2
    assert weights.shape[0]==4
    mean_per_unit = np.mean(weights, axis=0) 
    new_weights = weights-mean_per_unit[None,:]
    new_bias = bias + mean_per_unit
    return new_weights, new_bias


def get_mean_normalised_softmax_weights(weights, biases):
    new_weights = weights - np.mean(weights, axis=1)[:,None]
    new_biases = biases - np.mean(biases)
    return new_weights, new_biases


def get_effective_width_and_stride(widths,strides):
    effectiveStride = strides[0] 
    effectiveWidth = widths[0]
    assert len(strides)==len(widths)
    if len(strides)>1:
        for (stride, width) in zip(strides[1:],widths[1:]):
            effectiveWidth = ((width-1)*effectiveStride)+effectiveWidth 
            effectiveStride = effectiveStride*stride
    return effectiveWidth, effectiveStride


def get_lengthwise_widths_and_strides(layers):
    """
        layers: a list of convolutional/pooling blobs
    """
    import deeplift.blobs
    widths = [] 
    strides = []
    for layer in layers:
        if type(layer).__name__ == "Conv2D":
            strides.append(layer.strides[1]) 
            widths.append(layer.W.shape[3])
        elif isinstance(layer, deeplift.blobs.Pool2D):
            strides.append(layer.strides[1]) 
            widths.append(layer.pool_size[1])
        elif isinstance(layer, deeplift.blobs.Activation):
            pass
        elif isinstance(layer, deeplift.blobs.NoOp):
            pass
        else:
            raise RuntimeError("Please implement how to extract width and"
                               "stride from layer of type: "
                               +type(layer).__name__)
    return widths, strides


def get_lengthwise_effective_width_and_stride(layers):
    widths, strides = get_lengthwise_widths_and_strides(layers)
    return get_effective_width_and_stride(widths, strides)


def load_yaml_data_from_file(file_name):
    file_handle = get_file_handle(file_name)
    data = yaml.load(file_handle) 
    file_handle.close()
    return data


def get_file_handle(file_name, mode='r'):
    use_gzip_open = False
    #if want to read from file, check that is gzipped and set
    #use_gzip_open to True if it is 
    if (mode=="r" or mode=="rb"):
        if (is_gzipped(file_name)):
            mode="rb"
            use_gzip_open = True
    #Also check if gz or gzip is in the name, and use gzip open
    #if writing to the file.
    if (re.search('.gz$',filename) or re.search('.gzip',filename)):
        #check for the case where the file name implies the file
        #is gzipped, but the file is not actually detected as gzipped,
        #and warn the user accordingly
        if (mode=="r" or mode=="rb"):
            if (use_gzip_open==False):
                print("Warning: file has gz or gzip in name, but was not"
                      " detected as gzipped")
        if (mode=="w"):
            use_gzip_open = True
            #I think write will actually append if the file already
            #exists...so you want to remove it if it exists
            if os.path.isfile(file_name):
                os.remove(file_name)
    if (use_gzip_open):
        return gzip.open(file_name,mode)
    else:
        return open(file_name,mode) 


def is_gzipped(file_name):
    file_handle = open(file_name, 'rb')
    magic_number = file_handle.read(2)
    file_handle.close()
    is_gzipped = (magic_number == b'\x1f\x8b' )
    return is_gzipped


def apply_softmax_normalization_if_needed(layer, previous_layer):
    if (type(layer)==deeplift.blobs.Softmax):
        #mean normalise the inputs to the softmax
        previous_layer.W, previous_layer.b =\
         deeplift.util.get_mean_normalised_softmax_weights(
            previous_layer.W, previous_layer.b)


def connect_list_of_layers(deeplift_layers):
    if (len(deeplift_layers) > 1):
        #string the layers together so that subsequent layers take the previous
        #layer as input
        last_layer_processed = deeplift_layers[0] 
        for layer in deeplift_layers[1:]:
            #apply_softmax_normalization_if_needed(layer, last_layer_processed)
            layer.set_inputs(last_layer_processed)
            last_layer_processed = layer


def format_json_dump(json_data, indent=2):
    return json.dumps(jsonData, indent=indent, separators=(',', ': ')) 


def get_cross_corr_function(filters):
    from deeplift import backend as B

    if (len(filters.shape)==3):
        filters = filters[:,None,:,:]
    assert filters.shape[1]==1 #input channels=1
    assert filters.shape[2]==4 #acgt

    #set up the convolution. Note that convolutions reverse things
    filters = np.array(filters[:,:,::-1,::-1]).astype("float32") 
    input_var = B.tensor_with_dims(num_dims=4, name="input")
    conv_out = B.conv2d(inp=input_var,
                        filters=filters,
                        border_mode="valid",
                        subsample=(1,1))
    compiled_func = B.function(inputs=[input_var], outputs=conv_out)

    def cross_corr(regions_to_scan, batch_size, progress_update=None):
        assert len(regions_to_scan.shape)==4
        assert regions_to_scan.shape[1]==1 #input channels=1
        assert regions_to_scan.shape[2]==4 #acgt
        #run function in batches
        conv_results = np.array(deeplift.util.run_function_in_batches(
                                func=compiled_func,
                                input_data_list=[regions_to_scan],
                                batch_size=batch_size,
                                progress_update=progress_update))
        return conv_results
    return cross_corr


def get_smoothen_function(window_size, same_size_return=True):
    """
        Returns a function for smoothening inputs with a window
         of size window_size.

        Returned function has arguments of inp,
         batch_size and progress_update
    """
    from deeplift import backend as B
    inp_tensor = B.tensor_with_dims(2, "inp_tensor") 

    if (same_size_return):
        #do padding so that the output will have the same size as the input
        #remember, the output will have length of input length - (window_size-1)
        #so, we're going to pad with int(window_size/2), and for even window_size
        #we will trim off the value from the front of the output later on
        padding = int(window_size/2)  
        new_dims = [inp_tensor.shape[0], inp_tensor.shape[1]+2*padding]
        padded_inp = B.zeros(new_dims)
        #fill the middle region with the original input
        padded_inp = B.set_subtensor(
                        padded_inp[:,padding:(inp_tensor.shape[1]+padding)],
                        inp_tensor) 
        #duplicate the left end for padding
        padded_inp = B.set_subtensor(padded_inp[:,0:padding],
                                     inp_tensor[:,0:padding])
        #duplicate the right end for padding
        padded_inp = B.set_subtensor(
                        padded_inp[:,(inp_tensor.shape[1]+padding):],
                        inp_tensor[:,(inp_tensor.shape[1]-padding):])
    else:
        padded_inp = inp_tensor
    padded_inp = padded_inp[:,None,None,:]

    averaged_padded_inp = B.pool2d(
                            inp=padded_inp,
                            pool_size=(1,window_size),
                            strides=(1,1),
                            border_mode="valid",
                            ignore_border=True,
                            pool_mode=B.PoolMode.avg) 

    #if window_size is even, then we have an extra value in the output,
    #so kick off the value from the front
    if (window_size%2==0 and same_size_return):
        averaged_padded_inp = averaged_padded_inp[:,:,:,1:]

    averaged_padded_inp = averaged_padded_inp[:,0,0,:]
    smoothen_func = B.function([inp_tensor], averaged_padded_inp)

    def smoothen(inp, batch_size, progress_update=None):
       return run_function_in_batches(
                func=smoothen_func,
                input_data_list=[inp],
                batch_size=batch_size,
                progress_update=progress_update)

    return smoothen


def get_top_n_scores_per_region(
    scores, n, exclude_hits_within_window):
    scores = scores.copy()
    assert len(scores.shape)==2, scores.shape
    if (n==1):
        return np.max(scores, axis=1)[:,None]
    else:
        top_n_scores = []
        top_n_indices = []
        for i in range(scores.shape[0]):
            top_n_scores_for_region=[]
            top_n_indices_for_region=[]
            for j in range(n):
                max_idx = np.argmax(scores[i]) 
                top_n_scores_for_region.append(scores[i][max_idx])
                top_n_indices_for_region.append(max_idx)
                scores[i][max_idx-exclude_hits_within_window:
                          max_idx+exclude_hits_within_window-1] = -np.inf
            top_n_scores.append(top_n_scores_for_region) 
            top_n_indices.append(top_n_indices_for_region)
        return np.array(top_n_scores), np.array(top_n_indices)


def get_integrated_gradients_function(gradient_computation_function, 
                                      num_intervals, right_rectangle=False):
    def compute_integrated_gradients(
        task_idx, input_data_list, input_references_list,
        batch_size, progress_update=None):
        assert len(input_data_list)==1,\
         "only one mode of input supported atm; "+str(len(input_data_list))
        #cast to float32 because sometimes inputs are uint8 and then
        #when you subtract you can never get negative values!!
        input_data_list[0] = input_data_list[0].astype("float32")
        input_references_list[0] = input_references_list[0].astype("float32")
        outputs = []
        #remember, input_data_list and input_references_list are
        #a list with one entry per mode
        input_references_list =\
        [np.ones_like(np.array(input_data)) *input_reference for
         input_data, input_reference in
         zip(input_data_list, input_references_list)]
        #will flesh out multimodal case later...
        assert len(input_data_list)==1
        assert len(input_references_list)==1
        
        vectors = []
        interpolated_inputs = []
        interpolated_inputs_references = []
        for an_input, a_reference in zip(input_data_list[0],
                                         input_references_list[0]):
            #interpolate between reference and input with num_intervals
            vector = an_input - a_reference
            vectors.append(vector)
            step = vector/float(num_intervals)
            #prepare the array that has the inputs at different steps
            for i in range(num_intervals):
                interpolated_inputs.append(
                    a_reference + step*(i+(1.0 if right_rectangle else 0.5)))
                interpolated_inputs_references.append(a_reference)        
        #find the gradients at different steps for all the inputs
        interpolated_gradients =\
         np.array(gradient_computation_function(
            task_idx=task_idx,
            input_data_list=[interpolated_inputs],
            input_references_list=[interpolated_inputs_references],
            batch_size=batch_size,
            progress_update=progress_update))
        #reshape for taking the mean over all the steps
        #the first dim is the sample idx, second dim is the step
        #I've checked this is the appropriate axis ordering for the reshape
        interpolated_gradients = np.reshape(
                                interpolated_gradients,
                                [input_data_list[0].shape[0], num_intervals]
                                +list(input_data_list[0].shape[1:])) 
        #take the mean gradient over all the steps, multiply by vector
        #equivalent to the stepwise integral
        mean_gradient = np.mean(interpolated_gradients,axis=1)
        contribs = mean_gradient*np.array(vectors)
        return contribs
    return compute_integrated_gradients


def get_shuffle_seq_ref_function(score_computation_function, 
                                 shuffle_func, one_hot_func):
    def compute_scores_with_shuffle_seq_refs(
        task_idx, input_data_sequences, num_refs_per_seq,
        batch_size, seed=1, progress_update=None):

        import numpy as np
        np.random.seed(seed)
        import random
        random.seed(seed)

        to_run_input_data_seqs = []
        to_run_input_data_refs = []
        references_generated = 0
        for seq in input_data_sequences:
            for i in range(num_refs_per_seq):
                references_generated += 1
                if (progress_update is not None and
                    references_generated%progress_update==0):
                    print(str(references_generated)
                          +" reference seqs generated")
                to_run_input_data_seqs.append(seq) 
                to_run_input_data_refs.append(shuffle_func(seq)) 
        if (progress_update is not None):
            print("One hot encoding sequences...")
        input_data_list = [one_hot_func(to_run_input_data_seqs)] 
        input_references_list = [one_hot_func(to_run_input_data_refs)]
        if (progress_update is not None):
            print("One hot encoding done...")

        computed_scores = np.array(score_computation_function(
            task_idx=task_idx,
            input_data_list=input_data_list,
            input_references_list=input_references_list,
            batch_size=batch_size,
            progress_update=progress_update))
        
        computed_scores = np.reshape(
                            computed_scores,
                            [len(input_data_sequences),
                             num_refs_per_seq]
                             +list(input_data_list[0].shape[1:])) 
        #take the mean over all the refs
        mean_scores = np.mean(computed_scores,axis=1)
        return mean_scores
    return compute_scores_with_shuffle_seq_refs


def randomly_shuffle_seq(seq):
    return "".join(in_place_shuffle([x for x in seq]))


def in_place_shuffle(arr):
    import random
    len_of_arr = len(arr)
    for i in xrange(0,len_of_arr):
        #randomly select index:
        chosen_index = random.randint(i,len_of_arr-1)
        #swap
        val_at_index = arr[chosen_index]
        arr[chosen_index] = arr[i]
        arr[i] = val_at_index
    return arr


