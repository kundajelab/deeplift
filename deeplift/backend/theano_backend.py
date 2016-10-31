import theano
from theano import tensor as T
import deeplift.util
from deeplift.util import NEAR_ZERO_THRESHOLD
from .common import *


def dimshuffle(tensor, new_shape):
    return tensor.dimshuffle(new_shape)


def reshape(tensor, shape):
    return T.reshape(tensor, shape)


def pow(inp, power):
    return T.pow(inp, power)


def exp(inp):
    return T.exp(inp)


def switch(tensor, iftrue, iffalse):
    return T.switch(tensor, iftrue, iffalse)


def maximum(x, y):
    return T.maximum(x, y)


def minimum(x, y):
    return T.minimum(x, y)


def as_tensor_variable(x, name, ndim):
   return T.as_tensor_variable(x, name=name, ndim=ndim) 


def max(x, axis):
    return T.max(x, axis=axis)


def min(x, axis):
    return T.min(x, axis=axis)


def argmax(x, axis):
    return T.argmax(x, axis=axis)


def square(x):
    return T.sqr(x)


def sqrt(x):
    return T.sqrt(x) 


def sum(x, axis):
    return T.sum(x, axis=axis)


def ones_like(x):
    return T.ones_like(x, dtype=theano.config.floatX)


def zeros_like(x):
    return T.zeros_like(x)


def zeros(shape):
    return T.zeros(shape=shape)


def set_subtensor(subtensor, amnt):
    return T.inc_subtensor(subtensor, amnt, set_instead_of_inc=True)


def function(inputs, outputs, **kwargs):
    return theano.function(inputs, outputs,
                            allow_input_downcast=True,
                            on_unused_input='ignore',
                            **kwargs) 


def tensor_with_dims(num_dims, name):
    return T.TensorType(dtype=theano.config.floatX,
                        broadcastable=[False]*num_dims)(name)


def shared(value):
    return theano.shared(value=value)


def dot(x, y):
    return T.dot(x, y)


def relu(inp):
    return T.nnet.relu(inp)


def sigmoid(inp):
    return T.nnet.sigmoid(inp)


def hard_sigmoid(inp):
    return T.nnet.hard_sigmoid(inp)


def tanh(inp):
    return T.tanh(inp)


def softmax(inp):
    return T.nnet.softmax(inp)


def sigmoid_grad(inp):
    out = sigmoid(inp)
    grad = T.nnet.sigmoid.grad((inp,), (out,))
    return grad


def softmax_grad(inp):
    out = softmax(inp)
    grad = T.nnet.Softmax().grad((inp,), (out,))
    return grad


def abs(inp):
   return T.abs_(inp) 


def conv2d(inp, filters, border_mode, subsample):
    inp = T.cast(inp, dtype=theano.config.floatX) 
    if (border_mode==BorderMode.same):
        #'half' kernel width padding results in outputs of the same
        #dimensions as input
        border_mode=BorderMode.half
        assert filters.shape[2]%2 == 1 and filter_shape[3]%2 == 1,\
            "haven't handled even filter shapes for border mode 'half'"
    return T.nnet.conv2d(input=inp,
                         filters=T.cast(theano.shared(value=filters),
                                        dtype=theano.config.floatX),
                         border_mode=border_mode,
                         subsample=subsample,
                         filter_shape=filters.shape)


def conv2d_grad(out_grad, conv_in, filters, border_mode, subsample):
    out_grad=T.cast(out_grad, dtype=theano.config.floatX)
    conv_op = T.nnet.conv.ConvOp(output_mode=border_mode,
                                 dx=subsample[0],
                                 dy=subsample[1]) 
    inverse_conv2d = conv_op.grad(
                       (conv_in,
                       T.cast(T.as_tensor_variable(filters),
                         dtype=theano.config.floatX)),
                       (out_grad,))
    #grad returns d_input and d_filters; we just care about
    #the first
    return inverse_conv2d[0]


def get_pooling_padding_and_theano_pool_mode(
    pool_size, border_mode, pool_mode):
    if border_mode == BorderMode.same:
        padding = [x - (2 if x%2==1 else 1) for x in pool_size] 
    elif border_mode == BorderMode.valid:
        padding = (0, 0) 
    else:
        raise RuntimeError("Valid border modes are: "+str(BorderMode.vals)
                           +", got: "+str(border_mode))

    if (pool_mode == PoolMode.max):
        theano_pool_mode = 'max'
    elif (pool_mode == PoolMode.avg):
        theano_pool_mode = 'average_exc_pad'
    else:
        raise RuntimeError("Valid pool modes are: "+str(PoolMode.vals)
                           +", got: "+str(pool_mode))
    return padding, theano_pool_mode

def pool2d(inp, pool_size, strides, border_mode, ignore_border, pool_mode):

    padding, theano_pool_mode = get_pooling_padding_and_theano_pool_mode(
                                    pool_size, border_mode, pool_mode)

    to_return = T.signal.pool.pool_2d(input=inp,
                    ds=pool_size,
                    ignore_border=ignore_border,
                    st=strides,
                    padding=padding,
                    mode=theano_pool_mode)

    if border_mode==BorderMode.same:
        final_shape = [(inp.shape[2+i] + strides[i] - 1)//strides[i]
                       for i in [0,1]]
        to_return = to_return[:, :, final_shape[0], final_shape[1]]
    
    return to_return


def pool2d_grad(out_grad, pool_in,
                   pool_size, strides, border_mode,
                   ignore_border, pool_mode):
    if (border_mode == BorderMode.same):
        raise RuntimeError("I have not handled the cropping for border mode"
                           "'same' yet")
    padding, theano_pool_mode = get_pooling_padding_and_theano_pool_mode(
                                    pool_size, border_mode, pool_mode)
    pool_op = T.signal.pool.Pool(ds=pool_size,
                                 st=strides,
                                 ignore_border=ignore_border,
                                 padding=padding,
                                 mode=theano_pool_mode)
    return pool_op.grad((pool_in,),
                        (out_grad,))[0]
    

def flatten_keeping_first(x):
    """
        Flatten all but the first dimension
    """
    return T.reshape(x, (x.shape[0], T.prod(x.shape[1:]))) 


def unflatten_keeping_first(x, like):
    """
        shape x to resemble the shape of 'like'
    """
    return T.reshape(x, like.shape)


def zeropad2d(x, padding):
    output = T.zeros((x.shape[0], #batch
                      x.shape[1], #channel
                      x.shape[2] + 2*padding[0], #rows
                      x.shape[3] + 2*padding[1])) #cols
    output = set_subtensor(output[:, :, padding[0]:x.shape[2]+padding[0],
                                        padding[1]:x.shape[3]+padding[1]], x)
    return output 


def discard_pad2d(x, padding):
    return x[:, :, padding[0]:(x.shape[2]-padding[0]),
                   padding[1]:(x.shape[3]-padding[0])]


def for_loop(step_function, inputs, initial_hidden_states, go_backwards):
    """
        inputs: time axis must be first
    """ 
    results = theano.scan(
        step_function,
        sequences=inputs,
        outputs_info=initial_hidden_states,
        go_backwards=go_backwards)[0] #screw the updates
    #when results has length 1, it is not returned as a list. wrap it
    if (isinstance(results, list)==False):
        results = [results]
    #put the batch axis back in front
    results = [dimshuffle(tensor, [1,0]+[x for x in xrange(2, tensor.ndim)])
               for tensor in results]
    return results


def concat(tensor_list, axis):
    return T.concatenate(tensor_list=tensor_list, axis=axis)


def batch_normalization(inputs, gamma, beta, mean, std, epsilon):
    return T.nnet.batch_normalization(inputs=inputs, gamma=gamma,
                                      beta=beta, mean=mean,
                                      std=(std + epsilon),
                                      mode='high_mem')
