import tensorflow as tf
from deeplift.util import NEAR_ZERO_THRESHOLD

def gt_mask(inp, val):
    return tf.cast(tf.greater(inp, val), tf.float32)

def lt_mask(inp, val):
    return tf.cast(tf.less(inp, val), tf.float32)

def lte_mask(inp, val):
    return tf.cast(tf.less_equal(inp, val), tf.float32)

def gte_mask(inp, val):
    return tf.cast(tf.greater_equal(inp, val), tf.float32)

def eq_mask(inp, val):
    return tf.cast(tf.equal(inp, val), tf.float32)

def conv1d_transpose_via_conv2d(
    value, kernel, tensor_with_output_shape, stride, padding):
    return tf.squeeze(tf.nn.conv2d_transpose(
        value=tf.expand_dims(value,1),
        filter=kernel[None,:,:,:],
        #Note: tf.shape(var) doesn't give the same result
        #as var.get_shape(); one works, the other doesn't...
        output_shape=tf.shape(tf.expand_dims(tensor_with_output_shape,1)),
        strides=(1,1,stride,1),
        padding=padding),1)


def distribute_over_product(def_act_var1, diff_def_act_var1,
                            def_act_var2, diff_def_act_var2, mult_output):
    mult_var1 = mult_output*(def_act_var2 + 0.5*diff_def_act_var2)
    mult_var2 = mult_output*(def_act_var1 + 0.5*diff_def_act_var1)
    return (mult_var1, mult_var2)


def pseudocount_near_zero(tensor):
    
    return tensor + (NEAR_ZERO_THRESHOLD*(lt_mask(tf.abs(tensor),
                                                  0.5*NEAR_ZERO_THRESHOLD)*
                                          gte_mask(tensor,0)) -
                     NEAR_ZERO_THRESHOLD*(lt_mask(tf.abs(tensor),
                                                  0.5*NEAR_ZERO_THRESHOLD)*
                                          lt_mask(tensor,0)))


def add_val_to_col(var, col, val):
    vector_with_zeros = tf.Variable(tf.zeros(var.get_shape()[1]),
                                    dtype=tf.float32)
    vector_with_zeros = tf.scatter_update(vector_with_zeros,[col],[val])
    vector_with_zeros = tf.reshape(vector_with_zeros,
                                   [1,var.get_shape().as_list()[1]])
    return var+vector_with_zeros
