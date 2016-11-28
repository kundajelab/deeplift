import tensorflow as tf
from deeplift.util import NEAR_ZERO_THRESHOLD

def distribute_over_product(def_act_var1, diff_def_act_var1,
                            def_act_var2, diff_def_act_var2, mult_output):
    mult_var1 = mult_output*(def_act_var2 + 0.5*diff_def_act_var2)
    mult_var2 = mult_output*(def_act_var1 + 0.5*diff_def_act_var1)
    return (mult_var1, mult_var2)


def pseudocount_near_zero(tensor):
    
    return tensor + (NEAR_ZERO_THRESHOLD*((B.abs(tensor)
                                          < 0.5*NEAR_ZERO_THRESHOLD)*
                                          (tensor >= 0)) -
                     NEAR_ZERO_THRESHOLD*((B.abs(tensor)
                                          < 0.5*NEAR_ZERO_THRESHOLD)*
                                          (tensor < 0)))


def set_col_to_val(var, col, val):
    var.assign(tf.zeros_like(var, dtype=tf.float32))
    vector_with_zeros = tf.Variable(tf.zeros(var.get_shape()[1]),
                                    dtype=tf.float32)
    vector_with_zeros = tf.scatter_update(vector_with_zeros,[col],[val])
    vector_with_zeros = tf.reshape(vector_with_zeros,
                                   [1,var.get_shape().as_list()[1]])
    broadcast_add = var+vector_with_zeros
    var = var.assign(broadcast_add)
    return var
