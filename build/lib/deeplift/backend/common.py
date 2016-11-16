import deeplift.util
from deeplift.util import NEAR_ZERO_THRESHOLD

PoolMode = deeplift.util.enum(max='max', avg='avg')
BorderMode = deeplift.util.enum(same='same', half='half', valid='valid')

from . import theano_backend as B

def apply_iteratively_to_list(the_list, function_to_apply):
    to_return = the_list[0]
    for item in the_list[1:]:
        to_return = function_to_apply(to_return, item)
    return to_return

def maximum_over_list(the_list):
    return apply_iteratively_to_list(the_list=the_list,
                                     function_to_apply=B.maximum)

def mask_if_not_condition(tensor, mask_val, condition):
    #condition should return a matrix of ones and zeros when applied to tensor
    return (B.switch(condition, tensor, mask_val))
