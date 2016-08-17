import deeplift.util
from deeplift.util import NEAR_ZERO_THRESHOLD

PoolMode = deeplift.util.enum(max='max', avg='avg')
BorderMode = deeplift.util.enum(same='same', half='half', valid='valid')
