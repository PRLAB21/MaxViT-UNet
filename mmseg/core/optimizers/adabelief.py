from mmcv.runner import OPTIMIZERS
from adabelief_pytorch import AdaBelief

OPTIMIZERS.register_module(name='AdaBelief', force=True)(AdaBelief)
