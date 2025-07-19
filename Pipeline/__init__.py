from .augment_generator import SEAL_CNN, FEL, seal_adapt, infer_single_image
from .Fine_tuning import SEAL_CNN_EWC, EWC, finetune_ewc
from .Inference import infer_single_image as infer_single_image_ewc

__version__ = "0.1.0"
