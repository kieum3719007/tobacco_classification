from enum import Enum
import torch
from torchvision.transforms import Normalize, Resize, Compose, ConvertImageDtype, ToTensor


class Transformation(Enum):
    ConvertType = 0
    Nomarlize = 1
    Resize = 2
    ToTensor = 3


TRANSFORMATIONS = [
    {
        "type": Transformation.ToTensor,
    },
    {
        "type": Transformation.ConvertType,
        "dtype": torch.float32
    },
    {
        "type": Transformation.Resize,
        "size": (224, 224)
    },
    {
        "type": Transformation.Nomarlize,
        "std": [0.485, 0.456, 0.406],
        "mean": [0.229, 0.224, 0.225]
    }
]


def get_transformation(type: Transformation, **args):
    if (type == Transformation.Nomarlize):
        return Normalize(args["mean"], args["std"])
    elif (type == Transformation.Resize):
        return Resize(args["size"])
    elif (type == Transformation.ConvertType):
        return ConvertImageDtype(args["dtype"])
    elif (type == Transformation.ToTensor):
        return ToTensor()
    else:
        raise NotImplementedError


def get_transformations(configs):
    transformations = [get_transformation(**config) for config in configs]
    return Compose(transformations)
