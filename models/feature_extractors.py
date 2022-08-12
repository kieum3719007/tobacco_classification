import torch
from transformers import AutoModel
from enum import Enum


class BackboneName(Enum):
    DiT = 1
    Dense121 = 2
    VGG19 = 3


def get_feature_extractor(model_name: BackboneName = BackboneName.DiT):
    """
    Gets the backbone model for a given model name.
    Possible values are:\n
        1. microsoft/dit-base: The DiT pretrained model.\n
        2. densenet121: DenseNet121 is pretrained in Imagenet.\n
        3. vgg19: VGG19 is pretrained in Imagenet.\n
    """

    if model_name == BackboneName.DiT:
        return DiTFeatureExtractor()
    elif (model_name == BackboneName.Dense121):
        return DenseNet121FeatureExtractor()

    elif (model_name == BackboneName.VGG19):
        return VGG19FeatureExtractor()
    else:
        raise NotImplementedError(
            f"Getting backbone {model_name} is not implemented.")


class FeatureExtractor(torch.nn.Module):
    def __init__(self,
                 backbone):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def num_features(self):
        raise NotImplementedError()

    def forward(self, input):
        raise NotImplementedError()


class DiTFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super(DiTFeatureExtractor, self).__init__(
            AutoModel.from_pretrained("microsoft/dit-base"))

    @property
    def name(self):
        return "DiT model"

    @property
    def num_features(self):
        return self.backbone.config.hidden_size

    def forward(self, input):
        return self.backbone(input).pooler_output


class DenseNet121FeatureExtractor(FeatureExtractor):
    def __init__(self):
        super(DenseNet121FeatureExtractor, self).__init__(torch.hub.load(
            'pytorch/vision:v0.10.0', 'densenet121', pretrained=True))

    @property
    def name(self):
        return "DenseNet121"

    @property
    def num_features(self):
        return 1000

    def forward(self, input):
        return self.backbone(input)


class VGG19FeatureExtractor(FeatureExtractor):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__(torch.hub.load(
            'pytorch/vision:v0.10.0', 'vgg19', pretrained=True))

    @property
    def name(self):
        return "VGG19"

    @property
    def num_features(self):
        return 1000

    def forward(self, input):
        return self.backbone(input)
