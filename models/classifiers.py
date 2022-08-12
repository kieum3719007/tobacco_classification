import torch
from torch.nn import Module
from torch import Tensor
from .feature_extractors import (BackboneName, get_feature_extractor,
                                 DiTFeatureExtractor, DenseNet121FeatureExtractor, VGG19FeatureExtractor)

LABEL_LOCATION = "models/labels"


def get_labels():
    with open(LABEL_LOCATION, 'r') as f:
        return f.read().splitlines()


class ImageClassifier(Module):
    def __init__(self,
                 backbone,
                 num_classes,
                 traning_backbone=True
                 ):
        super(ImageClassifier, self).__init__()
        self.num_classes = num_classes
        self.traning_backbone = traning_backbone
        self.feature_extraction = backbone
        self.hidden = torch.nn.Linear(
            in_features=self.feature_extraction.num_features,
            out_features=1024
        )
        self.classifier = torch.nn.Linear(
            in_features=1024,
            out_features=self.num_classes
        )

    @property
    def name(self):
        return self.feature_extraction.name

    def forward(self, input: Tensor) -> Tensor:
        features = self.feature_extraction(input)
        if (not self.traning_backbone):
            features.detach()
        features = self.hidden(features)
        return self.classifier(features)


def get_classifier(name: BackboneName, num_classes: int) -> ImageClassifier:
    return ImageClassifier(get_feature_extractor(name), num_classes)


def load(image_classifier: ImageClassifier) -> ImageClassifier:
    if (isinstance(image_classifier.feature_extraction, DiTFeatureExtractor)):
        path = "dit.dict"
    elif (isinstance(image_classifier.feature_extraction, DenseNet121FeatureExtractor)):
        path = "densenet121.dict"
    elif (isinstance(image_classifier.feature_extraction, VGG19FeatureExtractor)):
        path = "vgg19.dict"

    state_dict = torch.load(path, map_location=torch.device('cpu'))
    image_classifier.load_state_dict(state_dict)
    return image_classifier
