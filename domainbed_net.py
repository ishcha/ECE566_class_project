import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from domainbed import networks

class domainbedNet(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(domainbedNet, self).__init__()
        self.hparams = hparams
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, classifier)

    def forward(self, x):
        x = self.network(x)
        return x
