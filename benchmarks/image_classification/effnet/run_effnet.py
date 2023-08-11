# Download an example image from the pytorch website

from PIL import Image
from torchvision import transforms
import torch
import effnet


def run_model():
    eff_model = effnet.effnet_model()