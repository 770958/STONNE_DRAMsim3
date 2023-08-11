# Download an example image from the pytorch website

from PIL import Image
from torchvision import transforms
import torch
import shufflenet


def run_model(simulation_file=''):
    shuffle_model = shufflenet.shufflenet_model(simulation_file)