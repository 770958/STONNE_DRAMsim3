from PIL import Image
from torchvision import transforms
import torch
import lenet


def run_model(simulation_file=''):
    le_model = lenet.lenet_model(simulation_file)
