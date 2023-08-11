# Download an example image from the pytorch website

from PIL import Image
from torchvision import transforms
import torch
import mobilenets_stonne


def run_model(simulation_file=''):
    mobile_model = mobilenets_stonne.mobilenets_model(simulation_file)