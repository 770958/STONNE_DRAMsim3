from PIL import Image
from torchvision import transforms
import torch
import resnet

def run_model(simulation_file=''):
    res_model = resnet.resnet_model(simulation_file)
