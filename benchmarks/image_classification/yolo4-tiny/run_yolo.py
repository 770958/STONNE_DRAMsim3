# Download an example image from the pytorch website

from PIL import Image
from torchvision import transforms
import torch
import yolo


def run_model(simulation_file=''):
    yolonet_model = yolo.yolo_model(simulation_file)