# Download an example image from the pytorch website

from PIL import Image
from torchvision import transforms
import torch
import vgg


def run_model(simulation_file=''):
    input_image = Image.open("/mnt/users/wangjs/stonne/benchmarks/image_classification/vgg/dog.jpg")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    vgg_model = vgg.vgg16_model(simulation_file)
    with torch.no_grad():
        output = vgg_model(input_batch)

    # # Get the predicted class label
    # pred = output.argmax(dim=1)
    # print(pred)
