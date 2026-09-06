import os.path

import torch.optim as optim
import torch.nn as nn
from .resnet2 import PunctaResnet
from .puncta_dataloader import PunctaDatasetTest128
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PunctaClassifier:
    def __init__(self, model_path, patch_size=128):
        self.net = PunctaResnet().to(device)
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net.eval()
        self.patch_size = patch_size

    def pred(self, blobs_log_3, img, draw_image):
        testdataset = PunctaDatasetTest128(blobs_log_3, img, draw_image, self.patch_size)
        testloader = DataLoader(testdataset, batch_size=32,
                                shuffle=False)
        predictions = []
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, _, _ = data
                inputs = inputs.to(device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu())

        return predictions
