'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

# DataLoading, classwise-accuracies, Plots of training & testing images, Plots of training & test curves, Plots of Mis-classified Images, Plots of GRADCAM images.
import os
import sys
import time
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt


import torch
import torchvision

import torch.nn as nn
import torch.optim
import torch.nn.init as init
import torchvision.transforms as transforms
# from model import Net

from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

import torch.optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Normalize

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from PIL import Image
import torchvision.models as models
from torchvision.utils import make_grid, save_image


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

class Params:
    def init_params(net):
        '''Init layer parameters.'''
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)
                if m.bias:
                    init.constant(m.bias, 0)


    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)

    TOTAL_BAR_LENGTH = 65.
    last_time = time.time()
    begin_time = last_time
    def progress_bar(current, total, msg=None):
        global last_time, begin_time
        if current == 0:
            begin_time = time.time()  # Reset for new bar.

        cur_len = int(TOTAL_BAR_LENGTH*current/total)
        rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

        sys.stdout.write(' [')
        for i in range(cur_len):
            sys.stdout.write('=')
        sys.stdout.write('>')
        for i in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write(']')

        cur_time = time.time()
        step_time = cur_time - last_time
        last_time = cur_time
        tot_time = cur_time - begin_time

        L = []
        L.append('  Step: %s' % format_time(step_time))
        L.append(' | Tot: %s' % format_time(tot_time))
        if msg:
            L.append(' | ' + msg)

        msg = ''.join(L)
        sys.stdout.write(msg)
        for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
            sys.stdout.write(' ')

        # Go back to the center of the bar.
        for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
            sys.stdout.write('\b')
        sys.stdout.write(' %d/%d ' % (current+1, total))

        if current < total-1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def format_time(seconds):
        days = int(seconds / 3600/24)
        seconds = seconds - days*3600*24
        hours = int(seconds / 3600)
        seconds = seconds - hours*3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes*60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds*1000)

        f = ''
        i = 1
        if days > 0:
            f += str(days) + 'D'
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + 'h'
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + 'm'
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's'
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + 'ms'
            i += 1
        if f == '':
            f = '0ms'
        return f



class Draw:
    def plotings(image_set):
        images = image_set
        img = images
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        print(len(image_set), "images are plotted")
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label


class args():
    def __init__(self, device='cpu', use_cuda=False) -> None:
        self.batch_size = 128
        self.device = device
        self.use_cuda = use_cuda
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}


class loader:
    def load_data():
        train_transforms = A.Compose([
            A.HorizontalFlip(p=0.49),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.45),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,
                            fill_value=0.4734, p=round(random.uniform(0.35, 0.52), 2)),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            A.pytorch.ToTensorV2()])

        trainset = Cifar10SearchDataset(root='./data', train=True,
                                        download=True, transform=train_transforms)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args().batch_size,
                                                  shuffle=True, **args().kwargs)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                 shuffle=False, **args().kwargs)
        return trainloader, testloader


class class_accuracy:
    def rate(testloader, model, classes):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                data, target = images.cuda(), labels.cuda()
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == target).squeeze()
                for i in range(4):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        dicts = {}
        for i in range(10):
            val = 100 * class_correct[i] / class_total[i]
            dicts[classes[i]] = val
        return dicts

#  Plots of training & test curves, Plots of Mis-classified Images, Plots of GRADCAM images.

class Plot_curves:
    def __init__(self):
        pass

    def performance_curves(self,train_acc,test_acc,train_losses,test_losses):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot([t.cpu().item() for t in train_losses])
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(test_acc)
        axs[1, 1].set_title("Test Accuracy")
        return axs

    def mis_classified(model,test_loader,images_needed=None):
        if images_needed==None:
            images_needed = random.choice([10,20])

        storing_images = []
        storing_predicted_labels = []
        storing_target_labels = []
        with torch.no_grad():
            model.eval()
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                for idx in range(len(pred)):
                    if pred[idx] != target[idx]:
                        storing_images.append(data[idx])
                        storing_predicted_labels.append(pred[idx])
                        storing_target_labels.append(target[idx])

        fig = plt.figure(figsize=(20, 14))

        if images_needed % 2 != 0:
            images_needed -= 1  # It becomes even so plotting would be good.

        num_rows = 0
        plots_per_row = 0
        if images_needed <= 10:
            num_rows = 2
            plots_per_row = images_needed // num_rows
        elif images_needed > 10:
            num_rows = 4
            plots_per_row = images_needed // num_rows

        for i in range(images_needed):
            sub = fig.add_subplot(num_rows, plots_per_row, i + 1)
            im2display = (np.squeeze(storing_images[i].permute(2, 1, 0)))
            sub.imshow(im2display.cpu().numpy())
            sub.set_title(f"Predicted as: {classes[storing_predicted_labels[i]]} \n But, Actual is: {classes[storing_target_labels[i]]}")
        plt.tight_layout()
        plt.show()

    def plotting_gradcam(pil_img):
        device = "cuda"
        resnet = models.resnet18(pretrained=True)
        torch_img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])(pil_img).to(device)

        normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
        configs = [
            dict(model_type='resnet', arch=resnet, layer_name='layer4')]

        for config in configs:
            config['arch'].to(device).eval()

        cams = [[cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
                for config in configs]

        images = []
        for gradcam, gradcam_pp in cams:
            mask, _ = gradcam(normed_torch_img)
            heatmap, result = visualize_cam(mask, torch_img)

            mask_pp, _ = gradcam_pp(normed_torch_img)
            heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

            images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])

        grid_image = make_grid(images, nrow=6)
        return transforms.ToPILImage()(grid_image)