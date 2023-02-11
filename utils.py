'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

# DataLoading, classwise-accuracies, Plots of training & testing images, Plots of training & test curves, Plots of Mis-classified Images, Plots of GRADCAM images.
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.optim
import torchvision.transforms as transforms

import torch.optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Normalize,transforms

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import torchvision.models as models
from torchvision.utils import make_grid, save_image


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class Draw:
    def plotings(image_set):
        images = image_set
        img = images
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
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
            A.PadIfNeeded (min_height=4, min_width=4,always_apply=False, p=0.32),
            A.RandomCrop (32,32, always_apply=False, p=0.63),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,
                            fill_value=0.4734, p=round(random.uniform(0.35, 0.62), 2)),
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

storing_images = []
storing_predicted_labels = []
storing_target_labels = []
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

    def mis_classified(model,testloader,images_needed=None):
        if images_needed==None:
            images_needed = random.choice([10,20])
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
            dict(model_type='resnet', arch=resnet, layer_name='layer2')]

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

def plotting_gradCams(imagesneeded):
    print("Diagnosis is happening for Layer2 of ResNet18. Lets go!")
    figure2 = plt.figure(figsize=(16, 32))
    for i in range(imagesneeded):
        sub = figure2.add_subplot(imagesneeded, 1, i + 1)
        p = Plot_curves.plotting_gradcam(transforms.ToPILImage()(storing_images[i]))
        sub.imshow(p)
        sub.set_title(
            f"Predicted as: {classes[storing_predicted_labels[i]]} \n But, Actual is: {classes[storing_target_labels[i]]}")
    plt.tight_layout()
    plt.show()

def mis_prediction():
    return storing_images, storing_predicted_labels, storing_target_labels