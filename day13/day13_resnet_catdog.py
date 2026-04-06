import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation

data_transforms = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(100)  # pause a bit so that plots are updated


if __name__ == '__main__':
    data_dir = "hymenoptera_data"

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                   shuffle=True, num_workers=4)
                       for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    print("class name is :", class_names[0], class_names[1])
    # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("device is: ", device)

    # Get a batch of training data
    inputs, classes = next(iter(data_loaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    #imshow(out, title=[class_names[x] for x in classes])


# till now we showed the data loading and visualization part. Now we will train a ResNet-18 model on this dataset. We will use a pre-trained model and fine-tune it on our dataset.
    def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
        since = time.time()
        #create a temporary folder.
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir,"best_model_params.pt")
            torch.save(model.state_dict(),best_model_params_path)
            best_acc =0.0
            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)
                for phase in ['train','val']:
                    if phase == 'train':
                        model.train() # set model to training mode.
                    else:
                        model.eval() # set model to evaluation mode.

                    running_loss =0.0
                    running_corrects =0.0
                    # now iterate over the data. 
                    for inputs, labels in data_loaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs) 
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            if phase =='train':
                                loss.backward() 
                                optimizer.step()
                        
                        running_loss+=loss.item()
                        running_corrects+=torch.sum(preds==labels.data)
                    
                    if phase=='train':
                        scheduler.step()
                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)

                    print()
            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

            model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
        return model
          
    # above we trained the model. Now make some output visualization with label and title.
    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0

        with torch.no_grad():
            for inputs, labels in data_loaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {class_names[preds[j]]}')
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs,2)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=5)

    visualize_model(model_ft)


