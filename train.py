import os
import time
import random
import copy
import numpy as np
import pandas as pd
from PIL import Image
import h5py
from sklearn.metrics import roc_auc_score
from preprocess import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchvision.models import *
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df_path, transform = None):
       self.df = pd.read_csv(df_path)
       self.transform = transform

    def __getitem__(self, index):
        path = self.df.image
    
        img = Image.open(path)
        label = self.df.label
        if label == "TUMOR":
          label = 1
        elif label == "NONTUMOR":
          label = 0
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)

augment = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(),
  transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform = transforms.Compose([
  transforms.Resize((imgSize,imgSize)),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class resnet(nn.Module):
  def __init__(self, num_classes):
    super(resnet, self).__init__()

    self.encoder = models.mobilenet_v2(pretrained=True)
    for param in self.encoder.parameters():
        param.requires_grad = True
    self.encoder.classifier = nn.Sequential(nn.Dropout(p=0.7), nn.Linear(1280, num_classes))
    self.encoder = self.encoder.cuda()
    

  def forward(self, x):
    x = self.encoder(x)
    return x

# https://github.com/rikiyay/HCCSurvNet/blob/master/risk_score_predictor.py
def train_model(model, loss, optimizer, scheduler, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    
    return model, best_acc, best_epoch

# https://github.com/rikiyay/HCCSurvNet/blob/master/risk_score_predictor.py
def test_model(model, loader, dataset_size, criterion):
    
    print('-' * 10)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    whole_probs = torch.FloatTensor(dataset_size)
    whole_labels = torch.LongTensor(dataset_size)
    
    with torch.no_grad():

        for i, data in enumerate(loader):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            outputs = F.softmax(outputs, dim=1)
            whole_probs[i*batchSize:i*batchSize+inputs.size(0)]=outputs.detach()[:,1].clone()
            whole_labels[i*batchSize:i*batchSize+inputs.size(0)]=labels.detach().clone()

        total_loss = running_loss / dataset_size
        total_acc = running_corrects.double() / dataset_size

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(total_loss, total_acc))

    return whole_probs.cpu().numpy(), whole_labels.cpu().numpy(), total_loss, total_acc

# https://github.com/rikiyay/HCCSurvNet/blob/master/risk_score_predictor.py
def bootstrap_auc(y_true, y_pred, n_bootstraps=2000, rng_seed=42):
    n_bootstraps = n_bootstraps
    rng_seed = rng_seed
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(len(y_pred), size=len(y_pred))
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    bootstrapped_scores = np.array(bootstrapped_scores)

    print("AUROC: {:0.3f}".format(roc_auc_score(y_true, y_pred)))
    print("Confidence interval for the AUROC score: [{:0.3f} - {:0.3}]".format(
        np.percentile(bootstrapped_scores, (2.5, 97.5))[0], np.percentile(bootstrapped_scores, (2.5, 97.5))[1]))
    
    return roc_auc_score(y_true, y_pred), np.percentile(bootstrapped_scores, (2.5, 97.5))
  
if __name__ == '__main__':

    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batchSize = 256
    lr=0.001
    beta1=0.5


    datasets = {}
    loaders = {}
    for phase in ['train', 'val', 'test']:
        if phase == 'train':
            data[phase] = Dataset('train.csv', transform = augment)
            loaders[phase] = torch.utils.data.DataLoader(data[phase], batch_size=batchSize, shuffle=True)
        elif phase == 'val':
            data[phase] = MyTissueData('val.csv', transform = transform)
            loaders[phase] = torch.utils.data.DataLoader(data[phase], batch_size=batchSize, shuffle=True)
        elif phase == 'test':
            data[phase] = MyTissueData('test.csv', transform = transform)
            loaders[dset_type] = torch.utils.data.DataLoader(data[dset_type], batch_size=batchSize, shuffle=False)
        print('Finished loading %s dataset: %s samples' % (dset_type, len(data[dset_type])))

    dataset_sizes = {phase: len(data[phase]) for phase in ['train', 'val', 'test']}

    model = resnet(2)
    model = model.to("cuda")
    criterion = nn.CrossEntropyLoss()


    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    if optim_method == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    elif optim_method == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr = lr)
    elif optim_method == "SGD": 
        optimizer = optim.SGD(model.parameters(), lr = lr)
    else: 
        raise ValueError('Optimizer not found. Accepted "Adam", "SGD" or "RMSprop"') 

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model, best_acc, best_epoch = train_model(model, criterion, optimizer, scheduler, num_epochs=25)
    torch.save(model.state_dict(), 'best_checkpoints_epoch_{0}_acc_{1}.pth'.format(str(best_epoch), str(best_acc.item())))

    prob_test, label_test, loss_test, acc_test = test_model(model, loaders['test'], dataset_sizes['test'], criterion)

    bootstrap_auc(label_test, prob_test)

    print("Training finished, weights saved!")
