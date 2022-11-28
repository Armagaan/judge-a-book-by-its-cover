# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import time
from PIL import Image,ImageStat
import os,sys
from numpy import asarray
from torchvision.io import read_image

csv_path = sys.argv[1]
torch.manual_seed(7)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



train_csv = csv_path+"/train_x.csv"
train_label_csv = csv_path+"/train_y.csv"

data = pd.read_csv(train_csv)
data2 = pd.read_csv(train_label_csv)

im_names = data['Cover_image_name']
act_im_path = csv_path+"/images/images/"
#../input/col774-2022/images/images


train_x = act_im_path+im_names
train_y = data2['Genre']



train_data = pd.concat([train_x, train_y], axis=1)

class MyDataset(Dataset):
    def __init__(self,annotations_file,im_transform,target_transform=None):
        self.dataframe = annotations_file
        self.im_transform = im_transform
        self.target_transform = target_transform
        
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self,idx):
        image_path = self.dataframe.iloc[idx,0]
        image = Image.open(image_path)
        
        label = self.dataframe.iloc[idx,1]
        
        image = self.im_transform(image)
        
        label = torch.tensor(label,dtype=int)
#         if self.target_transform:
#             label = self.target_transform(label)
        
        return image,label

b_size = 256
classes  = 30
learning_rate = 0.0005
epochs = 25

def find_mean_std(data_x):  
    R_mean = 0
    G_mean = 0
    B_mean = 0

    R_std = 0
    G_std = 0
    B_std = 0

    total_samples = data_x.shape[0]
    for im in data_x:
        img = Image.open(im)
        transform = transforms.ToTensor()
        im_trns = transform(img)
        mean_r, mean_g, mean_b = torch.mean(im_trns, dim=[1,2])
        std_r,std_g,std_b = torch.std(im_trns,dim=[1,2])
        R_std += std_r
        G_std += std_g
        B_std += std_b
        R_mean += mean_r
        G_mean += mean_g
        B_mean += mean_b

    R_mean = R_mean/total_samples
    G_mean = G_mean/total_samples
    B_mean = B_mean/total_samples

    R_std /= total_samples
    G_std /= total_samples
    B_std /= total_samples

    print("Mean along red channel = ",R_mean)
    print("Mean along green channel = ",G_mean)
    print("Mean along blue channel = ",B_mean)

    print("Std dev along red channel = ",R_std)
    print("Std dev along green channel = ",G_std)
    print("Std dev along blue channel = ",B_std)
    
    return R_mean,G_mean,B_mean,R_std,G_std,B_std

R_mean,G_mean,B_mean,R_std,G_std,B_std = find_mean_std(train_x)
image_transformer = transforms.Compose([transforms.ToTensor(),transforms.Resize((64,64)),transforms.Normalize(mean=[R_mean, G_mean, B_mean],
                                                          std=[R_std,G_std,B_std])
                                     ])

trn_dataset = MyDataset(train_data,image_transformer)



train_loader = DataLoader(trn_dataset,batch_size=b_size,shuffle=True)

class CNN(nn.Module):
    def __init__(self,classes):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(5,5))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(5,5))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(5,5))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.fc1 = nn.Linear(2048,128)
        self.relu4 = nn.ReLU()
        
        self.fc2 = nn.Linear(128,classes)
        
        
    def forward(self,x):
        output = self.conv1(x)
        output = self.relu1(output)
        output = self.pool1(output)
        
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool2(output)
        
        output = self.conv3(output)
        output = self.relu3(output)
        output = self.pool3(output)
        
        output = output.reshape(output.size(0),-1)
        
        output = self.fc1(output)
        output = self.relu4(output)
        output = self.fc2(output)
        
        return output
        

cnnModel = CNN(classes).to(device)

loss_fun = nn.CrossEntropyLoss()

# Set optimizer with optimizer
#optimizer = torch.optim.SGD(cnnModel.parameters(), lr=learning_rate,momentum=0.9)  
optimizer = torch.optim.Adam(cnnModel.parameters(), lr = learning_rate)

total_step = len(train_loader)

st_time = time.time()
for itr in range(epochs):
    print("Iteration No: ",itr)
    #cnnModel.train()

    
    for image,label in train_loader:
        image = image.to(device)
        label = label.to(device)
       
        im_out = cnnModel(image)
        loss = loss_fun(im_out,label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    print('Epoch = ',itr+1)
    print('Loss = ',loss.item())
    
en_time = time.time() 
tt = en_time-st_time
print("Total Training Time = ",tt)

torch.save(cnnModel,f"cnn_model_{time.time()}")

def prediction(data_loader):
    mypredictedList = []
    act_label = []
    accuracy = 0
    with torch.no_grad():
        samples = 0
        correct = 0
        for img,label in data_loader:
            img = img.to(device)
            label = label.to(device)
            out = cnnModel(img)
            _,predicted = torch.max(out.data,1)
            pred_y = out.argmax(1)
            mypredictedList += pred_y
            act_label += label
            samples +=label.size(0)
            correct += (predicted==label).sum().item()
        
        accuracy = (correct/samples)*100    
    return accuracy,mypredictedList,act_label

def check_acc(act,pred):
    count=0
    total = len(act)
    for i in range(total):
        if act[i]==pred[i]:
            count+=1
            
    return (count/total)*100


train_acc,train_pred,train_act = prediction(train_loader)
print("Accuracy of network on training set = ",train_acc)





test_csv = csv_path+"/non_comp_test_x.csv"
test_label_csv = csv_path+"/non_comp_test_y.csv"

data_t = pd.read_csv(test_csv)
data_tl = pd.read_csv(test_label_csv)

names = data_t['Cover_image_name']
im_path = csv_path+"/images/images/"
#../input/col774-2022/images/images

test_x = im_path+names
test_y = data_tl['Genre']

test_data = pd.concat([test_x, test_y], axis=1)

# image_transformer2 = transforms.Compose([transforms.ToTensor(),transforms.Resize((64,64)),transforms.Normalize(mean=[R_mean2, G_mean2, B_mean2],
#                                                           std=[R_std2, G_std2, B_std2])
#                                      ])

test_dataset = MyDataset(test_data,image_transformer)
test_loader = DataLoader(test_dataset,batch_size=b_size,shuffle=True)

test_acc,test_pred,test_act = prediction(test_loader)
print("Accuracy of network on test set = ",test_acc)
print("Test acc = ",check_acc(test_act,test_pred))


#df = pd.DataFrame({"Id": ids.to("cpu"), "Genre": preds_test.to("cpu")})
#df.sort_values(by=["Id"], inplace=True)
#df = pd.DataFrame("Genre":test_pred.to("cpu"))
lis = [* range(0, len(test_act), 1)]


pred_test = []
for i in test_pred:
    pred_test.append(int(i.item()))

df = pd.DataFrame({"Id":lis, "Genre":pred_test})
df.to_csv("non_comp_test_pred_y.csv")
