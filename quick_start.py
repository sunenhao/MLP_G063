
# coding: utf-8

# In[ ]:


# First, Please turn on the GPU on Kaggle.

import os
import time
import shutil
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torchvision
from torchvision.datasets import ImageFolder
import argparse

# get_ipython().run_line_magic('matplotlib', 'inline')

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--model_save_name", help="specify model name to save")
args = parser.parse_args()

print("This quick start")
if_gpu = torch.cuda.is_available()
print("GPU is on?", if_gpu)


# Define a dataset

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        return self.to_tensor(self.x_data[index]), self.y_data[index]

    def __len__(self):
        return len(self.x_data)


# In[ ]:


# Note that torchvision.transforms.ToTensor() will
# Converts a PIL Image or numpy.ndarray (H x W x C) in the range
# [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

train_dict = np.load("data/plant-train-data.npz")
whole_dataset = ImageDataset(train_dict["data"], train_dict["labels"])

print(whole_dataset[0][0].shape)
print(whole_dataset[4610])


# In[ ]:


# subset the whole train set for accuracy check while training.

def array_random_pick(array, pick_num):
    index = np.arange(len(array))
    pick = np.random.choice(len(array), pick_num, replace=False)
    unpick = np.equal(np.in1d(index, pick), False)
    return array[unpick], array[pick]

train_mask, valid_mask = array_random_pick(np.arange(len(whole_dataset)), 500)

train_set = torch.utils.data.Subset(whole_dataset, train_mask)
valid_set = torch.utils.data.Subset(whole_dataset, valid_mask)

print(len(train_set),len(valid_set))
print(train_set[4010])
print(valid_set[401])


# In[ ]:


# Use DataLoader to group data batchs. Here use size 4 for a batch.
# DataLoader will return a iterator.

train_loader = torch.utils.data.DataLoader(train_set, batch_size=5)
load_iter = iter(train_loader)
one_batch_x, one_batch_y = next(load_iter)

print(one_batch_y)
print(one_batch_x.shape)


# In[ ]:


# Use PyTorch's built-in model to generate AlexNet with classes 12.
# With input data of size [4, 3, 224, 224], AlexNet will output data of size [4, 12].

alex = torchvision.models.AlexNet(num_classes = 12)
alex_out = alex(one_batch_x)
print(alex_out.shape)
print(alex_out)


# We use the max index of alex_out to
# evaluate the accuracy of model predict.
# Now the accuracy is zero before model train.

predict = torch.argmax(alex_out, dim = 1)
compare = predict == one_batch_y
accuracy = compare.sum() / len(predict)

print(predict)
print(one_batch_y)
print(compare)
print("accuracy =", accuracy.data.numpy())

# Print model's state_dict

# In[ ]:

# define a base utility to train a net.

class BaseNetPyTorch:
    def __init__(self):
        self.train_loader = None
        self.sub_train_loader = None
        self.valid_loader = None

        self.model = None
        self.optimize_method = None
        self.loss_function = None

        if_gpu = torch.cuda.is_available()
        self.device_gpu = torch.device("cuda:1" if if_gpu else "cpu")

    def train_loss(self):
        # "training" mode for Dropout etc.
        self.model.train()

        train_loss = None
        for (x, y) in self.train_loader:
            x_gpu = x.to(self.device_gpu)
            y_gpu = y.long().to(self.device_gpu)

            predict = self.model(x_gpu)
            train_loss = self.loss_function(predict, y_gpu)

            self.optimize_method.zero_grad()
            train_loss.backward()
            self.optimize_method.step()
        return train_loss

    def predict_index(self, check_loader):
        predict_list = []
        for x, y in check_loader:
            x_gpu = x.to(self.device_gpu)
            predict = self.model(x_gpu)
            max_index = torch.argmax(predict, dim=1)
            predict_list += max_index.cpu().data.numpy().tolist()
        return predict_list

    def check_accuracy(self, check_set):
        num_correct = 0
        num_samples = 0
        # "test" mode for Dropout etc.
        self.model.eval()
        for x, y in check_set:
            x_gpu = x.to(self.device_gpu)
            y_gpu = y.to(self.device_gpu)

            predict = self.model(x_gpu)
            max_index = torch.argmax(predict, dim=1)

            num_correct += (max_index == y_gpu).sum()
            num_samples += max_index.shape[0]

        accuracy = float(num_correct) / float(num_samples)
        return num_correct, num_samples, accuracy

    def train(self, num_epochs=1):
        if self.model is None:
            raise ValueError("self.model is None! Please assign it.")
        if self.optimize_method is None:
            raise ValueError("self.optimize_method is None! Please assign it.")
        if self.loss_function is None:
            raise ValueError("self.loss_function is None! Please assign it.")

        print("begin training, length_of_one_mini_batch :", len(self.train_loader))

        self.model = self.model.to(self.device_gpu)

        train_time = time.time()
        for epoch in range(num_epochs):
            epoch_start = time.time()

            loss = self.train_loss()
            loss_time = time.time()

            train_correct, train_samples, train_acc = self.check_accuracy(self.sub_train_loader)
            train_acc_time = time.time()

            valid_correct, valid_samples, valid_acc = self.check_accuracy(self.valid_loader)
            valid_acc_time = time.time()

            epoch_time = time.time()

            print('epoch:%d/%d' % (epoch + 1, num_epochs), end=" ")
            print('loss:%.4f|%ds' % (loss.data, (loss_time - epoch_start)), end=" ")
            print('train_acc:(%d/%d %0.2f%%)|%ds' %
                  (train_correct, train_samples, 100 * train_acc, train_acc_time - loss_time), end=' ')
            print('valid_acc:(%d/%d %0.2f%%)|%ds' %
                  (valid_correct, valid_samples, 100 * valid_acc, (valid_acc_time - train_acc_time)), end=' ')
            print("take:%dmin remain:%dmin" %
                  ((epoch_time - train_time) / 60, (epoch_time - epoch_start) * (num_epochs - epoch) / 60))

            if (train_acc - 0.3 > valid_acc) and (train_acc > 0.5):
                print("Model Overfit 30.00%, stopped.")
                return True

        return True


# In[ ]:


# It is very important to turn on shuffle=True of training set

train_loader = torch.utils.data.DataLoader(train_set, batch_size=40, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=40)

net = BaseNetPyTorch()
net.model = torchvision.models.AlexNet(num_classes=12)
net.optimize_method = torch.optim.Adam(net.model.parameters(), lr=0.0001)
net.loss_function = torch.nn.CrossEntropyLoss()

net.train_loader = train_loader
net.sub_train_loader = train_loader
net.valid_loader = valid_loader

print("Net's state_dict:")
for param_tensor in net.model.state_dict():
    print(param_tensor, "\t", net.model.state_dict()[param_tensor].size())

#save state at prespecified filepath
model_save_dir = "models"
model_idx = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
torch.save(net.model.state_dict(), f=os.path.join(model_save_dir, "{}_{}".format(args.model_save_name, str(model_idx))))

#net.train(1)

# In[ ]:

'''
# predict test file labels
test_dict = np.load("data/plant-test-data.npz")
train_info_dict = np.load("data/plant-train-info.npz")

test_set = ImageDataset(test_dict["data"], test_dict["labels"])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=40)
label_names = train_info_dict["label_names"]

test_predict = net.predict_index(test_loader)
predict_names = [label_names[i] for i in test_predict]

print(test_predict[:10])
print(predict_names[:10])


# In[ ]:


# classify test_files to different sub_folders

test_file_paths = test_info_dict["file_paths"]
save_folder = "../working/tmp/predict"

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy2sub_folders(source_file_paths, sub_folder_names, to_folder):
    for i in tqdm(range(len(source_file_paths))):
        file_dir = os.path.join(to_folder, sub_folder_names[i])
        make_dirs(file_dir)
        shutil.copy2(source_file_paths[i], file_dir)
        
copy2sub_folders(test_file_paths, predict_names, save_folder)

print(os.listdir("../working/tmp/predict"))


# In[ ]:


# Create a predict submission file.

def folder_file_info(root):
    folder_file_list = []
    path_dirs = os.listdir(root)
    for folder in path_dirs:
        dir_files = os.listdir(os.path.join(root, folder))
        for file_name in dir_files:
            folder_file_list.append([file_name, folder])
    return folder_file_list


file_predict_table = folder_file_info("../working/tmp/predict")
df = pd.DataFrame(file_predict_table, columns=['file', 'species'])
df.to_csv("predict_submission.csv", index=False)
print(df)


# In[ ]:


# delect temporary working folder before Kaggle Commit.
if os.path.exists("../working/tmp"):
    shutil.rmtree("../working/tmp")

os.listdir("../working")
'''
