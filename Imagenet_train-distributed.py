#!/usr/bin/env python
# coding: utf-8

# In[30]:


import torch
import horovod.torch as hvd


# ### Are GPU's available?

# In[31]:


# Initialize Horovod
hvd.init()


# In[32]:


gpus = False
n_gpus = 0

if torch.cuda.is_available():
    gpus = True
    n_gpus = torch.cuda.device_count()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())


# In[33]:


n_gpus


# In[34]:


print(hvd.size())


# In[35]:


print(hvd.rank())


# In[36]:


from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

train_dataset = datasets.ImageFolder(root='tiny-imagenet-200/train',
                                           transform=data_transform)

# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 128,sampler=train_sampler)


# In[37]:


val_dataset = datasets.ImageFolder(root='tiny-imagenet-200/val',
                                           transform=data_transform)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=False
                                          )


# In[38]:


test_dataset = datasets.ImageFolder(root='tiny-imagenet-200/test',
                                           transform=data_transform)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                             shuffle=False
                                          )


# In[39]:


import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Uncomment for parallelism
        self.resnet_50 = nn.DataParallel(models.resnet50(pretrained = True),device_ids = [i for i in range(n_gpus)])
        #self.resnet_50 = models.resnet50(pretrained = True)
        self.output = nn.Linear(1000, 200)
        

    def forward(self, x):
        x = self.resnet_50(x)
        x = F.relu(x)
        # No need for softmax as CrossEntropy already implements it
        x = self.output(x)
        return x

net = Net()
net.train(True)


# In[40]:


if gpus:
    net = net.cuda()
#     net = net.to('cuda:0')


# In[41]:


import torch.optim as optim
from torch import nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())


# In[42]:


# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(net.state_dict(), root_rank=0)


# In[ ]:


from time import time

start = time()

for epoch in range(2):
    start_epoch = time()
    running_loss=0.0
    for i, data in enumerate(train_loader, start = 0):
        inputs, labels = data
        if gpus:
            labels = labels.to('cuda:0') #Move to GPU
        start_mini = time()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        end_mini = time()
        
        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 1000 mini-batches
            print('epoch %d, minibatch number %5d training loss: %.3f and time taken is %.3f seconds' %
                  (epoch + 1, i + 1, running_loss / 10,(end_mini - start_mini)))
            running_loss = 0.0
            
    end_epoch = time()
    print("Epoch %d took %.3f minutes"%(epoch + 1,(end_epoch - start_epoch )/60))
    print("")

end = time()    
print("")
print('Finished Training after %.3f hours'%((end-start) / 60 / 60 ))


# In[ ]:


def fetch_accuracy(loader):
    with torch.no_grad():
            net.to('cuda:0')
            net.eval()
            count = 0
            for inputs in loader:
                inputs ,labels = inputs
                if gpus:
                    inputs = inputs.to('cuda:0')
                    labels = labels.to('cuda:0')
                predictions = net(inputs)
                _, predicted = torch.max(predictions, 1)
                if predicted == labels:
                    count = count + 1
            return count / len(loader)


# In[ ]:


print("Test accuracy:")
print(fetch_accuracy(test_loader))

