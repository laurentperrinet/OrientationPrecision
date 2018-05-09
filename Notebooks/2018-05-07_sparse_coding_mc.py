
# coding: utf-8

# # 2018-05-07 Vers un meilleur réseau entrainé sur MC
# Avec du des relus pour sparsifier la première couche, 16 orientations de $\theta$ pour faire plus proche de la biologie.
# 
# Commençons par générer les motionclouds sur du B$\theta$ de 1 à 15° :

# In[1]:


import numpy as np
import MotionClouds as mc
import matplotlib.pyplot as plt
import os
import imageio

downscale = 1
fig_width = 21
fx, fy, ft = mc.get_grids(mc.N_X/downscale, mc.N_Y/downscale, 1)

N_theta = 120
N_theta_test = 12

#16 angles entre 0 et pi
theta_values = np.linspace(0,np.pi,16)

#120 values de traning, 12 de test
bw_values = np.pi*np.logspace(-7,-3.5, N_theta, base=2)
bw_test_values = np.pi*np.logspace(-7,-3.5, N_theta_test, base=2)

#generer training
for t in theta_values :
    if not os.path.exists('./16_clouds_easy/%s' % t): #si le folder n'existe pas on le crée
        os.makedirs('./16_clouds_easy/%s' % t)
    
    for i_ax, B_theta in enumerate(bw_values):
        mc_i = mc.envelope_gabor(fx, fy, ft, V_X=0., V_Y=0., B_V=0, theta=t, B_theta=B_theta)
        im = mc.random_cloud(mc_i)
        im = im
        imageio.imwrite('./16_clouds_easy/%s/B0 %s.png' % (t , (B_theta*180/np.pi) ) , im[:, :, 0])

#generer test
for t in theta_values :
    if not os.path.exists('./16_clouds_easy/%s' % t): #si le folder n'existe pas on le crée
        os.makedirs('./16_clouds_easy/%s' % t)
    if not os.path.exists('./16_clouds_easy_test/%s' % t):
        os.makedirs('./16_clouds_easy_test/%s' % t)

    for i_ax, B_theta in enumerate(bw_test_values):
        mc_i = mc.envelope_gabor(fx, fy, ft, V_X=0., V_Y=0., B_V=0, theta=t, B_theta=B_theta)
        im = mc.random_cloud(mc_i)
        
        imageio.imwrite('./16_clouds_easy_test/%s/B0 %s.png' % (t , (B_theta*180/np.pi) ) , im[:, :, 0])


# Cette fois ci on utilise les images en fullscale (256x256) :

# In[7]:


import torch
import torchvision
from torchvision import transforms, datasets

data_transform = transforms.Compose(
    [transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5,0.5), (0.5,0.5,0.5))])

#train
train_set = datasets.ImageFolder(root='16_clouds_easy',
                                transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=6, shuffle=True,
                                             num_workers=1, drop_last = True)

#test
test_set = datasets.ImageFolder(root='16_clouds_easy_test',
                                transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_set,
                                             batch_size=6,shuffle=False,
                                             num_workers=1, drop_last = True)


# Un test de display :

# In[3]:


# On utilise la fonction leaky_relu, qui est une relu normale mais avec une pente de x * 1e-2 quand x<0. Voir le benchmark de performance dans le notebook 2018-05-07_02 :
# 

# Le réseau, avec en entrée les images en 256 qui passent en RELU et en sortie la cross-entropy loss :

# In[8]:


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu1 = nn.Linear(256, 200)

        self.conv1 = nn.Conv2d(1, 6, 20)
        self.pool = nn.MaxPool2d(2,2)
        #self.conv2 = nn.Conv2d(6, 6, 5)

        self.fc1 = nn.Linear(63720,10000)
        self.fc2 = nn.Linear(10000,1000)
        self.fc3 = nn.Linear(1000,100)

        self.outlayer = nn.Linear(100,16)

    def forward(self, x):
        x = F.leaky_relu(self.relu1(x))

        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1) #reshape from conv to linear

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        x = self.outlayer(x)
        return x
        
model = Net()
print(model)


# On définit l'optimizer, en cross entropy (softmax+NLLL) :

# In[5]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# Et on entraine :

# In[9]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

import time
start_time = time.time()
print("Started training")

epochs = 5
print_interval = 50 #prints every p_i*4
tempo = []
acc = []

for epoch in range(epochs):  # nbr epochs
    for batch_idx, (data, target) in enumerate(train_loader): #nbr batch,in,out
        data, target = Variable(data), Variable(target)
        

        #init l'entrainement
        optimizer.zero_grad()
        net_out = model(data)

        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()

        #afficher la progression
        if batch_idx % print_interval == 0:
            #le print statement le plus illisible du monde
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
    tempo.append(epoch)
    acc.append(loss.data[0])
    
print("Finished training in  %.3f seconds " % (time.time() - start_time))

