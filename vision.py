import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim 
import os 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
import random 

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv4 = nn.Conv2d(128, 256, 5, padding=2)
        
        self.fc1 = nn.Linear(int(2*2*256), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 26)
        
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))       
        x = self.pool(F.relu(self.conv4(x)))       
        
                
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = F.softmax(self.fc3(x), dim=1)
        return x 
        

class FontDataset(Dataset):
    def __init__(self, root, mode, seed):
        self.root_folders = os.listdir(root)
        self.file_adds = []
        self.all_files = []
        for f in self.root_folders:
            files_in_folder = [root + '/' + f +'/'+ n for n in os.listdir(root + '/' + f)]
            self.all_files += files_in_folder
            
        random.seed(seed)
        random.shuffle(self.all_files)
        
        if mode == 'train':
            self.file_adds = [self.all_files[i] for i in range(int(0.8 * len(self.all_files)))]
        elif mode == 'validation':
            self.file_adds = [self.all_files[i] for i in range(int(0.8 * len(self.all_files)), int(0.9 * len(self.all_files)))]
        elif mode == 'test':
            self.file_adds = [self.all_files[i] for i in range(int(0.9 * len(self.all_files)), len(self.all_files))]
        elif mode == 'svm':
            self.file_adds = [self.all_files[i] for i in range(20_000)]
        
                             
    
    def __len__(self):
        return len(self.file_adds)
    
    def __getitem__(self, idx):
        raw_img = np.array(Image.open(self.file_adds[idx]))
        img_ten = torch.tensor(raw_img).unsqueeze(0) 
        img_ten = img_ten.float() / 255.0 
        
        alph = self.file_adds[idx].split('/')[2].split('_')[0]
        alph = ord(alph) - 65 
        
        label = torch.zeros(26)
        label[alph] = 1
        
        return img_ten, label
    
    
if __name__ == '__main__':
    SEED = 42         
    EPOCHS = 5
    bs = 128        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'        
    plot_every = 10 
    save_every = 10 
    
    ds = FontDataset('project_training_data', 'train', SEED)
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    
    
    
    net = ConvNet().to(device)
    loss_net = nn.BCELoss().to(device)
    
    
    
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.28, 0.93))
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
    
    
    
    losses = []
    
    
    for e in range(EPOCHS):
        for i_batch, (d, l) in enumerate(dl):
            d = d.to(device)
            l = l.to(device).float()
            
            optimizer.zero_grad()
            out = net(d)
            
            loss = loss_net(out,  l)
            
            loss.backward()
            optimizer.step()
            
            if i_batch % plot_every == 0:
                print(loss.item())
  
                losses.append(loss.item()) 
                plt.plot(losses)
                plt.show()
               
            if i_batch % save_every == 0:
                torch.save(net.state_dict(), 'model.pt')
            
        scheduler.step() 
        
        

        
    

