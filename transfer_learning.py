import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn import svm
from vision import FontDataset
import numpy as np 
from sklearn.metrics import accuracy_score


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
        dic = {}
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))       
        x = self.pool(F.relu(self.conv4(x)))       
        
                
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        dic['dense1'] = x.clone()
        x = F.relu(self.fc2(x))
        dic['dense2'] = x.clone()
        x = F.softmax(self.fc3(x), dim=1)
        
        return x, dic  
    
if __name__ == '__main__':
    SEED = 42         
    EPOCHS = 5
    bs = 1000        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'        
    plot_every = 10 
    save_every = 10 
    print_every = 2

    
    ds = FontDataset('project_training_data', 'svm', SEED)
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    
    data_size = len(ds)
    dense1_np = np.empty((0, 256))
    dense2_np = np.empty((0, 128))
    
    net = ConvNet().to(device)
    net.load_state_dict(torch.load('model-32c-64c-128c-256c-256L-128L-90p.pt'))
    
    clf1 = svm.SVC()
    clf2 = svm.SVC()
    
    labels = []
    for i_batch, (d, l) in enumerate(dl):
        d = d.to(device)
        
        l = torch.argmax(l, dim=1)
    
        labels += [chr(c + 65) for c in l]
        
        _, dic = net(d)
        dense_1 = dic['dense1'].detach().cpu().numpy()
        dense_2 = dic['dense2'].detach().cpu().numpy()
        
        dense1_np = np.vstack((dense1_np, dense_1))
        dense2_np = np.vstack((dense2_np, dense_2))
        
    labels = np.array(labels)    
    clf1.fit(dense1_np, labels)
    clf2.fit(dense2_np, labels)
    
    
    ds = FontDataset('project_training_data', 'test', SEED)
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    
    labels = np.empty((0, 1))
    preds_1_all = np.empty((0, 1))
    preds_2_all = np.empty((0, 1))
    
    for i_batch, (d, l) in enumerate(dl):
        d = d.to(device)
        
        l = torch.argmax(l, dim=1)
        tmp = np.array([chr(c + 65) for c in l])[:, np.newaxis]
        
        labels = np.vstack((labels, tmp))
        
        
        _, dic = net(d)
        dense_1 = dic['dense1'].detach().cpu().numpy()
        dense_2 = dic['dense2'].detach().cpu().numpy()
        
        preds_1 = clf1.predict(dense_1)[:, np.newaxis]
        preds_1_all = np.vstack((preds_1_all, preds_1))
        
        preds_2 = clf2.predict(dense_2)[:, np.newaxis]
        preds_2_all = np.vstack((preds_2_all, preds_2))
        
        if i_batch % print_every == 0:
            acc1 = accuracy_score(labels, preds_1_all)
            acc2 = accuracy_score(labels, preds_2_all)
            
            print(f'accuracy of the first dense layer: {acc1}')
            print(f'accuracy of the second dense layer: {acc2}\n')
            
        

    acc1 = accuracy_score(labels, preds_1_all)
    acc2 = accuracy_score(labels, preds_2_all)
    print(f'final accuracy of the first dense layer is: {acc1}')
    print(f'final accuracy of the second dense layer is: {acc2}')


        
        
        
        
        
        
        
        
        
        
    
    
    
