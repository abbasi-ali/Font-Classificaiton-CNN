from vision import ConvNet, FontDataset
import torch 
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

np.set_printoptions(linewidth=np.inf)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bs = 1000 
SEED = 42
print_every = 2

ds = FontDataset('project_training_data', 'test', SEED)
dl = DataLoader(ds, batch_size=bs, shuffle=False)

net = ConvNet().to(device)
net.load_state_dict(torch.load('model-32c-64c-128c-256c-256L-128L-90p.pt'))

loss_net = torch.nn.BCELoss().to(device)

labels = []
chrs = []
loss_av = 0 
cnt = 0
for i_batch, (img_ten, la) in enumerate(dl):
    img_ten = img_ten.to(device)
    
    la = la.to(device).float()
    
    l = torch.argmax(la, dim=1)
    
    labels += [chr(c + 65) for c in l]
    
    out = net(img_ten)
    loss = loss_net(out,  la)
    
    out = torch.argmax(out, dim=1)
    
    chrs += [chr(c + 65) for c in out]
    loss_av += loss.item()
    cnt = i_batch
    if i_batch % print_every == 0:
        acc = accuracy_score(labels, chrs)
        print(f'accuracy: {acc}      loss:{loss.item()}')
        
loss_av = loss_av / (cnt + 1)
acc = accuracy_score(labels, chrs)
conf = confusion_matrix(labels, chrs)
print(f'final accuracy is {acc}\n average loss is {loss_av}')
print(f'confusion matrix is\n{conf}')




