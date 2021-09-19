
from vision import ConvNet, FontDataset
import torch 
from torch.utils.data import DataLoader



def test_on_numpy_array(img_np):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ConvNet().to(device)
    net.load_state_dict(torch.load('model-32c-64c-128c-256c-256L-128L-90p.pt'))
    
    chrs = []
    
    img_ten = torch.tensor(img_np).unsqueeze(dim=1).to(device)
    out = net(img_ten)
    out = torch.argmax(out, dim=1)
    chrs = [chr(c + 65) for c in out]
    outF = open("result.txt", "w")
    
    for c in chrs:
        outF.write(f'{c}\n')
    
    outF.close()


# bs = 1000 
# SEED = 42


# ds = FontDataset('project_training_data', 'test', SEED)
# dl = DataLoader(ds, batch_size=bs, shuffle=False)

# img_ten, _ = next(iter(dl))
# img_np = img_ten.squeeze().detach().numpy()



# test_on_numpy_array(img_np)