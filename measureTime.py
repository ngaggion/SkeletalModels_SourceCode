import torch 

from finetune_dataset import MyDataset
from model.hybridGNet3D import HybridGNet3D
import numpy as np
import time 

# ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

dataset = MyDataset("splits/finetune_test_0.5.txt")

config = {}

# Assuming you have your device defined, e.g.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

tetra = np.load("hgn_data/srep_0_faces.npy")

config['device'] = device

model_paths = ["Training/finetune_50_v2/Epoch_19.pt"]

models = []

for path in model_paths:
    model = HybridGNet3D(config).to(device).float()               
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    models.append(model)
    
elapsed = []

with torch.no_grad():
    for i in range(len(dataset)):
        image, coords, img_path = dataset[i]
        name = img_path.split("/")[-1].split(".")[0]
        image = image.unsqueeze(0).to(device)
        

        for j in range(len(models)):
            model = models[j]
            
            t1 = time.time()
            
            output = model(image)
            
            t2 = time.time()
            
            elapsed.append(t2-t1)
    
        print(round(np.mean(elapsed),2))
    
print(round(np.mean(elapsed),2))