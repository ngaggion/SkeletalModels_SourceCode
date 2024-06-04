import torch 

from training_dataset import MyDataset, reverse_scale_and_pad_coords
from model.hybridGNet3D import HybridGNet3D
import numpy as np
import pandas as pd 
import os

# ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

dataset = MyDataset(split=0, training=False, validation=False)

config = {}

# Assuming you have your device defined, e.g.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tetra = np.load("hgn_data/srep_0_faces.npy")

config['device'] = device

model_paths = ["Training/random_rota/lastEpoch.pt"]

models = []

for path in model_paths:
    model = HybridGNet3D(config).to(device).float()               
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    models.append(model)

results = pd.DataFrame(columns=["Model", "Percentage", "MSE", "MAE", "RMSE"])

os.makedirs("Results/Simulated", exist_ok=True)
os.makedirs("Results/Simulated/GT", exist_ok=True)

with torch.no_grad():
    for i in range(len(dataset)):
        image, coords, img_path = dataset[i]
        name = img_path.split("/")[-1].split(".")[0]
        image = image.unsqueeze(0).to(device)
        coords = coords.squeeze(0).cpu().detach().numpy()
        GT = reverse_scale_and_pad_coords(img_path, i, coords, [256,256,256])
        #meshio.write_points_cells("hippocampi_realigned/finetune/%s_GT.vtk" %name, GT, {"tetra": tetra})
        np.save("Results/Simulated/GT/%s.npy" %name, GT)

        for j in range(len(models)):
            model = models[j]

            model_name = model_paths[j].split("/")[1]
            type_model = model_name.split("_")[0] if model_name.split("_")[0] != "random" else "Simulated"
            
            output = model(image)
            output = output.squeeze(0).cpu().detach().numpy()

            name = img_path.split("/")[-1].split(".")[0]
            reconstruct = reverse_scale_and_pad_coords(img_path, i, output, [256,256,256])
            
            os.makedirs("Results/Simulated/%s" %model_name, exist_ok=True)

            #meshio.write_points_cells("hippocampi_realigned/finetune/%s.vtk" %name, reconstruct, {"tetra": tetra})
            np.save("Results/Simulated/%s/%s.npy" %(model_name, name), reconstruct)
            
            MSE = np.mean((reconstruct - GT)**2)
            MAE = np.mean(np.abs(reconstruct - GT))
            RMSE = np.sqrt(np.mean((reconstruct - GT)**2))

            print("%s, %s, MSE: %f, MAE: %f, RMSE: %f"%(i, model_name, MSE, MAE, RMSE))
            
            results = results.append({"Model": type_model, "MSE": MSE, "MAE": MAE, "RMSE": RMSE}, ignore_index=True)

    results.to_csv("Results/Simulated/Results.csv", index=False)