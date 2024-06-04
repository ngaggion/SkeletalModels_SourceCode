import torch 

from finetune_dataset import MyDataset, reverse_scale_and_pad_coords
from model.hybridGNet3D import HybridGNet3D
import numpy as np
import pandas as pd 
import os

# ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

dataset = MyDataset("splits/finetune_test_0.5.txt")

config = {}

# Assuming you have your device defined, e.g.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tetra = np.load("hgn_data/srep_0_faces.npy")

config['device'] = device

model_paths = ["Training/finetune_50_v2/Epoch_19.pt",
               "Training/finetune_40_v2/Epoch_19.pt",
               "Training/finetune_30_v2/Epoch_19.pt",
               "Training/finetune_20_v2/Epoch_19.pt",
               "Training/finetune_10_v2/Epoch_19.pt",
               "Training/scratch_50/lastEpoch.pt",
               "Training/random_rota/lastEpoch.pt"]

models = []

for path in model_paths:
    model = HybridGNet3D(config).to(device).float()               
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    models.append(model)

results = pd.DataFrame(columns=["Model", "Percentage", "MSE", "MAE", "RMSE"])

os.makedirs("Results/Finetune", exist_ok=True)

with torch.no_grad():
    for i in range(len(dataset)):
        image, coords, img_path = dataset[i]
        name = img_path.split("/")[-1].split(".")[0]
        image = image.unsqueeze(0).to(device)
        coords = coords.squeeze(0).cpu().detach().numpy()
        GT = reverse_scale_and_pad_coords(img_path, i, coords, [256,256,256])
        #meshio.write_points_cells("hippocampi_realigned/finetune/%s_GT.vtk" %name, GT, {"tetra": tetra})
        os.makedirs("Results/Finetune/GT", exist_ok=True)
        np.save("Results/Finetune/GT/%s.npy" %name, GT)

        for j in range(len(models)):
            model = models[j]
            model_name = model_paths[j].split("/")[1]
            type_model = model_name.split("_")[0] if model_name.split("_")[0] != "random" else "Simulated"
            percentage = model_name.split("_")[1] if model_name.split("_")[0] != "rota" else "0"
            
            output = model(image)
            output = output.squeeze(0).cpu().detach().numpy()

            name = img_path.split("/")[-1].split(".")[0]
            reconstruct = reverse_scale_and_pad_coords(img_path, i, output, [256,256,256])
            
            os.makedirs("Results/Finetune/%s" %model_name, exist_ok=True)

            #meshio.write_points_cells("hippocampi_realigned/finetune/%s.vtk" %name, reconstruct, {"tetra": tetra})
            np.save("Results/Finetune/%s/%s.npy" %(model_name, name), reconstruct)
            
            MSE = np.mean((reconstruct - GT)**2)
            MAE = np.mean(np.abs(reconstruct - GT))
            RMSE = np.sqrt(np.mean((reconstruct - GT)**2))

            print("%s, %s, MSE: %f, MAE: %f, RMSE: %f"%(i, model_name, MSE, MAE, RMSE))
            
            results = results.append({"Model": type_model, "Percentage": percentage, "MSE": MSE, "MAE": MAE, "RMSE": RMSE}, ignore_index=True)

    results.to_csv("Results/Finetune/Results.csv", index=False)