import argparse
import os 
import time 

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from training_dataset import MyDataset

from model.hybridGNet3D import HybridGNet3D

def trainer(train_dataset, val_dataset, model, config):
    torch.manual_seed(0)

    device = config['device']
    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True, num_workers = 4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = config['val_batch_size'], num_workers = 0)

    optimizer = torch.optim.Adam(params = model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

    train_loss_avg = []
    train_kld_loss_avg = []
    train_rec_loss_avg = []
    val_loss_avg = []

    tensorboard = "Training"
    
    folder = os.path.join(tensorboard, config['name'])

    try:
        os.makedirs(folder)
    except:
        pass 

    writer = SummaryWriter(log_dir = folder)  

    bestMSE = 1e12
    
    print('Training ...')

    for epoch in range(config['epochs']):
        model.train()

        train_loss_avg.append(0)
        train_rec_loss_avg.append(0)
        train_kld_loss_avg.append(0)
    
        num_batches = 0

        t0 = time.time()
        
        for sample_batched in train_loader:
            print(num_batches, end = '\r')
            images, coordinates = sample_batched
            images = images.to(device)
            coordinates = coordinates.to(device)
            
            optimizer.zero_grad()

            reconstructed = model(images)
                        
            # MSE loss
            rec_loss = F.mse_loss(reconstructed, coordinates, reduction='mean')

            # Kullback-Leibler divergence term
            kld_loss = -0.5 * torch.mean(torch.mean(1 + model.log_var - model.mu ** 2 - model.log_var.exp(), dim=1), dim=0)
            
            loss = rec_loss + config['kld_weight'] * kld_loss

            loss.backward()

            optimizer.step()

            train_loss_avg[-1] += loss.item()
            train_rec_loss_avg[-1] += rec_loss.item()
            train_kld_loss_avg[-1] += kld_loss.item()

            num_batches += 1
        
        t2 = time.time()
        
        train_loss_avg[-1] /= num_batches
        train_rec_loss_avg[-1] /= num_batches
        train_kld_loss_avg[-1] /= num_batches

        print('Epoch [%d / %d] train average reconstruction error: %f' % (epoch+1, config['epochs'], train_rec_loss_avg[-1]))
        print('Time for training epoch: %s'%(t2-t0))

        num_batches = 0

        model.eval()
        val_loss_avg.append(0)

        with torch.no_grad():
            t0 = time.time()
            for sample_batched in val_loader:
                images, coordinates = sample_batched
                images = images.to(device)
                coordinates = coordinates.to(device)

                reconstructed = model(images)
                
                # compute reconstruction loss and KLD
                rec_loss = F.mse_loss(reconstructed, coordinates, reduction='mean')

                val_loss_avg[-1] += rec_loss.item()

                num_batches += 1

        val_loss_avg[-1] /= num_batches
        t2 = time.time()
        
        print('Epoch [%d / %d] validation average reconstruction error: %f' % (epoch+1, config['epochs'], val_loss_avg[-1]))
        print('Time for running validation: %s'%(t2-t0))

        writer.add_scalar('Train/Loss', train_loss_avg[-1], epoch)
        writer.add_scalar('Train/Loss kld', train_kld_loss_avg[-1], epoch)
        writer.add_scalar('Train/MSE', train_rec_loss_avg[-1], epoch)

        writer.add_scalar('Validation/MSE', val_loss_avg[-1] , epoch)

        if val_loss_avg[-1] < bestMSE:
            bestMSE = val_loss_avg[-1]
            print('Model Saved MSE')
            torch.save(model.state_dict(), os.path.join(folder,  "bestEpochMSE.pt"))
    
        torch.save(model.state_dict(), os.path.join(folder, "lastEpoch.pt"))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", default = "TestRun", type=str)    
    parser.add_argument("--epochs", default = 100, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 1, type = int)
    parser.add_argument("--gamma", default = 0.99, type = float)

    parser.add_argument("--batch_size", default = 4, type = int)
    parser.add_argument("--val_batch_size", default = 1, type = int)
    parser.add_argument("--kld_weight", default = 1e-3, type = float)
    parser.add_argument("--weight_decay", default = 0, type = float)
    parser.add_argument("--train_split", default = 0, type = int)

    config = parser.parse_args()
    config = vars(config)
    
    # Assuming you have your device defined, e.g.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    
    model = HybridGNet3D(config).to(device).float()
    
    training_dataset = MyDataset(split=config["train_split"], training=True)
    validation_dataset = MyDataset(split=config["train_split"], validation=True)
    
    print("Training on %d samples, validating on %d samples"%(len(training_dataset), len(validation_dataset)))

    trainer(training_dataset, validation_dataset, model, config)
