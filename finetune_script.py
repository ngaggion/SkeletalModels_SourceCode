import argparse
import os 
import time 

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from finetune_dataset import MyDataset
from model.hybridGNet3D import HybridGNet3D


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
            
            
def trainer(train_dataset, val_dataset, model, config):
    torch.manual_seed(0)

    device = config['device']
    model = model.to(device)

    train_loader = MultiEpochsDataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True, 
                                               num_workers = 4, pin_memory = True, persistent_workers = True)
                                       
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
    
    print('Training ...')

    N_images = len(train_dataset)
    subepochs = 450 // (N_images // 4) 
    print("Total subepochs:", subepochs)
    print("Batches per epoch:", subepochs * (N_images // 4))

    for epoch in range(config['epochs']):
        model.train()

        train_loss_avg.append(0)
        train_rec_loss_avg.append(0)
        train_kld_loss_avg.append(0)
    
        num_batches = 0

        t0 = time.time()

        for subepoch in range(subepochs):
            print("Subepoch", subepoch)
            t1 = time.time()

            for sample_batched in train_loader:
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

                print(num_batches, "loss", train_rec_loss_avg[-1] / num_batches, end = '\r')
                
            print("")
            t2 = time.time()
            print('Time for training subepoch: %s'%(t2-t1))
        
        t2 = time.time()
        
        train_loss_avg[-1] /= num_batches
        train_rec_loss_avg[-1] /= num_batches
        train_kld_loss_avg[-1] /= num_batches

        print('Epoch [%d / %d] train average reconstruction error: %f' % (epoch+1, config['epochs'], train_rec_loss_avg[-1]))
        print('Time for training epoch: %s'%(t2-t0))
        
        model.eval()
        val_loss_avg.append(0)
        num_batches = 0

        with torch.no_grad():
            t0 = time.time()
            
            for j in range(len(val_dataset)):
                images, coordinates, path = val_dataset[j]
                images = images.to(device).unsqueeze(0)
                coordinates = coordinates.to(device).unsqueeze(0)

                reconstructed = model(images)
                
                # compute reconstruction loss and KLD
                rec_loss = F.mse_loss(reconstructed, coordinates, reduction='mean')

                val_loss_avg[-1] += rec_loss.item()

                num_batches += 1
                
                del images, coordinates, reconstructed

        val_loss_avg[-1] /= num_batches
        t2 = time.time()
        
        print('Epoch [%d / %d] validation average reconstruction error: %f' % (epoch+1, config['epochs'], val_loss_avg[-1]))
        print('Time for running validation: %s'%(t2-t0))

        writer.add_scalar('Train/Loss', train_loss_avg[-1], epoch)
        writer.add_scalar('Train/Loss kld', train_kld_loss_avg[-1], epoch)
        writer.add_scalar('Train/MSE', train_rec_loss_avg[-1], epoch)
        writer.add_scalar('Validation/MSE', val_loss_avg[-1] , epoch)
        
        torch.save(model.state_dict(), os.path.join(folder, "Epoch_%s.pt"%(epoch+1)))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", default = "TestRun", type=str)    
    parser.add_argument("--epochs", default = 20, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 1, type = int)
    parser.add_argument("--gamma", default = 0.99, type = float)

    parser.add_argument("--batch_size", default = 4, type = int)
    parser.add_argument("--val_batch_size", default = 1, type = int)
    parser.add_argument("--kld_weight", default = 1e-3, type = float)
    parser.add_argument("--weight_decay", default = 0, type = float)
    parser.add_argument("--percentage", default = 50, type = int)
    
    # add bool arguments for loading or not loading the model
    parser.add_argument("--load", default = False, type = bool)

    config = parser.parse_args()
    config = vars(config)

    if config['percentage'] == 50:
        split = "splits/finetune_train_0.5.txt"
    elif config['percentage'] == 40:
        split = "splits/finetune_train_0.4.txt"
    elif config['percentage'] == 30:
        split = "splits/finetune_train_0.3.txt"
    elif config['percentage'] == 20:
        split = "splits/finetune_train_0.2.txt"
    elif config['percentage'] == 10:
        split = "splits/finetune_train_0.1.txt"
    else:
        raise ValueError("Percentage not valid, please choose between 10, 20, 30, 40, 50")
    
    # Assuming you have your device defined, e.g.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    
    model = HybridGNet3D(config).to(device).float()
    
    if config['load']:
        model.load_state_dict(torch.load("Training/random_rota/lastEpoch.pt"))
        print("Loaded model")
            
    training_dataset = MyDataset(split=split, training=True)
    val_dataset = MyDataset(split="splits/finetune_test_0.5.txt", training=False)
    
    print("Training on %d samples"%(len(training_dataset)))

    trainer(training_dataset, val_dataset, model, config)
