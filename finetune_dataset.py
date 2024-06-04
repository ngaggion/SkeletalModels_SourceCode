import numpy as np
import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import cv2


def reverse_scale_and_pad_coords(data_path, index, coords, desired_shape):
    coords[:, 0] = coords[:, 0] * desired_shape[2]
    coords[:, 1] = coords[:, 1] * desired_shape[1]
    coords[:, 2] = coords[:, 2] * desired_shape[0]
        
    image = sitk.ReadImage(data_path)
    img = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    
    z = img.shape[0]
    h = img.shape[1]
    w = img.shape[2]
    
    padz = desired_shape[0] - z
    padh = desired_shape[1] - h
    padw = desired_shape[2] - w

    padding = ((padz//2, padz//2 + padz%2), (padh//2, padh//2 + padh%2), (padw//2, padw//2 + padw%2))
    
    coords = coords - np.array([padding[2][0], padding[1][0], padding[0][0]])

    coords *= image.GetSpacing()
    coords += image.GetOrigin()
    
    return coords
    
class MyDataset(Dataset):
    def __init__(self, split, training = False, validation = False, desired_shape = (256, 256, 256)):
        
        # Checks if the desired shape slices is a multiple of 4
        if desired_shape[0] % 4 != 0:
            raise ValueError("The number of slices must be a multiple of 4")
        
        # Checks if the desired shape height is a multiple of 32
        if desired_shape[1] % 32 != 0:
            raise ValueError("The height must be a multiple of 32")
        
        # Checks if the desired shape width is a multiple of 32
        if desired_shape[2] % 32 != 0:
            raise ValueError("The width must be a multiple of 32")
        
        self.training = training
        self.files = np.loadtxt(split, dtype = str)
        self.desired_shape = desired_shape
        self.paths = []
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        
        data_path = "hippocampi_realigned/volumes/" + self.files[index]
        
        image = sitk.ReadImage(data_path)
        img = sitk.GetArrayFromImage(image).transpose(2, 1, 0)

        coords_path = data_path.replace("volumes", "sreps").replace("_volume", "").replace(".nrrd", "_points.npy")
        coords = np.load(coords_path) 
        
        coords = coords - image.GetOrigin()
        coords = coords / image.GetSpacing()

        if self.training:
            img, coords = self.RandomRotation(img, coords)
            img, coords = self.RandomScaling(img, coords)
            img, coords = self.pad_image_and_coords(img, coords)
            img, coords = self.ToTensor(img, coords)
        else:
            img, coords = self.pad_image_and_coords(img, coords)
            img, coords = self.ToTensor(img, coords)

        if self.training:
            return img, coords
        else:
            return img, coords, data_path

    def pad_image_and_coords(self, img, coords):
        # Sanity check to make sure none of the dimensions of the image exceed the desired dimensions
        if any(i > d for i, d in zip(img.shape, self.desired_shape)):
            print("Image Shape:", img.shape)
            print("Desired Shape:", self.desired_shape)
            raise ValueError("Image dimensions are larger than the desired shape")

        z = img.shape[0]
        h = img.shape[1]
        w = img.shape[2]
        
        padz = self.desired_shape[0] - z
        padh = self.desired_shape[1] - h
        padw = self.desired_shape[2] - w

        if padz > 0:
            padz_1 = np.random.randint(0, padz)
            padz_2 = padz - padz_1
        else:
            padz_1 = 0
            padz_2 = 0

        if padh > 0:
            padh_1 = np.random.randint(0, padh)
            padh_2 = padh - padh_1
        else:
            padh_1 = 0
            padh_2 = 0

        if padw > 0:
            padw_1 = np.random.randint(0, padw)
            padw_2 = padw - padw_1
        else:
            padw_1 = 0
            padw_2 = 0
            
        padding = ((padz_1, padz_2), (padh_1, padh_2), (padw_1, padw_2))
        
        img = np.pad(img, padding, mode='constant', constant_values=0)
        
        # Add padding to coordinates
        coords[:, 0] = coords[:, 0] + padding[2][0]
        coords[:, 1] = coords[:, 1] + padding[1][0]
        coords[:, 2] = coords[:, 2] + padding[0][0]
        
        return img, coords

    def ToTensor(self, img, coords):
        img = torch.from_numpy(img).float()
        # Normalize image to be in range 0-1
        img = (img - img.min()) / (img.max() - img.min())
        img = img.unsqueeze(0)
        
        coords = torch.from_numpy(coords).float()

        # Normalize coordinates to be in range 0-1
        coords[:, 0] = coords[:, 0] / self.desired_shape[2]
        coords[:, 1] = coords[:, 1] / self.desired_shape[1]
        coords[:, 2] = coords[:, 2] / self.desired_shape[0]

        return img, coords

    def RandomScaling(self, img, coords):        
        # Randomly scales the image and the coordinates

        max_z_factor = self.desired_shape[0] / img.shape[0]
        max_h_factor = self.desired_shape[1] / img.shape[1]
        max_w_factor = self.desired_shape[2] / img.shape[2]

        resize_h_factor = np.random.uniform(0.60, max_h_factor)
        resize_w_factor = np.random.uniform(0.60, max_w_factor)
        resize_z_factor = np.random.uniform(0.60, max_z_factor)
                
        z, h, w = img.shape
                
        new_z = int(round(z * resize_z_factor, 0))
        new_h = int(round(h * resize_h_factor, 0))
        new_w = int(round(w * resize_w_factor, 0))
        
        resize_h_factor = new_h / h
        resize_w_factor = new_w / w
        resize_z_factor = new_z / z
        
        # Not optimal, but it works for now
        image = cv2.resize(img, (new_h, new_z))
        image = image.transpose(2, 1, 0)  
        image = cv2.resize(image, (new_h, new_w))
        image = image.transpose(2, 1, 0)
        
        coords[:, 2] *= resize_z_factor        
        coords[:, 1] *= resize_h_factor
        coords[:, 0] *= resize_w_factor
        
        return image, coords
    
    def RandomRotation(self, img, coords):        
        # Pad 3D image to 350x350x350

        img = np.pad(img, ((50, 50), (50, 50), (50, 50)), mode="constant", constant_values=0)
        coords = coords + np.array([50, 50, 50])

        # Rotation in z-axis
        angle = np.random.normal(0, 5)
        center = (175, 175)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (350, 350))

        subset_coords = coords[:, [1,0]]
        subset_coords = subset_coords - np.array([175, 175])
        subset_coords = np.matmul(rotation_matrix[:,:2], subset_coords.T).T
        subset_coords = subset_coords + np.array([175, 175])
        coords[:, [1,0]] = subset_coords

        # Rotation in x-axis
        img = img.transpose(1, 2, 0)
        angle = np.random.normal(0, 5)
        center = (175, 175)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (350, 350))

        subset_coords = coords[:, [2,1]]
        subset_coords = subset_coords - np.array([175, 175])
        subset_coords = np.matmul(rotation_matrix[:,:2], subset_coords.T).T
        subset_coords = subset_coords + np.array([175, 175])
        coords[:, [2,1]] = subset_coords
        img = img.transpose(2, 0, 1)

        # Rotation in y-axis
        img = img.transpose(2, 0, 1)
        angle = np.random.normal(0, 5)
        center = (175, 175)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (350, 350))

        subset_coords = coords[:, [0,2]]
        subset_coords = subset_coords - np.array([175, 175])
        subset_coords = np.matmul(rotation_matrix[:,:2], subset_coords.T).T
        subset_coords = subset_coords + np.array([175, 175])
        coords[:, [0,2]] = subset_coords
        img = img.transpose(1, 2, 0)

        x_min = np.min(coords[:, 0])
        x_max = np.max(coords[:, 0])
        y_min = np.min(coords[:, 1])
        y_max = np.max(coords[:, 1])
        z_min = np.min(coords[:, 2])
        z_max = np.max(coords[:, 2])

        x0 = int(np.floor(x_min))
        x1 = int(np.ceil(x_max))
        y0 = int(np.floor(y_min))
        y1 = int(np.ceil(y_max))
        z0 = int(np.floor(z_min))
        z1 = int(np.ceil(z_max))

        x0 = max(0, x0)
        x1 = min(350, x1)
        y0 = max(0, y0)
        y1 = min(350, y1)
        z0 = max(0, z0)
        z1 = min(350, z1)

        coords = coords - np.array([x0, y0, z0])
        img = img[x0:x1, y0:y1, z0:z1]
                
        return img, coords
