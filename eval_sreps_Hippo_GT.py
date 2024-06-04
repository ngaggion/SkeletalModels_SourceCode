import numpy as np
import vtk

# 0:61 are skeletal points (61 total)
# 61:122 are top spoke ends (61 total)
# 122:183 are bottom spoke ends (61 total)
# 183: are crest ends (24 total)
# Crest edge points are 2,5,8,11,14,17,20,23,26,29,32,35,38,40,42,44,46,48,50,52,54,56,58,60

def medialness(srep):
    # for pairs of top/bottom spokes, compute ratio of top to bottom length
    # closer to 1 is ideal
    ratios = []
    for i in range(0,61):
        skel_pt = srep[i,:]
        top_pt = srep[61+i,:]
        bot_pt = srep[122+i,:]
        
        top_length = np.linalg.norm(top_pt - skel_pt)
        bot_length = np.linalg.norm(bot_pt - skel_pt)
        ratios.append(bot_length/top_length)
        
    return np.mean(ratios)

def angles_between_sreps(srep1,srep2):
    # Compute angle between corresponding spokes between sreps
    angles = []
    for i in range(0,61):
        skel_pt1 = srep1[i,:]
        top_pt1 = srep1[61+i,:]
        
        skel_pt2 = srep2[i,:]
        top_pt2 = srep2[61+i,:]
        
        # Angle between top spokes
        top_vec1 = top_pt1 - skel_pt1
        top_vec1 /= np.linalg.norm(top_vec1)
        
        top_vec2 = top_pt2 - skel_pt2
        top_vec2 /= np.linalg.norm(top_vec2)
        
        a = np.arccos( np.clip(np.dot(top_vec1,top_vec2),-1,1) )
        angles.append(a)
        
        # Angle between bottom spokes
        bot_pt1 = srep1[122+i,:]
        bot_vec1 = bot_pt1 - skel_pt1
        bot_vec1 /= np.linalg.norm(bot_vec1)
        
        bot_pt2 = srep2[122+i,:]
        bot_vec2 = bot_pt2 - skel_pt2
        bot_vec2 /= np.linalg.norm(bot_vec2)
        
        a = np.arccos( np.clip( np.dot(bot_vec1,bot_vec2),-1,1 ) )
        angles.append(a)
        
    # Angle between crest spokes
    crest_inds = [2,5,8,11,14,17,20,23,26,29,32,35,38,40,42,44,46,48,50,52,54,56,58,60]
    for i in range(0,24):
        skel_pt1 = srep1[crest_inds[i],:]
        bdry_pt1 = srep1[183+i,:]
        
        skel_pt2 = srep2[crest_inds[i],:]
        bdry_pt2 = srep2[183+i,:]
        
        vec1 = bdry_pt1 - skel_pt1
        vec1 /= np.linalg.norm(vec1)
        
        vec2 = bdry_pt2 - skel_pt2
        vec2 /= np.linalg.norm(vec2)
        
        a = np.arccos( np.clip( np.dot(vec1,vec2),-1,1 ) )
        angles.append(a)
        
    return np.mean(angles)
        
    
def orthogonality(srep,surface):
    norms = vtk.vtkPolyDataNormals()
    norms.SetInputData(surface)
    norms.AutoOrientNormalsOn()
    norms.Update()
    normals = norms.GetOutput().GetPointData().GetArray("Normals")
    
    loc = vtk.vtkKdTreePointLocator()
    loc.SetDataSet(surface)
    loc.BuildLocator() 
    
    angles = []
    
    for i in range(0,61):
        skel_pt = srep[i,:]
        top_pt = srep[61+i,:]
        bot_pt = srep[122+i,:]
        
        # Top first
        surf_ind = loc.FindClosestPoint(top_pt)
        n = np.array(normals.GetTuple(surf_ind))
        n /= np.linalg.norm(n)
        
        vec = top_pt - skel_pt
        vec /= np.linalg.norm(vec)
        angles.append( np.arccos(np.dot(vec,n)) ) 
        
        # Bottom
        surf_ind = loc.FindClosestPoint(bot_pt)
        n = np.array(normals.GetTuple(surf_ind))
        n /= np.linalg.norm(n)
        
        vec = bot_pt - skel_pt
        vec /= np.linalg.norm(vec)
        angles.append( np.arccos(np.dot(vec,n)) )
        
    crest_inds = [2,5,8,11,14,17,20,23,26,29,32,35,38,40,42,44,46,48,50,52,54,56,58,60]
    for i in range(0,24):
        skel_pt = srep[crest_inds[i],:]
        bdry_pt = srep[183+i,:]
        
        surf_ind = loc.FindClosestPoint(bdry_pt)
        n = np.array(normals.GetTuple(surf_ind))
        n /= np.linalg.norm(n)
        
        vec = bdry_pt - skel_pt
        vec /= np.linalg.norm(vec)
        angles.append( np.arccos(np.dot(vec,n)) ) 
        
    return np.mean(angles)

import os 
import pandas as pd 

models = os.listdir('Results/Finetune')
models = [m for m in models if 'GT' in m]

df = []

for model in models:
    preds = os.listdir('Results/Finetune/'+model)

    surface_path = "hippocampi_realigned/meshes/"

    for pred in preds:
        srep = np.load('Results/Finetune/'+model+'/'+pred)
        med = medialness(srep)

        #gt = np.load('Results/Finetune/GT/'+pred)
        #angle = angles_between_sreps(srep,gt)

        surface_name = surface_path + pred.split('.')[0].replace("_volume", "") + '_1.vtk'

        surface_reader = vtk.vtkPolyDataReader()
        surface_reader.SetFileName(surface_name)
        surface_reader.Update()

        ort = orthogonality(srep,surface_reader.GetOutput())

        # df.append is deprecated, do not use it
        if len(df) == 0:
            df = pd.DataFrame([[model, med, 0, ort]],columns=['Model', 'Medialness','Angle', 'Orthogonality'])
        else:
            df = pd.concat([df,pd.DataFrame([[model, med, 0, ort]],columns=['Model', 'Medialness','Angle', 'Orthogonality'])])

df.to_csv('Results/Finetune/Results_Sreps_GT.csv')