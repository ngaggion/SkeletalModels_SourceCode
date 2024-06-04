import numpy as np
import scipy.sparse as sp
import pickle as pkl

def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v),len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv

faces = np.load('combi.npy')
points = np.load("sreps_processed/groupA_01_hippo_pp_surf_SPHARM-srep_points.npy")

A = get_vert_connectivity(points, faces).tocoo()

pkl.dump(A, open("adj_combi.pkl", "wb"))