
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

# creates a transformation, return a function that applies to 3d coordinates
def make_rotation(ax, ay, az):
    r_matrix = np.asarray([[1, 0, 0], [0, np.cos(ax), np.sin(ax)], [0, -np.sin(ax), np.cos(ax)]]) # x axis
    r_matrix = np.matmul(r_matrix, np.asarray([[np.cos(ay), -np.sin(ay), 0], [np.sin(ay), np.cos(ay), 0], [0, 0, 1]])) # y axis
    r_matrix = np.matmul(r_matrix, np.asarray([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]])) # z axis
    return lambda coords: np.matmul(r_matrix, coords)

"""
Nsamples - the number of points generated
translation - [x, y, z]
Ngrid - the number of possible values along a single axis
 """
def genSphere(Nsamples, radius=1, translation=[0, 0, 0], Ngrid=999):

    parametrizej = lambda j: j*np.pi/2/Ngrid
    jacobian = lambda i, j: np.sqrt(np.abs(1**4*np.sin(parametrizej(j))**2))
    probGrid = np.fromfunction(jacobian, (Ngrid, Ngrid))
    probGrid_normalized = probGrid/np.sum(probGrid)

    # Create a flat copy of the array
    flat = probGrid_normalized.flatten()

    sample_index = np.random.choice(a=flat.size, p=flat, size=Nsamples)
    adjusted_index = np.zeros((Nsamples, 2))
    for i in range(Nsamples):
        adjusted_index[i] = np.unravel_index(sample_index[i], probGrid_normalized.shape)
    adjusted_index = adjusted_index.transpose()

    Azi = 0+adjusted_index[0]*2*np.pi/Ngrid  # 0 to 2pi
    Pol = 0+adjusted_index[1]*np.pi/2/Ngrid  # 0 to pi/2 for the upper hemisphere
    UpOrDown = np.random.choice([-1, 1], Nsamples)
    X = radius*np.sin(Pol)*np.cos(Azi)
    Y = radius*np.sin(Pol)*np.sin(Azi)
    Z = radius*np.cos(Pol) * UpOrDown

    pts = np.dstack([X, Y, Z])[0] + translation
    return pts

def genTorus(Nsamples, R1, R2, translation=[0, 0, 0], rotation=[0, 0, 0], Ngrid=999):
    
    parametrizei = lambda i: i*2*np.pi/Ngrid
    jacobian = lambda i, j: np.sqrt(np.abs((1+0.1*np.cos(parametrizei(i)))**2*0.1**2))
    probGrid_torus = np.fromfunction(jacobian, (Ngrid, Ngrid))
    probGrid_torus_normalized = probGrid_torus/np.sum(probGrid_torus)

    flat_torus = probGrid_torus_normalized.flatten()

    sample_index_torus = np.random.choice(a=flat_torus.size, p=flat_torus, size=Nsamples)
    adjusted_index_torus = np.zeros((Nsamples, 2))
    for i in range(Nsamples):
        adjusted_index_torus[i] = np.unravel_index(sample_index_torus[i], probGrid_torus_normalized.shape)
    adjusted_index_torus = adjusted_index_torus.transpose()

    V = 0+adjusted_index_torus[0]*2*np.pi/Ngrid
    U = 0+adjusted_index_torus[1]*2*np.pi/Ngrid
    X = (R1+R2*np.cos(V))*np.cos(U)
    Y = (R1+R2*np.cos(V))*np.sin(U)
    Z = 0.1*np.sin(V)

    rotate = make_rotation(rotation[0], rotation[1], rotation[2])
    pts_torus = np.dstack([X, Y, Z])
    pts_torus = np.apply_along_axis(rotate, 2, pts_torus)
    pts_torus = pts_torus[0] + translation

    return pts_torus

def sphereInSphereInSphere(Nsamples, r=1, center=[0, 0, 0]):
    pts1 = np.concatenate((genSphere(int(Nsamples/4), r, center), genSphere(int(Nsamples/4), r*1/3, center)), axis=0)
    pts2 = genSphere(int(Nsamples/2), r*2/3, center)
    return (pts1, pts2)

def interlockingTori(Nsamples):
    pts_torus1 = genTorus(int(Nsamples/2), 0.5, 0.2)
    pts_torus2 = genTorus(int(Nsamples/2), 0.5, 0.2, translation=[-0.5, 0, 0], rotation=[np.pi/2, 0, 0]) 
    return (pts_torus1, pts_torus2)

# i is the number of circles in the outer loop
def spheresInSphere2D(Nsamples, i):
    r = 1/50 # radius of small circle
    R = 3/5 # distance small circles are from center
    
    pts1 = []
    pts2 = []

    idx = 0

    # 8 small circles
    while idx < Nsamples/2:
        p = np.random.rand(2)*2-1

        # outer circles
        for c in range(i):
            center = np.array([np.cos(c*2*np.pi/i), np.sin(c*2*np.pi/i)]) * R # convert polar coordinates to cartesian to find the center of the small circles
            if (center[0] - p[0])**2 + (center[1] - p[1])**2 < r:
                pts1.append(p)
                idx += 1
                break
        # middle circle
        if p[0]**2 + p[1]**2 < r:
            pts1.append(p)
            idx += 1
    
    while idx < Nsamples:

        p = np.random.rand(2)*2-1

        outTake = False

        if p[0]**2 + p[1]**2 > 1:
            continue

        # outer circles
        for c in range(i):
            center = np.array([np.cos(c*2*np.pi/i), np.sin(c*2*np.pi/i)]) * R # convert polar coordinates to cartesian to find the center of the small circles
            if (center[0] - p[0])**2 + (center[1] - p[1])**2 < 2*r:
                outTake = True
                break
        # middle circle
        if p[0]**2 + p[1]**2 < 2*r:
            outTake = True
        
        if outTake == False:
            pts2.append(p)
            idx += 1
    
    return np.array(pts1), np.array(pts2)

def spheresInSpheresInSpheres3D(Nsamples):
    pts1 = np.array([]).reshape(0, 3)
    pts2 = np.array([]).reshape(0, 3)
    for x in (0.25, 0.75):
        for y in (0.25, 0.75):
            for z in (0.25, 0.75):
                points = sphereInSphereInSphere(int(Nsamples/8), 1/6, [x, y, z])
                pts1 = np.vstack((pts1, np.array(points[0])))
                pts2 = np.vstack((pts2, np.array(points[1])))
    return (pts1, pts2)

def plot3d(pts1, pts2):
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(pts1[:,0], pts1[:,1], pts1[:,2], s=0.05, color="blue")
    ax.scatter(pts2[:,0], pts2[:,1], pts2[:,2], s=0.05, color="red")
    ax.view_init(elev=0, azim=-35, roll=0)
    ax.axis("equal")
    plt.ion()
    plt.show(block=True)

def plot2d(pts1, pts2):
    ax = plt.figure().add_subplot()
    ax.scatter(pts1[:,0], pts1[:,1], s=0.05, color="blue")
    ax.scatter(pts2[:,0], pts2[:,1], s=0.05, color="red")
    ax.axis("equal")
    plt.ion()
    plt.show(block=True)
    

class TopologicalDataset(Dataset):

    def __init__(self, pts):
        self.data = np.concatenate((torch.tensor(pts[0], dtype=torch.float), torch.tensor(pts[1], dtype=torch.float)))
        self.labels = np.concatenate((np.zeros(len(pts[0])), np.ones(len(pts[1]))))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]