gpu_id = '5'
source_dir = 'demo_data/input'
target_dir = 'demo_data/output'

################################################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
from datetime import datetime
import numpy as np
import torch
from utils import *

all_classes = [
    #"04256520",
    "02691156",
    #"03636649",
    #"04401088",
    "04530566",
    #"03691459",
    #"03001627",
    #"02933112",
    #"04379243",
    #"03211117",
    #"02958343",
    #"02828884",
    #"04090263"
]

num_samples_and_method = [(100000, 'uniformly'), (100000, 'near')]

for c in all_classes:
    input_dir = os.path.join(source_dir, c)
    output_dir = os.path.join(target_dir, c)
    os.makedirs(output_dir, exist_ok=True)
    all_shapes = os.listdir(input_dir)
    all_shapes = [f.split('.')[0] for f in all_shapes]
    for i, shape_id in enumerate(all_shapes):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), c, 'processing: %d/%d'%(i,len(all_shapes)))
        in_path = os.path.join(input_dir, shape_id+'.obj')
        out_path = os.path.join(output_dir, shape_id+'.npy')

        vertices, faces = load_obj(in_path)
        mesh = obj2nvc(vertices, faces).cuda()
        mesh_normals = face_normals(mesh)
        distrib = area_weighted_distribution(mesh, mesh_normals)

        xyz = sample_points(mesh, num_samples_and_method, mesh_normals, distrib)
        sd = points_mesh_signed_distance(xyz, mesh)
        xyz_sd = torch.cat([xyz, sd.unsqueeze(1)], dim=1)
        rand_idx = torch.randperm(xyz_sd.shape[0])
        xyz_sd = xyz_sd[rand_idx].cpu().numpy()
        np.save(out_path, xyz_sd)

