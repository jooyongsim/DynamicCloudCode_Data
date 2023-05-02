Codes in `extensions` and `utils.py` are contributed by "Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Surfaces". [Original code repository.](https://github.com/nv-tlabs/nglod)

This directory contains codes for sampling points from meshes of ShapeNet and computing ground truth signed distances, which can be used as training data of our DCC-DIF.
Before starting, you have to:
1. follow `extensions/README.md` to compile the extension modules
1. download meshes from ShapeNet (https://shapenet.org/)
2. generate watertight meshes using code from OccNet (https://github.com/autonomousvision/occupancy_networks)
3. normalize meshes into an unit cube

We expect that preprocessed meshes are placed in a directory (denoted as `source_dir`) and organized as follows:
```
path/to/meshes
├── 02691156
│   ├── 10155655850468db78d106ce0a280f87.obj
│   ├── ......
│   └── fff513f407e00e85a9ced22d91ad7027.obj
│
├── ......
│
└── 04530566
    ├── 10212c1a94915e146fc883a34ed13b89.obj
    ├── ......
    └── ffffe224db39febe288b05b36358465d.obj
```

Then, specify the `gpu_id`, `source_dir` and `target_dir` in `main.py`, and run it to get sampled points.
If everything goes well, you will find files in `target_dir` with the same organization as `source_dir`, but replaced the file suffix with `.npy`. Each `.npy` file contains a numpy array with shape (#points, 4), and each row in this array is the (x, y, z) coordinate and ground truth signed distance of a sampled point.

We provide a demo in `demo_data`.
