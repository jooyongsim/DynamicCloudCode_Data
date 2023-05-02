# Install C++/CUDA extensions
for ext in mesh2sdf_cuda; do
    cd $ext && python setup.py clean --all install --user && cd -
done
