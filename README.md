# rot_inv_rep
Rotational - SO(2) invariant representation of images and classification 

Dependency : Requires tensorflow 1.7

The data files are in a different repository.
https://github.com/muthuvel-murugan/ds_mnist

The data files are in npz format. It has 2 keys, called data & labels.
data is a vector of dimension n_samples x 784, and labels is of dimension n_samples.

To try on your own dataset, please prepare the data in the above format.


To generate W using Autoencoder for mnist run the following:
-----------------------------------------------------------
python s2_ae_gen.py mnist_find_W_ae_tmp.json

The generated W will be in the folder: 
logs/mnist_ae_0_12000

To generate W using Coupled Autoencoder for mnist run the following:
-----------------------------------------------------------
python s2_cae_gen.py mnist_find_W_cae_tmp.json

The generated W will be in the folder: 
logs/mnist_cae_0_12000

For classification:
-------------------
Train on rotated:

python classify_gen.py mnist_28_train_rot.json


Train on upright:

python classify_gen.py mnist_28_train_org.json
