###README

This is the repo for Umich EECS 545 final project: **Loss Function Matters: Probing into the Reason behind Wasserstein GAN’s Success**

Contributors: Zixuan Pan, Yuqi Xie, Zhiyuan Chen

In this project,we implemented [DCGAN](https://arxiv.org/abs/1511.06434), [WGAN](https://arxiv.org/abs/1701.07875) and [WGANGP](https://arxiv.org/abs/1704.00028). 

### Repo Contents 

Toy Model: a jupyter notebook with simple example with MNIST

Pytorch code: main WGAN code implemented with pytorch

### Usage

####Toy model

```shell
mkdir data
mkdir experiments
```

Before running the notebook

####Pytorch code

**Quick Start**

```shell
mkdir experiments
mkdir saved_imgs
mkdir data

python3 train.py --dataset mnist dataroot data --kernel 4 --nc 1 > experiment/log.txt
```

**Some important arguments**

````shell
--cuda #Use gpu for training
--ngpu <int> #Number of GPUs for training
--dataset <mnist|celeb|lsun|other> #There's no need to download manually mnist dataset. For the rest, please download manually and adjust to the torchvision format.
--dataroot <str> #Root of where you put the data
--nc <int> #Number of image channels. 1 for mnist (only black and white), 3 for colored datasets.
--dcgan #Add when you'd like to train with DCGAN model
--withGP #Add when you want to have gradient penalty in wgan loss
````

Check other arguments in train,py

##### **Evaluation**

Use the function in utils.py

````python
###Calculate FID score of images
def calculate_fid(generator, real_dir, fake_dir, cuda=True, nz=100, batch_size = 64):
  #generator: generator model, load from your generator file
  #real_dir: location of real samples (only images)
  #out_dir: location where you want to save generated images
  #cuda, nz and batch_size set according to your training arguments

###Plot loss function  
def plot_loss(infile):
  #infile: the file of loss data you obtained in the 		   experiment
````

**Reproduction**

Use the following random seeds to reproduce our results.

LSUN bedroom: 7840

Animefaces-danbooru: 5635

CelebA: 3303