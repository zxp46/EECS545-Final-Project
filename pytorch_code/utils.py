import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import torchvision.utils as vutils
import torch
import pytorch_fid.fid_score as fid_torch

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def calculate_fid(generator, real_dir, fake_dir, cuda=True, nz=100, batch_size = 64):
    fid_stats_path = real_dir

    start_t = time.time()
    generated_images_folder_path = fake_dir
    number_fid = 5000//batch_size
    for idx in range(0, number_fid):
        z_fid = torch.randn(batch_size, nz, 1, 1, device=device)
        g_z_fid = generator(z_fid)
        for idx_fid in range(0, batch_size):
            vutils.save_image(tensor=g_z_fid[idx_fid],
                                         fp=generated_images_folder_path + '/' + 'img' + str(
                                             idx * batch_size + idx_fid) + '.png',
                                         nrow=1,
                                         padding=0)

    fid_score = fid_torch.calculate_fid_given_paths(paths=[generated_images_folder_path, fid_stats_path], batch_size=batch_size, dims=2048, cuda=cuda)
    finish_t = time.time()
    print('The fid score is {} and was calcualted in {} (seconds)'.format(fid_score, (finish_t - start_t)))
    return fid_score



def plot_loss(infile):
    nrows = 170000
    data = np.array(pd.read_table(infile, delimiter=' ', index_col=0, nrows=nrows))
    n = data.shape[0]
    per_iter = 1000
    x = range(0,n)
    loss_d_val = np.abs(data[:,1])
    loss_d_real = data[:,5]
    loss_d_fake = data[:,7]
    loss_d_val = (loss_d_val-loss_d_val.mean())/loss_d_val.std()
    i=0
    loss_d = []
    while (i+1 < nrows / per_iter):
        mean = np.mean(loss_d_val[i*per_iter:(i+1)*per_iter])
        loss_d.append(mean)
        i+=1
    x = range(0,i)
    plt.plot(x, loss_d)
    plt.title="loss_d"
    plt.xlabel("epoch")
    plt.ylabel("loss_D mean value")
    plt.show()

