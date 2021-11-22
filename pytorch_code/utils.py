

import time
import torchvision.utils as vutils
import torch
import pytorch_fid.fid_score as fid_torch

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def calculate_fid(generator, nz, data, batch_size, cuda=True):
    if data == 'cifar10':
        fid_stats_path = './fid_stats_cifar10_train.npz'
    elif data == 'celebA':
        fid_stats_path = './fid_stats_celeba.npz'

    #Saves images to be calculated for FID - not necessary why not pass through inception first to save time
    start_t = time.time()
    generated_images_folder_path = '/home/yoni/Datasets/fid_images'
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

    # fid_score = fid.calculate_fid_given_paths(paths=[generated_images_folder_path, fid_stats_path],
    #                                                           inception_path='./inception_model/')
    fid_score = fid_torch.calculate_fid_given_paths(paths=[generated_images_folder_path, fid_stats_path], batch_size=batch_size, dims=2048, cuda=cuda)
    finish_t = time.time()
    print('The fid score is {} and was calcualted in {} (seconds)'.format(fid_score, (finish_t - start_t)))
    return fid_score