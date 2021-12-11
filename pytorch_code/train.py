from model import Generator, Discriminator, weights_init
import argparse
import random
import os
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import json


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
    fake = fake.to(device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train(opt, dataloader):
    noise = torch.randn((opt.batchSize, opt.nz, 1, 1), device=device)
    fixed_noise = torch.randn((64, opt.nz, 1, 1), device=device)
    if opt.adam:
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.99))
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.99))
    else:
        discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lrD)
        generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=opt.lrG)
    gen_iterations = 0
    criterion = torch.nn.BCELoss()
    for epoch in range(opt.n_epochs):
        if epoch % 3 == 0:
            torch.save(generator.state_dict(), '{0}/Gen.pth.tar'.format(opt.experiment))
            torch.save(discriminator.state_dict(), '{0}/Disc.pth.tar'.format(opt.experiment))
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = Variable(imgs.type(torch.Tensor))
            real_imgs = real_imgs.to(device)
            label_size = real_imgs.size(0)
            ones = torch.full((label_size,), 1.0, dtype=real_imgs.dtype, device=device)
            zeros = torch.full((label_size,), 0.0, dtype=real_imgs.dtype, device=device)

            for _ in range(opt.Diters):
                noise = torch.randn((label_size, opt.nz, 1, 1), device=device)
                fake = generator(noise).detach()
                errD_fake = discriminator(fake)
                errD_real = discriminator(real_imgs)
                if opt.dcgan:
                    errD = criterion(errD_real.view(-1), ones) + criterion(errD_fake.view(-1), zeros)
                else:
                    errD_fake = torch.mean(errD_fake)
                    errD_real = torch.mean(errD_real)
                    errD = errD_fake - errD_real
                    if opt.withGP:
                        errD += 10 * compute_gradient_penalty(discriminator, real_imgs.data, fake.data)
                discriminator.zero_grad()
                errD.backward()
                discriminator_optimizer.step()


            gen_fake = generator(noise)
            if opt.dcgan:
                criterion = torch.nn.BCELoss()
                errG = criterion(discriminator(gen_fake).view(-1), ones)
            else:
                errG = -torch.mean(discriminator(gen_fake))

            generator.zero_grad()
            errG.backward()
            generator_optimizer.step()
            gen_iterations += 1

            if not opt.dcgan:
                print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                      % (epoch, opt.n_epochs, i, len(dataloader), gen_iterations,
                         errD.item(), -1 * errG.item(), errD_real.item(), errD_fake.item()))
            else:
                print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                      % (epoch, opt.n_epochs, i, len(dataloader), gen_iterations,
                         errD.item(), torch.mean(errG).item(), torch.mean(errD_real).item()
                         , torch.mean(errD_fake).item()))
            if gen_iterations % 1000 == 0:
                real_imgs = real_imgs.mul(0.5).add(0.5)
                save_image(real_imgs, '{0}/real_samples.png'.format(opt.save_imgs))
                with torch.no_grad():
                    fake = generator(fixed_noise)
                fake.data = fake.data.mul(0.5).add(0.5)
                save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.save_imgs, gen_iterations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageSize', type=int, default=64, help='the image size after transform')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--kernel', type=int, default=4, help='kernel size of CNN, for 1 channel image use 3'
                                                              ', 3 channel use 4')
    parser.add_argument('--dataset', required=True, help='lsun | celeb | mnist | other')
    parser.add_argument('--dataroot', default=None, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--nz', type=int, default=100, help='size of the z (noise)')
    parser.add_argument('--g_feature', type=int, default=512, help='Final feature maps for G')
    parser.add_argument('--d_feature', type=int, default=64, help='Final feature maps for D')
    parser.add_argument('--n_epochs', type=int, default=25, help='training epochs')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.001, help='learning rate for Generator, default=0.001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
    parser.add_argument('--manualSeed', type=int, default=0, help='random seed')
    parser.add_argument('--clamp_lower', type=float, default=-0.01, help='upper limit of weight')
    parser.add_argument('--clamp_upper', type=float, default=0.01, help='lower limit of weight')
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--experiment', default='experiment', help='Where to store samples and models')
    parser.add_argument('--save_imgs', default='saved_imgs', help='Where to save images')
    parser.add_argument('--cpG', default='experiment', help="checkpoints of G to continue training")
    parser.add_argument('--cpD', default='experiment', help="checkpoints of D to continue training")
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--dcgan', action='store_true', help='Whether to train on dcgan (default is wgan)')
    parser.add_argument('--withGP', action='store_true', help='Whether to use gradient penalty')
    opt = parser.parse_args()

    if opt.experiment is None:
        opt.experiment = 'samples'
    os.system('mkdir {0}'.format(opt.experiment))
    if opt.manualSeed == 0:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    torch.backends.cudnn.benchmark = True

    if opt.dataset == 'other':
        # folder dataset
        dataset = datasets.ImageFolder(root=opt.dataroot,
                                       transform=transforms.Compose([
                                           transforms.Resize(opt.imageSize),
                                           transforms.CenterCrop(opt.imageSize),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
    elif opt.dataset == 'lsun':
        dataset = datasets.LSUN(root=opt.dataroot, classes=['bedroom_train'], transform=transforms.Compose([
            transforms.Resize(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    elif opt.dataset == 'celeb':
        dataset = datasets.CelebA(root=opt.dataroot, download=False,
                                  transform=transforms.Compose([
                                      transforms.Resize(opt.imageSize),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ])
                                  )
    elif opt.dataset == 'mnist':
        dataset = datasets.MNIST(root=opt.dataroot, download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(opt.imageSize),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,)),
                                 ])
                                 )
    assert dataset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    g_feature = int(opt.g_feature)
    d_feature = int(opt.d_feature)
    kernel_size = int(opt.kernel)
    nc = int(opt.nc)

    with open(opt.experiment + '/config.json', 'w+') as arg_file:
        json.dump(vars(opt), arg_file)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")

    generator = Generator(in_channel=nz, feature_map=g_feature, out_channel=nc,
                          ngpu=ngpu, kernel_size=kernel_size, image_size=opt.imageSize).to(device)
    discriminator = Discriminator(in_channel=nc, feature_map=d_feature, ngpu=ngpu,
                                  kernel_size=kernel_size, dcgan=opt.dcgan, image_size=opt.imageSize).to(device)

    if opt.cpG is not None and opt.cpD is not None:
        try:
            generator.load_state_dict(torch.load(opt.cpG + '/Gen.pth.tar'))
            print('Loaded generator from ' + opt.cpG)
        except:
            print('Failed to load generator from ', opt.cpG)
        try:
            discriminator.load_state_dict(torch.load(opt.cpD + '/Disc.pth.tar'))
            print('loaded discriminator from ' + opt.cpD)
        except:
            print('Failed to load discriminator ', opt.cpD)

    else:
        generator.apply(weights_init)
        discriminator.apply(weights_init)

    if (device.type == 'cuda') and (opt.ngpu > 1):
        generator = torch.nn.DataParallel(generator, list(range(opt.ngpu)))
        discriminator = torch.nn.DataParallel(discriminator, list(range(opt.ngpu)))

    print(generator, discriminator)

    train(opt, dataloader)
