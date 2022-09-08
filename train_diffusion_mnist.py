from networks import SimpleAE
from configs.mnist_v1 import opt

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from models import simple_linear_sceduler
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
import os


def main():
    print(opt)
    os.makedirs(os.path.join(opt.checkpoint_dir, 'tf_logs'), exist_ok=True)
    os.makedirs(os.path.join(opt.checkpoint_dir, 'weights'), exist_ok=True)
    writer = SummaryWriter(os.path.join(opt.checkpoint_dir, 'tf_logs'))
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0))]
    )
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)

    diffusor = SimpleAE(in_channels=1, filters=opt.filters).to(opt.device)
    if opt.load_path:
        diffusor.load_state_dict(torch.load(opt.load_path))
        print(f'weights were loaded', opt.load_path)
    optim = torch.optim.Adam(diffusor.parameters(), lr=opt.lr)
    step = 0
    for i in range(opt.n_epoch):
        diffusor.train()
        for _, img in tqdm(enumerate(dataloader)):
            img = img[0].to(opt.device)
            T = torch.randint(0, opt.diffusion_steps, (opt.batch_size, 1)).to(opt.device).float()
            k = simple_linear_sceduler(T, opt.diffusion_steps)
            noise = torch.randn(img.size()).to(opt.device)
            pred_noise = diffusor(k[:, :, None, None]*img + (1-k)[:, :, None, None]*noise, T)
            loss = nn.MSELoss()(noise, pred_noise)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % opt.log_freq == 0:
                writer.add_scalar('mse loss', loss.item(), step)
                print(loss.item())
            if (step + 1) % opt.save_freq == 0:
                torch.save(diffusor.state_dict(), os.path.join(opt.checkpoint_dir, 'weights', 'latest.pth'))

            step += 1









if __name__ == "__main__":
    main()
