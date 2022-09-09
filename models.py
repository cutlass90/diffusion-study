import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self, diffusor, b1=1e-4, b2=0.02, T=1000, criterion=F.mse_loss):
        super(DDPM, self).__init__()
        self.diffusor = diffusor
        self.T = T
        self.criterion = criterion
        coeff = get_diffusion_coefficients(b1, b2, T)
        self.a_ = torch.from_numpy(coeff['a_']).float()
        self.a = torch.from_numpy(coeff['a']).float()

    def forward(self, x0):
        t = torch.randint(self.T, (x0.size(0),)).to(x0.device)
        eps = torch.randn_like(x0)
        pred_eps = self.diffusor(torch.sqrt(self.a_[t][:, None, None, None]).to(x0.device) * x0 +
                                 torch.sqrt(1 - self.a_[t][:, None, None, None]).to(x0.device) * eps, t.unsqueeze(1).float()/self.T)
        loss = self.criterion(eps, pred_eps)
        return loss

    @torch.no_grad()
    def sample(self, n_samples, data_size):
        xT = torch.randn([n_samples, *data_size]).to(list(self.diffusor.state_dict().values())[-1].device)
        for t in tqdm(range(self.T, 0, -1)):
            z = torch.randn_like(xT) if t > 1 else 0
            k = ((1 - self.a[t-1])/math.sqrt(1 - self.a_[t-1])).to(xT.device)
            t_tensor = t*torch.ones((n_samples, 1)).to(xT.device)/self.T
            sigma = torch.sqrt(1 - self.a[t-1]).to(xT.device)
            xT = 1/math.sqrt(self.a[t-1]) * (xT - k*self.diffusor(xT, t_tensor)) + z * sigma
        return xT



def get_diffusion_coefficients(b1=1e-4, b2=0.02, T=1000):
    b = np.linspace(b1, b2, T)
    a = 1 - b
    a_ = np.cumprod(a)
    return {'a_':a_, 'a':a}

if __name__ == "__main__":
    get_diffusion_coefficients()

    from networks import SimpleAE
    ae = SimpleAE(1, 32)
    img = torch.randn(2, 1, 28, 28)
    t = torch.randn(2)
    ddpm = DDPM(ae)
    ddpm(img)
    result = ddpm.sample(4, (1, 28, 28))
    print()


