import torch

from diffusion import Diffusion
from unet import Unet

from utils import get_loaders, make_default_dirs


if __name__ == '__main__':
    make_default_dirs()

    device = 'cuda'

    config = {
        'timesteps': 1000,
        'K': 16,
        'coeff': 0.001,
        'bs': 64,
        'lr': 2e-4,
        'epochs': 200,
        'j': 2,
    }

    diffusion = Diffusion(config, device)
    model = Unet(
        K = config['K'],
        in_ch = 1,
        ch = 128,
        out_ch = 1,
        att_channels = [0, 1, 0, 0],
        groups = 32
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])

    train_loader, eval_loader = get_loaders(config)

    def sample_time(bs=config['bs'], T=config['timesteps']):
        return torch.randint(0, T, (bs,), device=device)

    step = 0
    for epoch in range(config['epochs']):
        model.train()
        for x, _ in train_loader:
            t = sample_time()
            x = x.to(device)
            x = (x * config['K']).long().clamp(0, config['K'] - 1)
            noise = torch.rand((*x.shape, config['K']), device=device)

            xt = diffusion.q_sample(x, t, noise)
            loss, _ = diffusion.compute_losses(model, x, xt, t)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            if (step + 1) % config['print_freq'] == 0:
                print(f'Step: {step} ({epoch}) | Loss: {loss.item():.5f}')
            
            step += 1