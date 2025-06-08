from datasets import DataLoader
from model import Model
from utils import DWT, IWT, gauss_noise, random_data, computePSNR
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm

def accuracy(output, target):
    return (output > 0.0).eq(target>=0.5).sum().float() / target.numel()


epoch = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = DataLoader('../steg/div2k/train/_')
val_dataset = DataLoader('../steg/div2k/val/_')

model = Model()
params_trainable = (list(filter(lambda p: p.requires_grad, model.parameters())))

optim = torch.optim.Adam(params_trainable, lr=10 ** (-4.5), betas=(0.85,0.999), eps=1e-6, weight_decay=1e-5)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, 5, gamma=0.78)

dwt = DWT()
iwt = IWT()

for i_epoch in range(epoch):
    print('Epoch: ', i_epoch)
    for data in tqdm(train_dataset):
        data = data.to(device)
        secret = random_data(data, device)
        data_input = dwt(data)
        secret_input = dwt(secret)
        
        input_img = torch.cat([data_input, secret_input], dim=1)
        
        output = model(input_img)
        output_steg = output.narrow(1, 0, 4 * 3)
        output_z = output.narrow(1, 4 * 3, output.shape[1] - 4 * 3)
        output_img = iwt(output_steg)
        
        output_z_guass = gauss_noise(output_z.shape)
        
        output_rev = torch.cat((output_steg, output_z_guass), 1)
        output_image = model(output_rev, rev=True)

        secret_rev = output_image.narrow(1, 4 * 3, output_image.shape[1] - 4 * 3)
        secret_rev = iwt(secret_rev)
        
        g_loss = F.mse_loss(output_img, data,reduction='sum')
        r_loss = F.binary_cross_entropy_with_logits(secret_rev, secret,reduction='sum')
        d_acc = accuracy(secret_rev, secret)        
        
        steg_low = output_steg.narrow(1, 0, 3)
        cover_low = data_input.narrow(1, 0, 3)
        
        l_loss = F.mse_loss(steg_low, cover_low,reduction='sum')

        total_loss = 4 * r_loss + 1.8 * g_loss + 1 * l_loss
        optim.zero_grad()
        total_loss.backward()
        optim.step()
        
    print(f'train loss: {total_loss.item()}')
    print(f'reconstruction loss: {r_loss.item()}')
    print(f'guide loss: {g_loss.item()}')
    print(f'low frequency loss: {l_loss.item()}')
    print(f'decoder accuracy: {d_acc.item()}')
    
    if i_epoch % 5 == 0:
        with torch.no_grad():
            psnr_s = []
            psnr_c = []
            model.eval()
            for data in tqdm(val_dataset):
                data = data.to(device)
                secret = random_data(data, device)
                data_input = dwt(data)
                secret_input = dwt(secret)

                input_img = torch.cat([data_input, secret_input], dim=1)

                output = model(input_img)
                output_steg = output.narrow(1, 0, 4 * 3)
                steg = iwt(output_steg)
                output_z = output.narrow(1, 4 * 3, output.shape[1] - 4 * 3)
                output_z_guass = gauss_noise(output_z.shape)
                
                output_steg = output_steg.cuda()
                output_rev = torch.cat((output_steg, output_z_guass), 1)
                output_image = model(output_rev, rev=True)
                secret_rev = output_image.narrow(1, 4 * 3, output_image.shape[1] - 4 * 3)
                secret_rev = iwt(secret_rev)

                data = data.cpu().numpy().squeeze() * 255
                np.clip(data, 0, 255)
                steg = steg.cpu().numpy().squeeze() * 255
                np.clip(steg, 0, 255)
                psnr_temp_c = computePSNR(data, steg)
                psnr_c.append(psnr_temp_c)
            print(f'PSNR cover: {np.mean(psnr_c)}')
        if i_epoch > 0 and (i_epoch % 25) == 0:
            torch.save({'opt': optim.state_dict(),
                'net': model.state_dict()},'./models/'+ 'model_checkpoint_4_1.8_1_%.5i' % i_epoch + '.pt')
    weight_scheduler.step()
    current_lr = optim.param_groups[0]['lr']
    print(f"Epoch {i_epoch+1}/{epoch}, Current Learning Rate: {current_lr}")

    torch.save({'opt': optim.state_dict(),
                'net': model.state_dict()},'./models/' + 'model_4_1.8_1' + '.pt')