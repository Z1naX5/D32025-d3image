import torch
from model import Model
from utils import DWT, IWT, make_payload, gauss_noise, bits_to_bytearray, bytearray_to_text
from datasets import DataLoader
import torchvision
from collections import Counter
from PIL import Image
import torchvision.transforms as T

transform_test = T.Compose([
    T.CenterCrop((720,1280)),
    T.ToTensor(),
])

def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    d3net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')
        


def transform2tensor(img):
    img = Image.open(img)
    img = img.convert('RGB')
    return transform_test(img).unsqueeze(0).to(device)

def encode(cover, text):
    cover = transform2tensor(cover)
    B, C, H, W = cover.size()       
    payload = make_payload(W, H, C, text, B)
    payload = payload.to(device)
    cover_input = dwt(cover)
    payload_input = dwt(payload)        

    input_img = torch.cat([cover_input, payload_input], dim=1)

    output = d3net(input_img)
    output_steg = output.narrow(1, 0, 4 * 3)
    output_img = iwt(output_steg)

    torchvision.utils.save_image(cover, f'./images/cover/_{text}.png')
    torchvision.utils.save_image(output_img,f'./images/steg/_{text}.png')


def decode(steg):
    output_steg = transform2tensor(steg)
    output_steg = dwt(output_steg)
    backward_z = gauss_noise(output_steg.shape)

    output_rev = torch.cat((output_steg, backward_z), 1)
    bacward_img = d3net(output_rev, rev=True)
    secret_rev = bacward_img.narrow(1, 4 * 3, bacward_img.shape[1] - 4 * 3)
    secret_rev = iwt(secret_rev)

    image = secret_rev.view(-1) > 0

    candidates = Counter()
    bits = image.data.int().cpu().numpy().tolist()
    for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
        candidate = bytearray_to_text(bytearray(candidate))
        if candidate:
            candidates[candidate] += 1
    if len(candidates) == 0:
        raise ValueError('Failed to find message.')
    candidate, count = candidates.most_common(1)[0]
    print(candidate)

        
if __name__ == '__main__':
    d3net = Model()
    params_trainable = (list(filter(lambda p: p.requires_grad, d3net.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=10 ** (-4.5), betas=(0.85,0.999), eps=1e-6, weight_decay=1e-5)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, 5, gamma=0.78)
    load('/home/vidar/m1aoo0bin/d3image/models/model_checkpoint_4_1.8_1_00150.pt')
    d3net.eval()

    dwt = DWT()
    iwt = IWT()
    text = r'd3ctf{aaaaaaaaaaaaaaaaaaaaa}'
    cover = './poster.png'

    steg = r'/home/vidar/m1aoo0bin/d3image/mysterious_invitation.png'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # encode(cover, text)
    decode(steg)
    