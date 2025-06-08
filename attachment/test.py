import torch
from model import Model
from utils import DWT, IWT, make_payload, auxiliary_variable, bits_to_bytearray, bytearray_to_text
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
    # torchvision.utils.save_image(cover, f'./{text}.png')
    torchvision.utils.save_image(output_img,f'./steg.png')


def decode(steg):
    secret_rev = your_decode_net(steg)

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
    load('magic.potions')
    d3net.eval()

    dwt = DWT()
    iwt = IWT()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    text = r'd3ctf{Getting that model to converge felt like pure sorcery}'
    steg = r'./steg.png'
    cover = './poster.png'
    encode(cover, text)
    # decode(steg)
    