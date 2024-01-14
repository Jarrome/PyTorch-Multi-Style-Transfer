import os
import sys
sys.path.append(os.path.dirname(__file__))


import cv2
import numpy as np
import torch
from torch.autograd import Variable

from net import Net
from option import Options
import utils_
#from utils import StyleLoader
import pdb

sys.path.pop()

def get_api(style_idx=1):
    style_model = Net(ngf=128)#args.ngf)
    model_dict = torch.load( os.path.dirname(__file__) + '/models/21styles.model')#args.model)
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    style_model.eval()
    if True:
        style_loader = utils_.StyleLoader(os.path.dirname(__file__) +'/images/21styles/', 512)
                #args.style_folder, args.style_size)
        style_model.cuda(0)
    else:
        style_loader = StyleLoader(args.style_folder, args.style_size, False)

    style_v = style_loader.get(int(style_idx))
    style_v = Variable(style_v.data).cuda(0)
    style_model.setTarget(style_v)


    def style_api(img):
        cuda=True

        img = np.array(img).transpose(2, 0, 1)
        # changing style 
        img=torch.from_numpy(img).unsqueeze(0).float()
        if cuda:
                img=img.cuda(0)
        img = Variable(img)
        img = style_model(img)

        if cuda:
                simg = style_v.cpu().data[0].numpy()
                img = img.cpu().clamp(0, 255).data[0].numpy()
        else:
                simg = style_v.data.numpy()
                img = img.clamp(0, 255).data[0].numpy()
        simg = np.squeeze(simg)
        img = img.transpose(1, 2, 0).astype('uint8')
        simg = simg.transpose(1, 2, 0).astype('uint8')

        return img
    return style_api
