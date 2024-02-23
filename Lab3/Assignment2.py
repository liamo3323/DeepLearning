# Torchvision
import sys
import matplotlib
import matplotlib.pyplot as plt
import torch
from mae import models_mae
import pandas as pd 
import sys
import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0

# Read the csv file
csv_path = "6640106/shuffle_info.csv"
read_csv_path = pd.read_csv(csv_path, header=None)

# Add the path to the sys.path
sys.path.append("./Lab3/mae")
chkpt_dir = 'mae_visualize_vit_large.pth'

# Load the ids from the csv file 
ids_restore = torch.Tensor(eval(read_csv_path.loc[1][1])).type(torch.int64)
ids_keep = torch.Tensor(eval(read_csv_path.loc[0][1])).type(torch.int64)

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
        # build model
        model = getattr(models_mae, arch)()
        # load model
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        return model
    
def masking(x):
    N, L, D = x.shape  # batch, length, dim
    return torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

def forward_encoder(self, x):
    # embed patches
    x = self.patch_embed(x)

    # add pos embed w/o cls token
    x = x + self.pos_embed[:, 1:, :]

    x = masking(x)

    # append cls token
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # apply Transformer blocks
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)

    return x

def restoring_image(img, model):
    img = Image.open(img).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    img = img - imagenet_mean
    img = img / imagenet_std


    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    temp_x = forward_encoder(model, x.float())

    y = model.forward_decoder(temp_x, ids_restore)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    plt.rcParams['figure.figsize'] = [24, 24]

    # show the original and reconstructed images
    plt.subplot(1, 2, 1)
    show_image(x[0], "original")

    plt.subplot(1, 2, 2)
    show_image(y[0], "reconstructed")

    plt.show()

def show_image(image, title=''):

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return
    
def main():
    model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    restoring_image("6640106.png", model_mae)
        
if __name__ == "__main__": 
    main()