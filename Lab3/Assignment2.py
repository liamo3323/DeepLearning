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
import math
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

orig_img = Image.open("6640106.png").resize((224,224))

# Load the ids from the csv file 
ids_restore = torch.Tensor(eval(read_csv_path.loc[1][1])).type(torch.int64) # ids_restore are the squares that are removed
ids_keep = torch.Tensor(eval(read_csv_path.loc[0][1])).type(torch.int64) # ids_keep are the squares that are kept

# normalize by ImageNet mean and std
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
        # build model 
        model = getattr(models_mae, arch)()
        # load model from checkpoint
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        return model
    
def masking(x):
    # taken and edited  from model_MAE code
    N, L, D = x.shape  # batch, length, dim 
    return torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # mask the squares that are removed

def forward_encoder(self, x):
    # embed patches
    x = self.patch_embed(x) 

    # add pos embed w/o cls token
    x = x + self.pos_embed[:, 1:, :]

    x = masking(x) # masking the patches 

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

    # load image
    img = Image.open(img).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    img = img - imagenet_mean
    img = img / imagenet_std


    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # forward pass through the model
    temp_x = forward_encoder(model, x.float())

    # forward pass through the decoder restoring the squaress that were removed
    y = model.forward_decoder(temp_x, ids_restore)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    plt.rcParams['figure.figsize'] = [24, 24]

    # shows the original and reconstructed images
    plt.subplot(1, 2, 1)
    show_image(x[0], "original")
    # plt.savefig('original.png', bbox_inches='tight')

    plt.subplot(1, 2, 2)
    show_image(y[0], "reconstructed")
    # plt.savefig('reconstricted.png', bbox_inches='tight')

    plt.show()
    return y[0]

def show_image(image, title=''):

    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def replace_pixels(square_idx, grid_size, square_size, restored_img):
    row = math.floor(square_idx / grid_size)
    col = square_idx % grid_size
    for width_pixel in range(square_size*col, square_size*(col+1)):
        for height_pixel in range(square_size*row, square_size*(row+1)):
            replace_val = orig_img.getpixel((width_pixel, height_pixel))
            restored_img.putpixel((width_pixel, height_pixel), replace_val)
    
def main():
    model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16') # Load the model from the checkpoint link
    shuffle_data = pd.read_csv("6640106/shuffle_info.csv", header=None)

    restored_image = np.array(restoring_image("6640106.png", model_mae)) # Restore the image 
    restored_image *= imagenet_std
    restored_image += imagenet_mean
    restored_image *= 255
    restored_image = np.uint8(restored_image)

    replace_img = Image.fromarray(restored_image, 'RGB')
    width, height = replace_img.size
    pixel_count = (int(width/14))

    for square in eval(shuffle_data.loc[0][1])[0]:
        replace_pixels(square, 14, pixel_count, replace_img) 

    plt.imshow(replace_img)
    replace_img.save("6640106_reconstructed.png")

if __name__ == "__main__": 
    # calling main function
    main()