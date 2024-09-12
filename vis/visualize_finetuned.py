

import nibabel as nib
import numpy as np
import os
import sys
sys.path.append('..')
import models_vim
import models_vit
from utils.brats_160 import Brats160Data
import torch
import torchio as tio
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *
import timm.models.vision_transformer

input_activations = {}
def get_input(name):
        def hook(model, input, output):
            input_activations[name] = output
        return hook


def viz_one_patch(model, x, log_dir = "", device = 'cuda'):
    # set hooker
    os.makedirs(log_dir, exist_ok=True)
    
    model.blocks[-1].register_forward_hook(get_input('head'))
    model = model.to(device)
    
    
    
    
    # save_nifty_img(x, os.path.join(log_dir, 'original_image'))
    x.unsqueeze_(0)  # Adding the batch dimension
    # make it a batch-like
    x = x.to(device)
    model = model.to(device)
    if isinstance(model, timm.models.vision_transformer.VisionTransformer):
        y = model(x.float())
        latent = input_activations['head']
        # remove cls token
        latent = latent[:, 1: , :]
    else:
        
        y = model(x.float(), return_features = True)
    
        latent, _ = input_activations['head']
    latent = latent.detach()
    
    # print('latent', latent.shape)
    if isinstance(model, timm.models.vision_transformer.VisionTransformer):
        pca = PCA(n_components=1)
        latent_np = latent.squeeze(0).cpu().numpy()
        latent_np_reduced = pca.fit_transform(latent_np)
        print("latent reduced shape", latent_np_reduced.shape)
        latent = torch.from_numpy(latent_np_reduced).unsqueeze(0)
        latent = latent.squeeze(-1)
    else:
        latent = torch.max(latent, dim = -1).values
    
    
    
    print('latent', latent.shape)
    latent = latent.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 3 * 1)
    print('latent shape', latent.shape)
    latent = model.unpatchify_3d(latent)
    print('after unpatchify_3d', latent.shape)
    latent = latent.cpu()
    latent = latent.squeeze(0).squeeze(0)

    np.save(os.path.join(log_dir, 'latent_space_before_prediction.npy'), latent.numpy())
    print("x shape", x.squeeze(0).shape)
    
   
    
    

def save_npy(img, file_name):
    np.save(file_name, img)

def save_nifty_img(image, file_name):
    # We need to scale the values so that they lie in [0, 1]
    img = (image.numpy()[0] + 1) / 2 # original [-1, 1], now [0, 1]
    img_scaled = 255 * img
    img_scaled = img_scaled[0]  # (1, 1, 96, 96, 96) -> We take the first element form the batch and the values of its only channel.
    img_pasted = nib.Nifti1Image(img_scaled.astype(np.int8), np.eye(4))
    print(f"Saving file {file_name}")
    nib.save(img_pasted, file_name)

def load_model(model_name, input_size,  num_channels, num_classes, resume  = None, drop_path = 0.1):
    if "vim" in model_name :
        
        model = models_vim.__dict__[model_name](
            img_size = input_size,
            num_classes = num_classes, 
            drop_path_rate= drop_path,
            channels = num_channels,
               
                                    )
    else:
        model = models_vit.__dict__[model_name](
            img_size = input_size,
        num_classes=num_classes,
        drop_path_rate=drop_path,
        global_pool= 'mean',
        in_chans = num_channels
        )
    if resume:
        checkpoint = torch.load(resume, map_location= 'cpu')
        model.load_state_dict(checkpoint['model'], strict= True)
        print('Model loaded from', resume)
    # print(model)
    return model
if __name__ == "__main__":
    
    image_num =13
        
    base_path = None
    seed = 42
    use_z_score = True
    
    
    input_size = 160
    num_channels = 4 
    drop_path = 0.1
    label_type = 'IDH'
    device = 'cuda'
    
    
    patch = 16
    num_classes = 2
    
    
    whole_dataset = Brats160Data( 
                                    transform= None,
                                    mode = 'test',
                                    use_z_score=use_z_score,
                                    pre_load=False,
                                    label_type = label_type
                                    )
    length = len(whole_dataset)
    train_size= int(0.8 * length)
    val_size = length - train_size 
    generator1 = torch.Generator().manual_seed(seed)
    dataset_train, dataset_val  = torch.utils.data.random_split(whole_dataset, [train_size, val_size], generator=generator1)
    x, target = dataset_val[image_num]
    print("target is ", target)
    
    model_name = f'vit_3D_small_patch{patch}'
    resume = f"../vit_small_brats_160_patch16_cr_{label_type.lower()}/checkpoint-_fold_3_epoch_99.pth"
    log_path = f"visualization/vit_small_brats_160_p{patch}_finetuned_image{image_num}_{label_type.lower()}_max_label{int(target)}"
    
    # data_list = [data[1] for data in dataset_val]
    # print(data_list)
    os.makedirs(log_path, exist_ok=True)
    
    model = load_model(model_name, input_size, num_channels, num_classes, resume= resume)
    
    np.save(os.path.join('visualization', f"input_original_image{image_num}_{label_type.lower()}.npy"), x.cpu().numpy())
    viz_one_patch(model, x, device = device, log_dir=log_path)

    for patch in [4, 16]:

        model_name = f'vim_3D_small_patch{patch}_stride{patch}_224_bimambav2_final_pool_mean_abs_pos_embed_div2'
        resume = f"../vim_small_brats_160_patch{patch}_cr_{label_type.lower()}/checkpoint-_fold_3_epoch_99.pth"
        
        log_path = f"visualization/vim_small_brats_160_p{patch}_finetuned_image{image_num}_{label_type.lower()}_max_label{int(target)}"

        model = load_model(model_name, input_size, num_channels, num_classes, resume= resume)
        x, target = dataset_val[image_num]
        print("target is ", target)
        # np.save(os.path.join('visualization', f"input_original_image{image_num}.npy"), x.cpu().numpy())
        viz_one_patch(model, x, device = device, log_dir=log_path)