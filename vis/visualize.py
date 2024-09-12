import nibabel as nib
import numpy as np
import os
import sys
sys.path.append('..')
from torch.utils.data import DataLoader
from utils.brats_160 import Brats160Data
import torch
import torchio as tio

import models_vim_mae
import matplotlib.pyplot as plt



# input_activations = {}
# def get_input(name):
#         def hook(model, input, output):
#             input_activations[name] = input
#         return hook

def viz_one_patch(model, dataset_test, log_dir = "", device = 'cuda'):
    # set hooker

    
    # dataset_test = build_dataset(mode='test', use_z_score=True)
    x, target = dataset_test[0]
    print("target is ", target)
    
    x.unsqueeze_(0)  # Adding the batch dimension
    # make it a batch-like
    x = x.to(device)
    model = model.to(device)
    
    
    # run MAE
    # loss, y, mask = model(x.float(), mask_ratio=0.75)
    latent, mask, ids_restore = model.forward_encoder(x.float(), mask_ratio= 0.75)
    y = model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
    # loss = model.forward_loss(x.float(), pred, mask)
    y = model.unpatchify_3d(y)
    # y = torch.einsum('nclhw->nlhwc', y).detach().cpu()
    y = y.detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 3 * 1)  # (N, L*H*W, p*p*1) 1=num_channel
    mask = model.unpatchify_3d(mask)  # 1 is removing, 0 is keeping
    # mask = torch.einsum('nclhw->nlhwc', mask).detach().cpu()
    mask = mask.detach().cpu()
    
    x = x.detach().cpu()
    # x = torch.einsum('nclhw->nlhwc', x)

    # masked image
    im_masked = x * (1 - mask)
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    
    np.save(os.path.join(log_dir, 'input.npy'), x.numpy())
    np.save(os.path.join(log_dir, 'masked.npy'), im_masked.numpy()) 
    # save_nifty_img(image=x, file_name=os.path.join(log_dir, 'input.nii.gz'))
    # save_nifty_img(image=im_masked, file_name=os.path.join(log_dir, 'masked.nii.gz'))

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    np.save(os.path.join(log_dir, 'recons_und_visible.npy'), im_paste.numpy())
    # save_nifty_img(image=im_paste, file_name=os.path.join(log_dir,'recons_und_visible.nii.gz'))

    # Only the reconstruction
    # save_nifty_img(image=y, file_name=os.path.join(log_dir,'reconstruct.nii.gz'))
    np.save(os.path.join(log_dir, 'reconstruct.npy'), y.numpy())
    
    # MAE latent space with visible patches
    # print(input_activations.keys())
    # mask_token = model.mask_token
    
    # input_act, ids_restore = input_activations['for_construction']
    # input_act = input_act.detach()
    # ids_restore = ids_restore.detach()
    # mask_token = mask_token.unsqueeze(0).repeat(input_act.shape[0], ids_restore.shape[1] + 1 - input_act.shape[1], 1)
    # input_act = torch.cat([input_act, mask_token], dim=1)
    # input_act = torch.gather(input_act, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, input_act.shape[2]))  # unshuffle
    
    input_act = latent
    input_act = input_act.detach()
    ids_restore = ids_restore.detach()
    
    input_act = torch.max(input_act, dim=-1, keepdim = True).values
    
    mask_token = torch.zeros(1, 1, input_act.shape[2]).to(input_act.device)
    mask_token = mask_token.repeat(input_act.shape[0], ids_restore.shape[1] + 1 - input_act.shape[1], 1)
    input_act = torch.cat([input_act, mask_token], dim=1)
    input_act = torch.gather(input_act, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, input_act.shape[2]))  # unshuffle
    
    print('input_act', input_act.shape)
    input_act = input_act.repeat(1, 1, model.patch_embed.patch_size[0] ** 3 * 1)
    
    
    input_act = model.unpatchify_3d(input_act) # (N, C, L, H, W)
    # mean_act = torch.mean(input_act, dim=1, keepdim=True)
    input_act = input_act.cpu()
    np.save(os.path.join(log_dir, 'latent_space.npy'), input_act.numpy())
    # # plot heatmap
    # mean_act = torch.mean(input_act, dim=-1, keepdim=True)
    
    # mean_act = mean_act.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy
    input_act = input_act.squeeze(0).squeeze(0).numpy()
    print('input_act', input_act.shape) 
    plt.imshow(input_act[0], cmap='viridis')
    plt.colorbar()
    plt.savefig(os.path.join(log_dir, 'latent_space.png'))
    
    
    

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

def load_model(model_name, input_size,  num_channels, resume  = None):
    model = models_vim_mae.__dict__[model_name](
                img_size=input_size, in_chans = num_channels,
                norm_pix_loss=True, 
                use_sigmoid = False
                )
    if resume:
        checkpoint = torch.load(resume, map_location= 'cpu')
        model.load_state_dict(checkpoint['model'])
        print('Model loaded from', resume)
    return model


if __name__ == "__main__":
    base_path = None
    seed = 42
    use_z_score = True
    model_name = 'mae_3d_vim_small_patch32'
    input_size = 160
    nb_classes = 2 
    num_channels = 4 
    drop_path = 0.1
    resume = "../vim_small_brats_160_patch32/checkpoint-999.pth"
    device = 'cuda'
    test_trs = tio.RandomBiasField(
        coefficients=(0.1, 0.1)
    )
    whole_dataset = Brats160Data( 
                                transform= test_trs,
                                mode = 'test',
                                use_z_score=use_z_score,
                                pre_load=False
                                )
    length = len(whole_dataset)
    train_size= int(0.8 * length)
    val_size = length - train_size 
    generator1 = torch.Generator().manual_seed(seed)
    dataset_train, dataset_val  = torch.utils.data.random_split(whole_dataset, [train_size, val_size], generator=generator1)
    data_list = [data[1] for data in dataset_val]
    print(data_list)
    model = load_model(model_name, input_size, num_channels, resume= resume)
    viz_one_patch(model, dataset_val, device = device, log_dir="visualization/vim_small_brats_160_p32")
    
    