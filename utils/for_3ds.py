from utils import misc
import numpy as np
import torch 

def select_best_model(args, epoch, loss_scaler, max_val, model, model_without_ddp, optimizer, cur_val,
                      model_name='best_ft_model'):
    if cur_val > max_val:
        print(f"saving {model_name} @ epoch {epoch}")
        max_val = cur_val
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=model_name)  # A little hack for saving model with preferred name
    return max_val



class MixUp3D:
    def __init__(self, mixup_alpha):
        super(MixUp3D, self).__init__()
        self.mixup_alpha = mixup_alpha

    def partial_mixup(self, input, indices):
        if input.size(0) != indices.size(0):
            raise RuntimeError("Size mismatch!")
        perm_input = input[indices]
        lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        return input.mul(lam_mix).add(perm_input, alpha=1 - lam_mix)

    def __call__(self, input, target):
        indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
        return self.partial_mixup(input, indices), self.partial_mixup(target, indices)


