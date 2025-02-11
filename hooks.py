import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageFilter

class AttentionRecorder:
    def __init__(self, modules):
        ## init
        self.modules = modules if isinstance(modules, list) else [modules]
        self.saved_attentions = {i: None for i in range(len(self.modules))}
        self.hook_handles = []

    def hook_fn(self, idx):
        # get attentions
        def inner_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                self.saved_attentions[idx] = output[1].detach().cpu()
                print(f"Hook {idx}: Captured attention with shape {self.saved_attentions[idx].shape}")
            else:
                self.saved_attentions[idx] = None
                print ("Missed!")
        return inner_hook

    def __enter__(self):
        # register hooks
        for idx, module in enumerate(self.modules):
            hook = module.register_forward_hook(self.hook_fn(idx))
            self.hook_handles.append(hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # dump hooks
        for hook in self.hook_handles:
            hook.remove()

# Attntion roll out adapted for ViT
# paper: https://arxiv.org/pdf/2005.00928
def attention_rollout(atten_weights, image):  
    rollout = torch.eye(atten_weights[0].size(-1))
    with torch.no_grad():
        for weight in atten_weights:
            attention_heads_avg = weight.mean(axis=1)
                
            I = torch.eye(attention_heads_avg.size(-1), 
              device=attention_heads_avg.device, 
              dtype=attention_heads_avg.dtype)
            alpha = (attention_heads_avg + I) * 0.5
            alpha = F.normalize(alpha, p=1, dim=-1)
            rollout = alpha @ rollout
    
    ## image processing --> ViT patches
    rollout = rollout[:, 0, 1:] ## we gonna use the cls token
    rollout = rollout.squeeze(0)
    num_patches = rollout.shape[0]
    grid_size = int(np.sqrt(num_patches))  
    attn_map = rollout.reshape(grid_size, grid_size).numpy()
    # norm to [0 1]
    attn_map = attn_map - attn_map.min()  
    attn_map = attn_map / attn_map.max()  
    attn_map = (attn_map * 255).astype(np.uint8)  # uint8 conv


    rollout = Image.fromarray(attn_map)
    attn_map_resized = rollout.resize((image.width, image.height), Image.Resampling.BICUBIC)
    ## fancy gaussian blur
    attn_map_resized = attn_map_resized.filter(ImageFilter.GaussianBlur(radius = 10))
    
    return attn_map_resized 
