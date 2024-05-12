"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from model import Model
from process_data import  preprocess,  preprocess_mask
from torchvision.datasets import Cityscapes
from torch.utils.data import random_split, DataLoader
from argparse import ArgumentParser
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import map_id_to_train_id

import os
import random


import sys
sys.path.append("/gpfs/home6/scur0004/.local/lib/python3.6/site-packages")


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser



def plot_images_with_masks(images, predictions, masks, indices, num_images_per_row=2, title="default_title1",
                                  save=False, save_path="./results_2/plots"):
    num_images = len(indices)
    num_rows = (num_images + num_images_per_row - 1) // num_images_per_row

    fig, axes = plt.subplots(num_rows, num_images_per_row * 3, figsize=(15, 5*num_rows))  # Changed from 2 to 3

    for i, idx in enumerate(indices):
        img = images[idx]
        pred = predictions[idx]
        mask = masks[idx]

        ax_img = axes[i // num_images_per_row, 3 * (i % num_images_per_row)]  # Changed from 2 to 3
        ax_mask = axes[i // num_images_per_row, 3 * (i % num_images_per_row) + 1]  # Changed from 2 to 3
        ax_pred = axes[i // num_images_per_row, 3 * (i % num_images_per_row) + 2]  # Changed from 2 to 3

        ax_img.imshow(img.cpu().permute(1, 2, 0).numpy())
        ax_img.set_title(f'Input Image')
        ax_img.axis('off')

        ax_mask.imshow(mask.cpu().numpy(), cmap='jet')
        ax_mask.set_title(f'Target')
        ax_mask.axis('off')

        ax_pred.imshow(pred.argmax(axis=0).cpu().numpy(), cmap='jet')
        ax_pred.set_title(f'Prediction')
        ax_pred.axis('off')

    plt.tight_layout()

    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{title}.png'))

    plt.show()

def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    # set cuda reservation max split size
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32' 

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    # data loading
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic',
                         transform=preprocess, target_transform=preprocess_mask)
    

    # define batch size and epochs
    batch_size = 16

    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    model_name = 'efficientnet_e15_15.pth' #'model_10e_sgd_01.pth'#'model_10e_sgd_01.pth' # # 'intermediate_model_1x1convs_depthwiseconvs_10e_no_wd.pth'


    # Define the model and optimizer
    model = Model().to(device)

    model_path = model_name 
    # Load the model state dict
    model_state_dict = torch.load(model_path)

    # Load the state dict into the model
    model.load_state_dict(model_state_dict)

    # Visualize outputs from the model
    model.eval()
    
    num_images = 0
    with torch.no_grad():
        for (images, masks) in val_loader:
                
                images = images.to(device)

                masks = map_id_to_train_id(masks).to(device)

                predictions = model(images) 
  
                num_images +=1
                if num_images==8:
                    break

    plot_images_with_masks(images=images, predictions=predictions, masks=masks, indices=[0, 1, 2, 3], num_images_per_row=2,
                            save=True, save_path="./results_2/" + model_name)
    
if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
