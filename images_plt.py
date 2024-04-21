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

import os
import random


import sys
sys.path.append("/gpfs/home6/scur0004/.local/lib/python3.6/site-packages")


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def plot_images_with_masks(dataset, indices, num_images_per_row=2, title="default_title",
                           save=True, save_path="./results/plots"):
    num_images = len(indices)
    num_rows = (num_images + num_images_per_row - 1) // num_images_per_row

    fig, axes = plt.subplots(num_rows, num_images_per_row, figsize=(10, 5*num_rows))

    for i, idx in enumerate(indices):
        img, mask = dataset[idx]
        # img, mask = preprocess_train(img), preprocess_train(mask)
        ax = axes[i // num_images_per_row, i % num_images_per_row] if num_rows > 1 else axes[i % num_images_per_row]
        ax.imshow(img.permute(1, 2, 0))
        # ax.imshow(mask.permute(1, 2, 0), alpha=0.35, cmap='jet')
        ax.set_title(f'Image {idx}')
        ax.axis('off')

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
    
    plot_images_with_masks(dataset, indices=[0, 1, 2, 3, 4, 5, 6, 7], num_images_per_row=4,
                            save=False, save_path="./results/plots"    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    trainset, valset = random_split(dataset, [train_size, val_size])


    batch_size = 16

    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)



    # Define the model and optimizer
    model = Model().to(device)

    model_path = 'model_10e_sgd_01.pth'#'intermediate_model_1x1convs_depthwiseconvs_10e_no_wd.pth' #'efficientnet_e15_15.pth' #

    # Load the model state dict
    model_state_dict = torch.load(model_path)


    # Load the state dict into the model
    model.load_state_dict(model_state_dict)

    # Visualize outputs from the model
    model.eval()
    num_batches_to_visualize = 1  
    save_path = "./results/" + model_path 

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            if i >= num_batches_to_visualize:
                break

            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            outputs_np = outputs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # Save images and their corresponding masks
            for j in range(len(images)):
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(np.transpose(images[j].cpu().numpy(), (1, 2, 0)))
                plt.title("Input Image")
                plt.axis("off")


                plt.subplot(1, 3, 3)
                plt.imshow(masks_np[j][0], cmap='viridis')
                plt.title("Masks")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(outputs_np[j][0], cmap='viridis')
                plt.title("Model Output")
                plt.axis("off")


                # Save the figure
                plt.savefig(os.path.join(save_path, f"output_image_{i * len(images) + j}.png"))
                plt.close()


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
