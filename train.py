"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from model import Model
from process_data import preprocess, preprocess_mask
from torchvision.datasets import Cityscapes
from torch.utils.data import random_split, DataLoader
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
import numpy as np

import os
import gc
import random
from utils import map_id_to_train_id
import torch.quantization as quantization


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
        ax = axes[i // num_images_per_row, i % num_images_per_row] if num_rows > 1 else axes[i % num_images_per_row]
        ax.imshow(img.permute(1, 2, 0))
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

    # Split the dataset into training and validation sets
    trainset, valset = random_split(dataset, [train_size, val_size])

    # define batch size and epochs
    batch_size = 16

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)



    # Define the model and optimizer
    model = Model().to(device)

    # Quantization
    model = quantization.quantize_dynamic(model, {torch.nn.Conv2d}, dtype=torch.qint8)

    epochs = 2
    lr = 0.15
    model_name = 'image_plot.pth'

    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3) #
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)


    # training
    train_losses = []
    val_losses = []
    val_dices = []
    for epoch in range(1, epochs + 1):

        print(f"\nEpoch {epoch}")

        # train
        model.train()
        total_train_loss = 0.0
        num_train_batches = len(train_loader)
        
        for (images, masks) in train_loader:
            images = images.to(device)
            masks = map_id_to_train_id(masks).to(device)

            optimizer.zero_grad()
            outputs = model(images) 
            loss = criterion(outputs, masks.squeeze().long())
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # clear cuda memory
            gc.collect()
            torch.cuda.empty_cache()
        train_losses.append(total_train_loss / num_train_batches)
        print(f"Training Loss: {train_losses[-1]}")

        # validation
        model.eval()
        total_val_loss = 0.0
        num_val_batches = len(val_loader)
        with torch.no_grad():
            for (images, masks) in val_loader:
                images = images.to(device)
                masks = map_id_to_train_id(masks).to(device)

                outputs = model(images)
                loss = criterion(outputs, masks.squeeze().long())
                total_val_loss += loss.item()
            val_losses.append(total_val_loss / num_val_batches)


            # clear cuda memory
            gc.collect()
            torch.cuda.empty_cache()
        print(f"Validation Loss: {val_losses[-1]}")
        
        scheduler.step()

    torch.save(model.state_dict(),model_name)
    print(model_name)

    # Visualize some outputs from the model
    model.eval()
    num_batches_to_visualize = 2  # Number of batches to visualize
    save_path = "./results/outputs"  # Directory to save the images

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            if i >= num_batches_to_visualize:
                break

            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)

            # Convert predictions and masks to numpy arrays
            outputs_np = outputs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # Save images and their corresponding masks
            for j in range(len(images)):
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(np.transpose(images[j].cpu().numpy(), (1, 2, 0)))
                plt.title("Input Image")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                #plt.imshow(outputs_np[j], cmap='gray')
                plt.imshow(outputs_np[j][0], cmap='gray')
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
