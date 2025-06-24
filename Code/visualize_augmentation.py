import os
import matplotlib.pyplot as plt
from dataloader import data_transforms # We import the transforms from our dataloader
import cv2
import math

# loads an image, applies transformations and visualizes everything
def visualize_augmentations(image_path, num_examples=8):

    if not os.path.exists(image_path):
        print(f"Error: Image not found")
        return

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    train_transform = data_transforms['train']

    num_cols = 3
    # Calculate the number of rows needed
    total_images = num_examples + 1
    num_rows = math.ceil(total_images / num_cols)
    
    _, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    axes = axes.flatten()


    # display the original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # generate and display augmented versions
    for i in range(1, total_images):
        augmented_tensor = train_transform(original_image)
        augmented_image_numpy = augmented_tensor.permute(1, 2, 0).numpy()
        
        axes[i].imshow(augmented_image_numpy)
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis('off')

    for i in range(total_images, len(axes)):
        axes[i].axis('off')

    plt.suptitle("Demonstration of Data Augmentation", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()



if __name__ == '__main__':
    visualize_augmentations('Archive/instruments/test/ocarina/3.jpg' )