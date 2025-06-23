# In Dataset/create_csv.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_and_split_dataset_csv(root_dir, validation_split=0.2):
    """
    Scans a directory with class-named subfolders, splits the data into
    training and validation sets, and creates train.csv and validation.csv.
    """
    # --- Make paths robust ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_train_csv = os.path.join(script_dir, 'train.csv')
    output_val_csv = os.path.join(script_dir, 'validation.csv')
    output_class_names = os.path.join(script_dir, 'class_names.txt')

    data = []
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    print(f"Found {len(class_names)} classes in {root_dir}.")

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                relative_path = os.path.join(class_name, file_name)
                label = class_to_idx[class_name]
                data.append([relative_path, label])

    full_df = pd.DataFrame(data, columns=['image_path', 'label'])
    
    # Split the data, ensuring all classes are represented proportionally
    train_df, val_df = train_test_split(
        full_df, 
        test_size=validation_split, 
        random_state=42, # Ensures the split is the same every time
        stratify=full_df['label']
    )

    train_df.to_csv(output_train_csv, index=False)
    val_df.to_csv(output_val_csv, index=False)
    
    print(f"Successfully created {os.path.basename(output_train_csv)} with {len(train_df)} entries.")
    print(f"Successfully created {os.path.basename(output_val_csv)} with {len(val_df)} entries.")

    with open(output_class_names, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Saved class names to {os.path.basename(output_class_names)}")

if __name__ == '__main__':
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # This path should point to your single folder of categorized images
    image_root_directory = os.path.join(project_root, 'Archive', 'all_images')
    
    create_and_split_dataset_csv(image_root_directory)