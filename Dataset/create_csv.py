# In Dataset/create_csv.py

import os
import pandas as pd

def catalog_folder(root_dir, output_csv_path):
    """Scans a directory and creates a CSV file listing images and their labels."""
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
    
    df = pd.DataFrame(data, columns=['image_path', 'label'])
    df.to_csv(output_csv_path, index=False)
    print(f"Successfully created {os.path.basename(output_csv_path)} with {len(df)} entries.")
    return class_names

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define paths to the train, valid, and test directories
    train_dir = os.path.join(project_root, 'Archive', 'instruments', 'train')
    valid_dir = os.path.join(project_root, 'Archive', 'instruments', 'valid')
    test_dir = os.path.join(project_root, 'Archive', 'instruments', 'test') # <-- New path
    
    output_train_csv = os.path.join(script_dir, 'train.csv')
    output_val_csv = os.path.join(script_dir, 'validation.csv')
    output_test_csv = os.path.join(script_dir, 'test.csv') # <-- New output file
    output_class_names = os.path.join(script_dir, 'class_names.txt')

    # Create CSVs for all three sets
    class_names = catalog_folder(train_dir, output_train_csv)
    catalog_folder(valid_dir, output_val_csv)
    catalog_folder(test_dir, output_test_csv) # <-- Create test.csv

    with open(output_class_names, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Saved class names to {os.path.basename(output_class_names)}")