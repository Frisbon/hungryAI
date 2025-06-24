import os
import pandas as pd

# Scans the dir and creates a CSV file w images and labels.
def catalog_folder(root_dir, output_csv_path):

    data = []
    class_names = []
    
    for d in os.listdir(root_dir): # take all dir in root_dir
        dir_path = os.path.join(root_dir, d)
        if os.path.isdir(dir_path):
            class_names.append(d)
    class_names.sort()

    # Create a dict tale che [class_name] -> class_index
    class_to_idx = {}
    for idx, name in enumerate(class_names):
        class_to_idx[name] = idx
    
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
    print(f"Created {os.path.basename(output_csv_path)} with {len(df)} entries.")
    return class_names

if __name__ == '__main__':
    
    class_names = catalog_folder('Archive/instruments/train', 'Dataset/train.csv')
    catalog_folder('Archive/instruments/valid', 'Dataset/validation.csv')
    catalog_folder('Archive/instruments/test', 'Dataset/test.csv')
    
    if class_names:
        with open('Dataset/class_names.txt', 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        print("saved class names to Dataset/class_names.txt")