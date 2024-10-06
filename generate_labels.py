import os

def generate_labels_file(dataset_path, output_file):
    # Get sorted list of subdirectories (class names)
    class_names = sorted(os.listdir(dataset_path))
    
    # Filter out any non-directory items
    class_names = [d for d in class_names if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Write class names to file
    with open(output_file, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"Labels file created at {output_file}")
    print(f"Classes found: {', '.join(class_names)}")

# Example usage
dataset_path = 'fruits/Training'  # Path to your training data directory
output_file = 'fruit_labels.txt'  # Name of the output file

generate_labels_file(dataset_path, output_file)