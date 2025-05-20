import os

def analyze_dataset_distribution(dataset_path):
    class_counts = {}

    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            num_images = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
            class_counts[class_name] = num_images

    return class_counts

if __name__ == "__main__":
    dataset_path = "../dataset/plantvillage_dataset/color"
    distribution = analyze_dataset_distribution(dataset_path)

    print("Dataset Distribution:")
    for class_name, count in distribution.items():
        print(f"{class_name}: {count} images")