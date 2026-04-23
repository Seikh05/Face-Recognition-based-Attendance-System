import os
import cv2

def check_dataset_properties(dataset_dir):
    print(f"Analyzing dataset at: {dataset_dir}\n")
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Directory {dataset_dir} does not exist.")
        return
        
    total_images = 0
    extensions = set()
    dimensions = set()
    min_width, max_width = float('inf'), 0
    min_height, max_height = float('inf'), 0
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                total_images += 1
                extensions.add(ext)
                
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"⚠️ Warning: Could not read image {img_path}")
                    continue
                    
                h, w, c = img.shape
                dimensions.add((h, w, c))
                
                min_width = min(min_width, w)
                max_width = max(max_width, w)
                min_height = min(min_height, h)
                max_height = max(max_height, h)

    print("--- Dataset Summary ---")
    print(f"Total Images   : {total_images}")
    if total_images > 0:
        print(f"File Types     : {', '.join(extensions)}")
        print(f"Width Range    : {min_width} to {max_width} pixels")
        print(f"Height Range   : {min_height} to {max_height} pixels")
        
        if len(dimensions) == 1:
            h, w, c = next(iter(dimensions))
            print(f"Dimensions     : All images are {w}x{h} with {c} channels")
        else:
            print(f"Dimensions     : Mixed dimensions. Found {len(dimensions)} unique shapes.")
            if len(dimensions) <= 5:
                # If there aren't too many, just list them
                print(f"Found Shapes   : {dimensions}")
    else:
        print("No valid images found.")

if __name__ == "__main__":
    # Get to the PDD folder (since utils is inside PDD)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "augmented_dataset")
    check_dataset_properties(DATASET_DIR)
