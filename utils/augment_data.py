import os
import cv2
import numpy as np

# ==============================
# 🔷 AUGMENTATION FUNCTIONS
# ==============================

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def change_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Convert to int to prevent overflow
    v = v.astype(np.int16)

    v = v + value
    v = np.clip(v, 0, 255)

    v = v.astype(np.uint8)

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.int16)
    noisy = image.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def blur_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def zoom(image, scale=1.1):
    h, w = image.shape[:2]
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    return resized[0:h, 0:w]


def flip_image(image):
    return cv2.flip(image, 1)


# ==============================
# 🔷 MAIN PIPELINE
# ==============================

def main():

    # Calculate the base directory (one level up from utils)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dataset_dir = os.path.join(base_dir, "dataset")
    augmented_dir = os.path.join(base_dir, "augmented_dataset")

    os.makedirs(augmented_dir, exist_ok=True)

    people = os.listdir(dataset_dir)
    total_augmented = 0

    for person in people:
        person_dir = os.path.join(dataset_dir, person)

        if not os.path.isdir(person_dir):
            continue

        person_aug_dir = os.path.join(augmented_dir, person)
        os.makedirs(person_aug_dir, exist_ok=True)

        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png'))]

        print(f"Processing '{person}' - found {len(images)} images")

        for img_name in images:
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            base = os.path.splitext(img_name)[0]

            # ==============================
            # 🔥 AUGMENTATIONS
            # ==============================

            variations = {
                "orig": img,
                "flip": flip_image(img),
                "rot15": rotate_image(img, 15),
                "rotm15": rotate_image(img, -15),
                "bright": change_brightness(img, 40),
                "dark": change_brightness(img, -40),
                "blur": blur_image(img),
                "noise": add_noise(img),
                "zoom": zoom(img, 1.1),
            }

            for key, aug_img in variations.items():
                save_path = os.path.join(person_aug_dir, f"{base}_{key}.jpg")
                cv2.imwrite(save_path, aug_img)
                total_augmented += 1

    print("\n✅ Augmentation completed!")
    print(f"Total generated images: {total_augmented}")


# ==============================
# 🔷 RUN
# ==============================

if __name__ == "__main__":
    main()