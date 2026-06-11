import os
import sys
from tqdm import tqdm

# Add the parent directory to the path to allow importing 'datasets'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the dataset class directly
from datasets.dataset_256 import ImageFolderMaskDataset


def verify_dataset(data_path):
    """
    Iterates through the entire dataset to check for loading errors.
    """
    if not os.path.exists(data_path):
        print(f"Error: Dataset path not found at '{data_path}'")
        return

    print(f"Verifying dataset at: {data_path}")

    try:
        # Initialize the dataset
        dataset = ImageFolderMaskDataset(path=data_path)
        print(f"Successfully initialized dataset with {len(dataset)} images.")
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        return

    # Iterate through all images with a progress bar
    for i in tqdm(range(len(dataset)), desc="Checking images"):
        try:
            # This will call the __getitem__ method
            image, mask, label = dataset[i]

            # Optional: Add checks for shape or type if needed
            if image is None or mask is None:
                print(f"\nError: Item {i} ({dataset._image_fnames[i]}) returned None.")
                break

        except Exception as e:
            print(f"\nCRITICAL ERROR: Failed to load item at index {i}.")
            print(f"The problematic file is likely: {dataset._image_fnames[i]}")
            print(f"Error details: {e}")
            break
    else:  # This block runs only if the loop completes without a 'break'
        print("\nVerification complete! All files loaded successfully.")


if __name__ == '__main__':
    # --- IMPORTANT: Set the path to your training data ---
    training_data_path = '/home/jincheng/Mural/mural_project/train_mural_png'
    verify_dataset(training_data_path)
