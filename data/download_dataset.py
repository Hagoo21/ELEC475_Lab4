import kagglehub
import os

def download_and_inspect():
    print("Downloading COCO 2014 dataset via KaggleHub...")
    
    # Download latest version
    # Use the COCO 2014 dataset from jeffaudi/coco-2014-dataset-for-yolov3
    path = kagglehub.dataset_download("jeffaudi/coco-2014-dataset-for-yolov3")

    print(f"\nSUCCESS! Dataset downloaded to: {path}")
    print("-" * 50)
    print(f"Contents of {path}:")
    
    # Inspect the directory to confirm folder names for your config
    try:
        contents = os.listdir(path)
        for item in contents:
            print(f" - {item}")
            
        # Check for annotations specifically as they are critical for captions
        [cite_start]# [cite: 48] Captions from captions_train2014.json
        if "annotations" in contents:
             print("\nInside 'annotations' folder:")
             print(os.listdir(os.path.join(path, "annotations")))
        else:
             # Sometimes they are in the root
             print("\n(Check if .json caption files are in the root list above)")
             
    except Exception as e:
        print(f"Could not list directory: {e}")

    print("-" * 50)
    print("ACTION REQUIRED: Copy the 'Dataset downloaded to' path above.")
    print("Paste it as the DATA_DIR variable in 'preprocess_data.py' and 'dataset.py'.")

if __name__ == "__main__":
    download_and_inspect()