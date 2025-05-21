# scripts/download_arc.py

import os
import requests
import zipfile
import io

ARC_DATASET_URL = "https://github.com/fchollet/ARC/raw/master/data/training_evaluation.zip"
DATA_DIR = "../data/arc_dataset_raw" # Raw download destination
TARGET_DIR = "../data/arc" # Processed/organized data destination

def download_and_extract_arc(url, download_dir, extract_to_dir):
    """Downloads and extracts the ARC dataset."""
    print(f"Downloading ARC dataset from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors

        # Ensure download directory exists
        os.makedirs(download_dir, exist_ok=True)

        zip_file_path = os.path.join(download_dir, "arc_dataset.zip")

        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {zip_file_path}")

        print(f"Extracting ARC dataset to {extract_to_dir}...")
        os.makedirs(extract_to_dir, exist_ok=True)
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            zf.extractall(extract_to_dir)
        print("Extraction complete.")
        
        # Clean up the zip file
        # os.remove(zip_file_path)
        # print(f"Removed zip file: {zip_file_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file or is corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Note: This script assumes it's run from the 'scripts' directory.
    # Adjust paths if running from project root or elsewhere.
    
    # Determine base path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Assumes scripts is one level down from root
    
    raw_data_path = os.path.join(project_root, "data", "arc_dataset_raw")
    target_extraction_path = os.path.join(project_root, "data", "arc")

    # Create these directories if they don't exist to avoid issues with os.path.join creating partial paths
    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(target_extraction_path, exist_ok=True) 
    # The ARC zip extracts into 'data/training' and 'data/evaluation' folders directly.
    # We want them inside our target_extraction_path (e.g. data/arc/training, data/arc/evaluation)
    # So we will extract to target_extraction_path/data and then move them or adjust.
    # For now, the structure within the zip is 'data/training/*.json' and 'data/evaluation/*.json'.
    # The zip file itself contains a 'data' directory.
    # So, if we extract to `target_extraction_path`, it will create `target_extraction_path/data/training` etc.
    # This is not ideal. We should extract to a temporary place or handle the paths carefully.

    # Let's extract to a temporary subfolder within target_extraction_path and then move if needed.
    temp_extract_dir = os.path.join(target_extraction_path, "temp_arc_extract")
    os.makedirs(temp_extract_dir, exist_ok=True)

    print(f"Attempting to download ARC dataset from {ARC_DATASET_URL}")
    print(f"Temporary extraction directory will be: {temp_extract_dir}")
    print(f"Final ARC data should reside in subdirectories of: {target_extraction_path}")

    # We'll download the zip to raw_data_path first.
    zip_target_path = os.path.join(raw_data_path, "arc_dataset.zip")

    try:
        response = requests.get(ARC_DATASET_URL, stream=True)
        response.raise_for_status()
        with open(zip_target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded ARC dataset to {zip_target_path}")

        print(f"Extracting ARC dataset from {zip_target_path} into {temp_extract_dir}...")
        with zipfile.ZipFile(zip_target_path, 'r') as zf:
            zf.extractall(temp_extract_dir)
        print("Extraction to temporary directory complete.")

        # Now, organize the files. The zip extracts to a 'data' subdirectory.
        extracted_data_folder = os.path.join(temp_extract_dir, "data")
        if os.path.exists(extracted_data_folder):
            # Move training files
            source_training = os.path.join(extracted_data_folder, "training")
            dest_training = os.path.join(target_extraction_path, "training")
            if os.path.exists(source_training):
                print(f"Moving {source_training} to {dest_training}")
                if os.path.exists(dest_training):
                    # If destination exists, remove it or merge carefully.
                    # For simplicity here, we'll assume we can overwrite or it's empty.
                    # A more robust script would handle this better (e.g. shutil.rmtree then move)
                    import shutil
                    if os.listdir(dest_training): # if not empty
                        print(f"Warning: Destination {dest_training} is not empty. Content might be overwritten.")
                    shutil.move(source_training, dest_training)
                else:
                    shutil.move(source_training, dest_training)
                print("Moved training files.")
            else:
                print(f"Warning: {source_training} not found after extraction.")

            # Move evaluation files
            source_evaluation = os.path.join(extracted_data_folder, "evaluation")
            dest_evaluation = os.path.join(target_extraction_path, "evaluation")
            if os.path.exists(source_evaluation):
                print(f"Moving {source_evaluation} to {dest_evaluation}")
                if os.path.exists(dest_evaluation):
                    import shutil
                    if os.listdir(dest_evaluation):
                         print(f"Warning: Destination {dest_evaluation} is not empty. Content might be overwritten.")
                    shutil.move(source_evaluation, dest_evaluation)
                else:
                    shutil.move(source_evaluation, dest_evaluation)
                print("Moved evaluation files.")
            else:
                print(f"Warning: {source_evaluation} not found after extraction.")
            
            # Clean up the 'data' folder inside temp_extract if it's empty
            if not os.listdir(extracted_data_folder):
                os.rmdir(extracted_data_folder)
        else:
            print(f"Error: Expected 'data' subfolder not found in {temp_extract_dir}")

        # Clean up the temporary extraction directory itself if it's empty
        if not os.listdir(temp_extract_dir):
             os.rmdir(temp_extract_dir)
        elif os.listdir(temp_extract_dir) == ['data'] and not os.listdir(os.path.join(temp_extract_dir, 'data')) : # if only empty data dir remains
            os.rmdir(os.path.join(temp_extract_dir, 'data'))
            os.rmdir(temp_extract_dir)
        else:
            print(f"Note: Temporary extraction directory {temp_extract_dir} may still contain unexpected files.")

        print(f"ARC dataset should now be available in {target_extraction_path}/training and {target_extraction_path}/evaluation")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
    except zipfile.BadZipFile:
        print(f"Error: Downloaded file {zip_target_path} is not a valid zip file or is corrupted.")
    except FileNotFoundError as e:
        print(f"Error during file operations: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Optional: Clean up the downloaded zip file
    # if os.path.exists(zip_target_path):
    #     os.remove(zip_target_path)
    #     print(f"Removed downloaded zip file: {zip_target_path}") 