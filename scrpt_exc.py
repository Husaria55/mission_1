import os
import zipfile

# 1. Define your folders
# Where are all your zip files right now?
source_folder = r"C:\Users\Bartek\Desktop\SUAV\labels"

# Where do you want the extracted folders to go?
main_destination_folder = r"C:\Users\Bartek\Desktop\SUAV\extracted"

# Create the main destination folder if it doesn't exist
os.makedirs(main_destination_folder, exist_ok=True)

# 2. Loop through every file in your source folder
for filename in os.listdir(source_folder):
    
    # 3. Process ONLY the .zip files
    if filename.endswith(".zip"):
        
        # Get the full path of the current zip file
        zip_path = os.path.join(source_folder, filename)
        
        # Create a name for the new subfolder (removes the ".zip" extension)
        # e.g., "project1.zip" becomes a folder named "project1"
        folder_name = filename[:-4] 
        specific_extraction_path = os.path.join(main_destination_folder, folder_name)
        
        # Create that specific subfolder
        os.makedirs(specific_extraction_path, exist_ok=True)
        
        # 4. Extract the contents into the new subfolder
        print(f"Extracting {filename} into \\{folder_name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(specific_extraction_path)
        except zipfile.BadZipFile:
            print(f"  -> ERROR: {filename} is corrupted or not a valid zip file. Skipping.")

print("\nAll done! Your files are extracted and organized.")