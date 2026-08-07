import os

# --- Configuration ---
# 1. Path to your already-extracted labels
labels_extracted_dest = r"C:\Users\Bartek\Desktop\SUAV\extracted"

# 2. Point this at the TOP LEVEL images folder
# os.walk will automatically go into Actor001-Actor010, Actor081-Actor090, etc.
images_source_folder = r"C:\Users\Bartek\Desktop\SUAV\images"

# The altitude markers we want to KEEP
KEEP_ALTITUDES = ["_a50", "_a70"]

# --- 1. Filter Labels ---
print("--- PHASE 1: FILTERING ALREADY-EXTRACTED LABELS ---")
removed_labels = 0
kept_labels = 0

for root, dirs, files in os.walk(labels_extracted_dest):
    for file in files:
        if file.endswith(".txt"):
            if any(alt in file for alt in KEEP_ALTITUDES):
                kept_labels += 1
            else:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    removed_labels += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

print(f"Done! Kept {kept_labels} labels. Deleted {removed_labels} non-target labels.")


# --- 2. Filter Images ---
print("\n--- PHASE 2: FILTERING IMAGES ---")
removed_images = 0
kept_images = 0

for root, dirs, files in os.walk(images_source_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg")):
            if any(alt in file for alt in KEEP_ALTITUDES):
                kept_images += 1
            else:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    removed_images += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

print(f"Done! Kept {kept_images} images. Deleted {removed_images} non-target images.")
print("\nCleanup complete! Your entire dataset is now restricted to 50m and 70m altitudes.")