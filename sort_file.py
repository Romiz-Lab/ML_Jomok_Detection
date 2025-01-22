import os

folder_path = "data/train/sexy"

custom_prefix = "sxy" # Custom name prefix
counter = 1  # Start numbering

for filename in os.listdir(folder_path):
    if os.path.isdir(os.path.join(folder_path, filename)) or not filename.lower().endswith(".jpeg"):
        continue

    new_name = f"{custom_prefix}_{counter}.jpeg"
    counter += 1

    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_name)
    os.rename(old_file, new_file)

print(f"Semua file .jpeg berhasil diganti dengan nama custom {custom_prefix}")
