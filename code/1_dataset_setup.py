import os
import shutil
from sklearn.model_selection import train_test_split


dataset_path = r"C:\Users\aamod\MLmodel\dataset"
emotions = ["angry", "happy", "sad", "relaxed"]


os.makedirs(os.path.join(dataset_path, "train"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "val"), exist_ok=True)

for emotion in emotions:
    
    os.makedirs(os.path.join(dataset_path, "train", emotion), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "val", emotion), exist_ok=True)
    
    
    src_folder = os.path.join(dataset_path, emotion)
    all_files = [f for f in os.listdir(src_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    
    for f in train_files:
        shutil.copy(os.path.join(src_folder, f), 
                   os.path.join(dataset_path, "train", emotion, f))
    for f in val_files:
        shutil.copy(os.path.join(src_folder, f), 
                   os.path.join(dataset_path, "val", emotion, f))

print("Datasets have been arranged successfully!")