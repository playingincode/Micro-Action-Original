import numpy as np
import torch
import os

output_file = "output_files_not_properly_generated_files.txt"

folders = [
    "/srv/storage/stars@storage3.sophia.grid5000.fr/areka/areka/CVPR/features_test_ma52/test_face_ma_52/test/",
    "/srv/storage/stars@storage3.sophia.grid5000.fr/areka/areka/CVPR/features_test_ma52/test_body_ma_52/test/",
    "/srv/storage/stars@storage3.sophia.grid5000.fr/areka/areka/CVPR/features_test_ma52/test_upper_limb_ma_52/test/",
    "/srv/storage/stars@storage3.sophia.grid5000.fr/areka/areka/CVPR/features_test_ma52/test_lower_limb_ma_52/test/"
]

with open(output_file, "w") as f:
    for folder_name in folders:
        for file in os.listdir(folder_name):
            if file.endswith(".npy"):
                full_path = os.path.join(folder_name, file)
                try:
                    features_from_np = np.load(full_path)
                    features_tensor = torch.tensor(features_from_np)
                    if features_tensor.ndim != 3:
                        f.write(full_path + "\n")
                except Exception as e:
                    f.write(f"ERROR loading {full_path}: {e}\n")