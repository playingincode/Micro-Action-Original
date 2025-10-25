import os
base_dir = "/data/stars/user/npoddar/UNIK_saved_models/"
output_file = "output_files_deleted_new.txt"
with open(output_file, "w") as f: 
    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if not "all_experts_with_mpii_our_model_with_pcan_idea_final_model" in full_path:
            if os.path.isdir(full_path):
                for file in os.listdir(full_path):
                    if ".pt" in file:
                        if not "best" in file and not "latest" in file:
                            full_path_file=os.path.join(full_path,file)
                            # print(full_path_file)
                            f.write(full_path_file + "\n")
                            os.remove(full_path_file)
        
        
