'''
Packaging to generate zip files for uploading, 
the default coarse-grained labels are derived from the fine-grained labels, 
if you use separately generated coarse-grained labels, be sure to modify the relevant code.
'''
import os
import pickle
import zipfile
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
pickle_file_path = 'online_evaluation/test_result_our_model_on_mpii.pickle'
pred_file_path = 'online_evaluation/predicition_our_model_mpii.csv'
zip_file_path = 'online_evaluation/submission_our_model_mpii.zip'

def fine2coarse(x):
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

with open(pickle_file_path, 'rb') as file:
    datas = pickle.load(file)
    # print(datas)

with open('/data/stars/user/npoddar/val_list_videos_mpII.txt', 'r') as f:
    lines = f.readlines()
    file_names = [line.strip().split()[0] for line in lines]
    labels = [int(line.strip().split()[1]) for line in lines]
    y_true = []
    y_pred = []
    for pred_scores, label in zip(datas, labels):

        pred_label = int(np.argmax(pred_scores))

        y_true.append(label)
        y_pred.append(pred_label)

        # Top-1 accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0
    print(f"Top-1 Accuracy (Classes 0–10): {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(19)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(19)))

    fig, ax = plt.subplots(figsize=(10,8))  # increase or adjust size
    disp.plot(cmap='Blues', ax=ax)  # format to avoid too many decimals

    # Tweak label rotation and font size
    # plt.xticks(ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Confusion Matrix (Classes 0–18)', fontsize=12)

    # Ensure there's enough space
    plt.tight_layout()
    plt.savefig('./confusion_matrix_with_our_model_on_mpii_eval.png', dpi=300)
    plt.show()
    # print(labels)

with open(pred_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['vid'] + \
             [f'action_pred_{i}' for i in range(1, 6)] + \
             [f'body_pred_{i}' for i in range(1, 6)]
    writer.writerow(header)

    for index, data in enumerate(datas):
        file_name = file_names[index]

        pred_scores = data
        top5_fine = np.argsort(pred_scores)[-5:][::-1].tolist()

        # convert action-level label to body-level label. 
        # Note that this is a simple conversion.
        top5_coarse = [fine2coarse(x) for x in top5_fine]

        writer.writerow([file_name] + top5_fine + top5_coarse)

with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    zipf.write(pred_file_path, os.path.basename(pred_file_path))