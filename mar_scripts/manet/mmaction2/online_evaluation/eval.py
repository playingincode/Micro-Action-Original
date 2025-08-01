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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
# pickle_file_path = 'online_evaluation/test_result_pcan.pickle'
# pred_file_path = 'online_evaluation/prediction.csv'
# zip_file_path = 'online_evaluation/submission.zip'

def lv_evaluate(predictions, labels):
    # prediction and labels are action-level
    # predictions = np.argsort(predictions, axis=1)[:, -1:][:, ::-1]
    # pre=[]
    # for i in predictions:
    #     pre.append(i[0])
    # predictions=pre
    # print("Type of predictions[0]:", type(predictions[0]))
    # print("Sample predictions[0]:", predictions[0])
    # print("Type of labels[0]:", type(labels[0]))
    # print("Sample labels[0]:", labels[0])
    # prediction_copy=predictions
    # predictions = [d['pred_labels']['item'].numpy() for d in prediction_copy]
    # # print(predictions)
    # labels = [d['gt_labels']['item'].item() for d in prediction_copy]
    predictions_copy= np.argsort(predictions, axis=1)[:, -1:][:, ::-1]
    pre=[]
    for i in predictions_copy:
        pre.append(i[0])
    predictions_copy=pre
    # print(predictions_copy)
    lv1_preds = [fine2coarse(lv2id) for lv2id in predictions_copy]
    lv1_labels = [fine2coarse(lv2id) for lv2id in labels]
    
    
    lv2_f1_micro = f1_score(labels, predictions_copy, average='micro')
    lv2_f1_macro = f1_score(labels, predictions_copy, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0
    cm_lv2 = confusion_matrix(labels, predictions_copy)
    per_class_acc_lv2 = cm_lv2.diagonal() / cm_lv2.sum(axis=1)

    # Optionally, for level-1 (coarse) classes too
    cm_lv1 = confusion_matrix(lv1_labels, lv1_preds)
    per_class_acc_lv1 = cm_lv1.diagonal() / cm_lv1.sum(axis=1)

    # If you have class names:
    # lv2_class_names = [fine_id2name(i) for i in range(len(per_class_acc_lv2))]
    # lv1_class_names = [coarse_id2name(i) for i in range(len(per_class_acc_lv1))]

    # Add to eval_results dictionary
    

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions_copy),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}
    eval_results.update({
        'lv2_per_class_accuracy': dict(enumerate(per_class_acc_lv2)),
        'lv1_per_class_accuracy': dict(enumerate(per_class_acc_lv1))
    })

    return eval_results


def load_predictions_and_labels(pickle_path, gt_label_file=None):
    
    with open(pickle_path, 'rb') as f:
        predictions = pickle.load(f)  # List of np.array
    # print(predictions)
    predictions = np.array(predictions)  # shape: (N, num_classes)

    if gt_label_file is not None:
        with open(gt_label_file, 'r') as f:
            labels = [int(line.strip().split()[1]) for line in f.readlines()]
        labels = np.array(labels)
    else:
        labels = None  # Or raise an error if needed

    return predictions, labels

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


preds, labels = load_predictions_and_labels(
pickle_path='/data/stars/user/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/online_evaluation/test_result_manet.pickle',
gt_label_file='/data/stars/user/npoddar/val_list_videos_mpII.txt'  # assuming labels are in second column
)
# print(preds)
eval_results = lv_evaluate(preds, labels)
for k, v in eval_results.items():
        if isinstance(v, float):
            print(f'{k}: {v:.4f}')
        elif isinstance(v, dict):
            print(f'{k}:')
            for class_id, acc in v.items():
                print(f'  Class {class_id}: {acc:.4f}')
        else:
            print(f'{k}: {v}')  # fallback for unexpected types
# with open(pickle_file_path, 'rb') as file:
#     datas = pickle.load(file)

# with open('./data/ma52/test_list_videos.txt', 'r') as f:
#     file_names = [line.strip().split()[0] for line in f.readlines()]

# with open(pred_file_path, 'w', newline='') as f:
#     writer = csv.writer(f)
#     header = ['vid'] + \
#              [f'action_pred_{i}' for i in range(1, 6)] + \
#              [f'body_pred_{i}' for i in range(1, 6)]
#     writer.writerow(header)

#     for index, data in enumerate(datas):
#         file_name = file_names[index]

#         pred_scores = data
#         top5_fine = np.argsort(pred_scores)[-5:][::-1].tolist()

#         # convert action-level label to body-level label. 
#         # Note that this is a simple conversion.
#         top5_coarse = [fine2coarse(x) for x in top5_fine]

#         writer.writerow([file_name] + top5_fine + top5_coarse)

# with zipfile.ZipFile(zip_file_path, 'w') as zipf:
#     zipf.write(pred_file_path, os.path.basename(pred_file_path))