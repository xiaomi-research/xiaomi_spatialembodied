# https://github.com/YuanJianhao508/RAG-Driver/blob/main/evaluate.py
try:
    from evalcap.utils_caption_evaluate import evaluate_on_coco_caption
except ImportError as e:
    # `evalcap` is the external COCO-caption evaluation package (see the URL above); it is
    # NOT part of this repository.
    raise ImportError(
        "eval_BDD-X.py requires `evalcap.utils_caption_evaluate`. It is not part of this "
        "repository.\n  Clone the EvalCap / RAG-Driver evaluation code (the URL in the "
        "comment above) and add it to `sys.path`."
    ) from e
import argparse
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import json


def load_json_file(file_path):
    """
    Load JSON data from a file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def evaluate_signal(gt_signals, pred_signals, predict_file):
    print("Computing signal prediction score")

    # Define sigma values
    sigma_values = [0.1, 0.5, 1, 5, 10]
    accuracies = []

    # Convert lists to numpy arrays
    gt_signals = np.array(gt_signals)
    pred_signals = np.array(pred_signals)

    assert gt_signals.shape == pred_signals.shape, "The shapes of GT and predicted signals must match."

    # Compute RMSE
    rmse_signal = np.sqrt(mean_squared_error(gt_signals, pred_signals))
    print(f"RMSE: {rmse_signal}")

    # Compute accuracies for each sigma
    all_num = np.product(gt_signals.shape)  # Total number of data points
    for sigma in sigma_values:
        accuracy = np.count_nonzero(abs(gt_signals - pred_signals) < sigma) / all_num
        accuracies.append(accuracy)
        print(f"Accuracy within {sigma}: {accuracy}")

    # Create directory if it does not exist
    if not os.path.exists(os.path.dirname(predict_file)):
        os.makedirs(os.path.dirname(predict_file))

    # Write results to a JSON file
    with open(predict_file, 'w') as json_file:
        json.dump({
            "rmse": rmse_signal,
            "accuracies": accuracies
        }, json_file)

    return rmse_signal, accuracies


if __name__ == "__main__":    
    
    eval_caption, eval_signal = True, True 
    
    version = "final_conv_base"
    
    if eval_caption:
        # Caption files in COCO format
        caption_file_paths = [f"./evalcap/BDDX_gt/BDDX_Test_coco_{cap}.json" for cap in ['action','justification']]
        predict_file_paths = [f"./results/{version}/BDDX_Test_pred_{cap}.json" for cap in ['action','justification']]
        
        for idx in range(2):
            result = evaluate_on_coco_caption(predict_file_paths[idx], caption_file_paths[idx])
            
    if eval_signal:
        caption_file_path = f"./evalcap/BDDX_gt/BDDX_Test_control_signal.json"
        predict_file_path = f"./results/{version}/BDDX_Test_pred_control_signal.json" 
        
        with open(caption_file_path,"r") as fs:
            sig_data = json.load(fs)
            
        with open(predict_file_path,"r") as fp:
            pred_data = json.load(fp)
        id_pred_match = {ele['image_id']:ele['caption'] for ele in pred_data}    
        
        speed_gts, curvature_gts = [], []
        speed_preds, curvature_preds = [], []
        for idx, gt_sig in sig_data.items():
            speed_gt, curvature_gt = gt_sig['speed'], gt_sig['course']
            pred = id_pred_match[idx]
            speed_pred, curvature_pred = float(pred.split(": ")[1].split(" ")[0]),  float(pred.split(": ")[-1])
            # print(speed_gt,speed_pred, curvature_gt, curvature_pred)
            # break
            speed_gts.append(speed_gt)
            curvature_gts.append(curvature_gt)
            speed_preds.append(speed_pred)
            curvature_preds.append(curvature_pred)
        
        speed_json = f"./results/{version}/speed_eval.json" 
        curvature_json = f"./results/{version}/curvature_eval.json" 
        evaluate_signal(speed_gts, speed_preds, speed_json)
        evaluate_signal(curvature_gts, curvature_preds, curvature_json)
            

        