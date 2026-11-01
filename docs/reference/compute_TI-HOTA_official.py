# Official TI-HOTA implementation, copied verbatim from open-starlab/TrackID3x3 script/compute_TI-HOTA.ipynb (Apache-2.0).
# Notebook cells concatenated; paths in cells 2-3 refer to the repo layout. Reference only.

# ### CELL 0 (code)
import numpy as np
import math
from scipy.optimize import linear_sum_assignment

# Calculation of location similarity LocSim
def loc_sim(P_pos, G_pos, tau):
    """
    Function to compute location similarity LocSim
    
    \[
    \operatorname{LocSim}(P, G) = \exp\!\left( \ln(0.05) \cdot \frac{\|P - G\|_2^2}{\tau^2} \right)
    \]
    
    :param P_pos: Predicted position (coordinates in list or numpy array)
    :param G_pos: Ground Truth Location
    :param tau: normalized parameter
    :return: LocSim Value
    """
    P_pos = np.array(P_pos)
    G_pos = np.array(G_pos)
    distance_sq = np.sum((P_pos - G_pos)**2)
    sim = math.exp(math.log(0.05) * (distance_sq / (tau**2)))
    return sim

# Calculation of attribute similarity IdSim
def id_sim(P_attr, G_attr):
    """
    Function to calculate attribute similarity IdSim
    \[
    \operatorname{IdSim}(P, G) =
    \begin{cases}
    1, & \text{if } P_{\text{attr}} = G_{\text{attr}}, \\
    0, & \text{otherwise.}
    \end{cases}
    \]
    :param P_attr: Attributes of prediction (specify each attribute in a dictionary, etc.)
    :param G_attr: Ground Truth Attributes
    :return: 1 if all attributes match, 0 otherwise
    """

    return 1 if P_attr == G_attr else 0

# Overall similarity of each pair
def sim_ti_hota(P, G, tau):
    """
    Function to calculate the overall similarity of TI-HOTA
    \[
    \operatorname{Sim}_{\mathrm{TI\text{-}HOTA}}(P, G) = \operatorname{LocSim}(P, G) \times \operatorname{IdSim}(P, G)
    \]
    :param P: Prediction Information {'position': [x, y, ...], 'attributes': {...}} 
    :param G: Ground Truth Information {'position': [x, y, ...], 'attributes': {...}} 
    :param tau: normalized parameter for loc_sim calculation
    :return: overall similarity Sim_{TI-HOTA}
    """
    
    return loc_sim(P['position'], G['position'], tau) * id_sim(P['attributes'], G['attributes'])

def compute_ti_hota_alpha(frames, tau, alpha):
    total_tp = total_fp = total_fn = 0
    assoc_counts = {}
    fpa_counts = {}

    for frame in frames:
        gt_objs = frame.get('ground_truths', [])
        pred_objs = frame.get('predictions', [])
        n_gt, n_pred = len(gt_objs), len(pred_objs)
        
        # Build similarity and cost matrices
        combined = np.zeros((n_gt, n_pred))
        cost = np.zeros((n_gt, n_pred))
        for i, gt in enumerate(gt_objs):
            for j, pred in enumerate(pred_objs):
                loc = loc_sim(pred['position'], gt['position'], tau)
                combined[i, j] = loc * id_sim(pred['attributes'], gt['attributes'])
                cost[i, j] = 1 - combined[i, j]

        # Exclude below-threshold pairs
        cost[combined < alpha] = 1e6
        row_ind, col_ind = linear_sum_assignment(cost)

        # Determine current valid matches
        curr_gt_match = {}
        for i, j in zip(row_ind, col_ind):
            if combined[i, j] >= alpha:
                curr_gt_match[i] = j

        # Update detection counts
        tp = len(curr_gt_match)
        fp = n_pred - tp
        fn = n_gt   - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Association counts (for AssA calculation)
        matched_gt = set(curr_gt_match.keys())
        for i, gt in enumerate(gt_objs):
            gid = gt['attributes']
            assoc_counts.setdefault(gid, {'TPA': 0, 'FNA': 0})
            if i not in matched_gt:
                assoc_counts[gid]['FNA'] += 1
        for i, j in curr_gt_match.items():
            gid = gt_objs[i]['attributes']
            assoc_counts[gid]['TPA'] += 1
        for j, pred in enumerate(pred_objs):
            if j not in curr_gt_match.values():
                pid = pred['attributes']
                fpa_counts[pid] = fpa_counts.get(pid, 0) + 1

    # Compute average DetA and AssA
    total_det = total_tp + total_fp + total_fn
    det_acc = total_tp / total_det if total_det > 0 else 0
    sum_A = n_tracks = 0
    for gid, cnt in assoc_counts.items():
        denom = cnt['TPA'] + cnt['FNA'] + fpa_counts.get(gid, 0)
        if denom > 0:
            sum_A += cnt['TPA'] / denom
            n_tracks += 1
    ass_acc = sum_A / n_tracks if n_tracks > 0 else 0

    # TI-HOTA for this alpha
    ti_hota_alpha = math.sqrt(det_acc * ass_acc)
    
    return ti_hota_alpha, det_acc, ass_acc, total_tp, total_fp, total_fn

# The function that calculates TI-HOTA for each threshold value α and takes the mean of the TI-HOTA_α
def compute_ti_hota(frames, tau):
    alphas = np.arange(0.05, 1.0, 0.05)
    ti_hota_alphas = []
    det_acc_values = []
    ass_acc_values = []
    total_tp_values = []
    total_fp_values = []
    total_fn_values = []
    
    for alpha in alphas:
        ti_hota_alpha, det_acc, ass_acc, tp, fp, fn = compute_ti_hota_alpha(frames, tau, alpha)
        ti_hota_alphas.append(ti_hota_alpha)
        det_acc_values.append(det_acc)
        ass_acc_values.append(ass_acc)
        total_tp_values.append(tp)
        total_fp_values.append(fp)
        total_fn_values.append(fn)
        
    ti_hota = np.mean(ti_hota_alphas)
    mean_det_acc = np.mean(det_acc_values)
    mean_ass_acc = np.mean(ass_acc_values)
    mean_total_tp = np.mean(total_tp_values)
    mean_total_fp = np.mean(total_fp_values)
    mean_total_fn = np.mean(total_fn_values)
    
    metrics = {
        "TI-HOTA": ti_hota,
        "TI-HOTA_alphas": ti_hota_alphas,
        "TI-DetA": mean_det_acc,
        "TI-AssA": mean_ass_acc,
        "TI-TP": mean_total_tp,
        "TI-FP": mean_total_fp,
        "TI-FN": mean_total_fn,
    }
    return metrics, alphas

# ### CELL 1 (code)
# Function to read txt files of round truth and prediction results and group them by frames
def load_frames(file_path):
    frames_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 5:
                continue
            # Row contents: frame, ID, x, y, attribute
            frame = int(parts[0])
            id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            attributes = parts[4]
            obj = {'position': [x, y], 'id': id, 'attributes': attributes}
            if frame not in frames_dict:
                frames_dict[frame] = []
            frames_dict[frame].append(obj)
    return frames_dict

# Reads the ground truth file of the same name and the prediction result file and merges them frame by frame
def load_video(gt_file, pred_file):
    gt_frames = load_frames(gt_file)
    pred_frames = load_frames(pred_file)
    all_frame_ids = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))
    frames = []
    for frame_id in all_frame_ids:
        frame_data = {
            'ground_truths': gt_frames.get(frame_id, []),
            'predictions': pred_frames.get(frame_id, [])
        }
        frames.append(frame_data)
    return frames

# ### CELL 2 (code)
import os
import glob
import numpy as np

# --- Indoor Settings ---
gt_folder = '../ground_truth/Indoor/transformed_MOT_with_attributes'
pred_folder = '../output/CAMELTrack_outputs/Indoor/transformed_with_attributes/'
tau = 100

# List of txt files in ground truth folder
gt_files = glob.glob(os.path.join(gt_folder, '*.txt'))

# List to store evaluation results and number of frames for each video
indoor_results = []   # Stores metrics (dictionary) for each video
indoor_frame_counts = []  # Number of frames for each video

for gt_file in gt_files:
    filename = os.path.basename(gt_file)
    pred_file = os.path.join(pred_folder, filename)
    if not os.path.exists(pred_file):
        print(f"Pred result file {pred_file} not found.Skip.")
        continue

    frames = load_video(gt_file, pred_file)
    num_frames = len(frames)
    indoor_frame_counts.append(num_frames)
    
    metrics, alphas = compute_ti_hota(frames, tau)
    indoor_results.append(metrics)
    
    print(f"Video: {filename}")
    print(f"  Number of frames: {num_frames}")
    print(f"  TI-HOTA: {metrics['TI-HOTA']:.4f}")
    print(f"  TI-DetA: {metrics['TI-DetA']:.4f}")
    print(f"  TI-AssA: {metrics['TI-AssA']:.4f}")
    print(f"  TI-TP: {metrics['TI-TP']:.2f}, TI-FP: {metrics['TI-FP']:.2f}, TI-FN: {metrics['TI-FN']:.2f}")
    print("-" * 50)

# Organized into a list by evaluation metrics
keys = ["TI-HOTA", "TI-DetA", "TI-AssA", "TI-TP", "TI-FP", "TI-FN"]
metrics_data = { key: [] for key in keys }
for m in indoor_results:
    for key in keys:
        metrics_data[key].append(m[key])

print(f"===== Indoor Videos: Mean ± SD (τ={tau}) =====")
if indoor_frame_counts:
    frame_avg = np.mean(indoor_frame_counts)
    frame_std = np.std(indoor_frame_counts)
    print(f"Frame Count: {frame_avg:.2f} (std: {frame_std:.2f})")
    
for key in keys:
    if metrics_data[key]:
        avg = np.mean(metrics_data[key])
        std = np.std(metrics_data[key])
        print(f"{key}: {avg:.4f} (std: {std:.4f})")
    else:
        print(f"{key}: No data")


# ### CELL 3 (code)
import os
import glob
import numpy as np

# --- Outdoor Settings ---
# Multiple ground truth folders and categories
gt_folders = [
    ('free_throw', '../ground_truth/Outdoor/MOT_files/split_transformed/free_throw'),
    ('check_ball', '../ground_truth/Outdoor/MOT_files/split_transformed/check_ball')
]
# Folder where the forecast results are stored (all forecast files are in this folder)
pred_folder = '../output/CAMELTrack_outputs/Outdoor/transformed/with_jersey_number/with_team/'
#pred_folder = '../output/CAMELTrack_outputs/Outdoor/transformed_no_merging/with_jersey_number/with_team/'
tau = 100

# 全体用
outdoor_results = []
outdoor_frame_counts = []

# カテゴリ別用
results_by_cat = {name: [] for name, _ in gt_folders}
frames_by_cat = {name: [] for name, _ in gt_folders}

for folder_name, gt_folder in gt_folders:
    gt_files = glob.glob(os.path.join(gt_folder, '*.txt'))
    for gt_file in gt_files:
        filename = os.path.basename(gt_file)
        pred_file = os.path.join(pred_folder, filename)
        if not os.path.exists(pred_file):
            print(f"Pred result file {pred_file} not found. Skip.")
            continue

        # ここは load_video, compute_ti_hota を呼び出す部分
        frames = load_video(gt_file, pred_file)
        num_frames = len(frames)
        metrics, alphas = compute_ti_hota(frames, tau)

        # 全体に追加
        outdoor_frame_counts.append(num_frames)
        outdoor_results.append(metrics)

        # カテゴリ別に追加
        results_by_cat[folder_name].append(metrics)
        frames_by_cat[folder_name].append(num_frames)

        print(f"Video: {filename} (category: {folder_name})")
        print(f"  Number of frames: {num_frames}")
        print(f"  TI-HOTA: {metrics['TI-HOTA']:.4f}")
        print(f"  TI-DetA: {metrics['TI-DetA']:.4f}")
        print(f"  TI-AssA: {metrics['TI-AssA']:.4f}")
        print(f"  TI-TP: {metrics['TI-TP']:.2f}, TI-FP: {metrics['TI-FP']:.2f}, TI-FN: {metrics['TI-FN']:.2f}")
        print("-" * 50)

# -------- 全体集計 --------
keys = ["TI-HOTA", "TI-DetA", "TI-AssA", "TI-TP", "TI-FP", "TI-FN"]
print(f"===== Outdoor Videos: Overall Average ± SD (τ={tau}) =====")
if outdoor_frame_counts:
    print(f"Frame Count: {np.mean(outdoor_frame_counts):.2f} (std: {np.std(outdoor_frame_counts):.2f})")
for key in keys:
    vals = [m[key] for m in outdoor_results]
    if vals:
        print(f"{key}: {np.mean(vals):.4f} (std: {np.std(vals):.4f})")
    else:
        print(f"{key}: No data")

# -------- カテゴリ別集計 --------
print("\n===== Outdoor Videos: Category-wise Average ± SD =====")
for cat in results_by_cat:
    cat_results = results_by_cat[cat]
    cat_frames = frames_by_cat[cat]
    print(f"\n-- {cat} --")
    if cat_frames:
        print(f"Frame Count: {np.mean(cat_frames):.2f} (std: {np.std(cat_frames):.2f})")
    else:
        print("Frame Count: No data")
    for key in keys:
        vals = [m[key] for m in cat_results]
        if vals:
            print(f"{key}: {np.mean(vals):.4f} (std: {np.std(vals):.4f})")
        else:
            print(f"{key}: No data")

# ### CELL 4 (code)

