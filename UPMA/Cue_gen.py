import os
import cv2
import numpy as np

# ==== PATHS ====
Pred_Map_path = '/path/to/SAM_mask_dir'
GT_path = '/path/to/GroundTruth_dir'#Ground truth is only used to create one blank image.
save_path = '/path/to/save/two_cue'  # Output directory

os.makedirs(save_path, exist_ok=True)
file_list = sorted(os.listdir(Pred_Map_path))

#cue_size
cue_radius = 8

for file in file_list:
    GT = cv2.imread(os.path.join(GT_path, file), cv2.IMREAD_GRAYSCALE)
    SAM_mask = cv2.imread(os.path.join(Pred_Map_path, file), cv2.IMREAD_GRAYSCALE)

    if GT is None or SAM_mask is None:
        print(f"[Skipped] {file}")
        continue

    H, W = GT.shape # We are using GT just here to take the shape only
    GT_cue = np.zeros((H, W), dtype=np.uint8)

    SAM_bin = np.zeros_like(SAM_mask)
    SAM_bin[SAM_mask > 150] = 1

    fg_mask = SAM_bin.copy()
    bg_mask = 1 - fg_mask

    # Foreground cue (1 point)
    P_fg = fg_mask.argmax()
    y_fg, x_fg = P_fg // W, P_fg % W
    cv2.circle(GT_cue, (x_fg, y_fg), cue_radius, 1, -1)

    # Background cue (1 point)
    P_bg = bg_mask.argmax()
    y_bg, x_bg = P_bg // W, P_bg % W
    cv2.circle(GT_cue, (x_bg, y_bg), cue_radius, 2, -1)
    cv2.imwrite(os.path.join(save_path, file), GT_cue)

print("Finished")
