import os
import pandas as pd
import json
import yaml
from matplotlib import pyplot as plt

DEFAULT_REL_PATH = os.path.join(os.curdir, 'runs', 'train')
DEFAULT_EXP_PREFIX = 'exp'

def read_exp_results(exp_no, exp_prefix=DEFAULT_EXP_PREFIX, rel_path=DEFAULT_REL_PATH):
    exp_name = f'{exp_prefix}{exp_no}'
    exp_path = os.path.join(rel_path, exp_name)
    res_file_name = os.path.join(exp_path, 'results.csv')
    results_df = pd.read_csv(res_file_name)
    pred = json.load(open(os.path.join(exp_path, '_predictions.json'))) #todo
    hyp = yaml.safe_load(open(os.path.join(exp_path, 'hyp.yaml'))) #todo
    opt = yaml.safe_load(open(os.path.join(exp_path, 'opt.yaml'))) #todo
    cm_path = os.path.join(exp_path, 'confusion_matrix_abs_0.25.png')
    cm = plt.imread(cm_path)
    return {'results': results_df, 'hyp': hyp, 'opt': opt, 'pred': pred, 'cm': cm}

def get_im_vol_list(target_volunteers, num_images):
    im_vol_list = []
    for i in range(num_images):
        x = target_volunteers[target_volunteers[:,0] == i, -1].int().detach().numpy()
        num_boxes = x.shape[0]
        vol_per_im = []
        for j in range(num_boxes):
            vol_per_im.append(x[j])
        im_vol_list.append(vol_per_im)
    return im_vol_list