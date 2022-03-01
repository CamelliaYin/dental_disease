import os
import sys
import json
import yaml
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatch
import matplotlib

sys.path.append('../src/cyolo_utils')

# the following results are produced by testtestLARGE100Min3
# given path: /Users/camellia/Desktop/OneDrive_Surrey/dental_disease/notebooks/runs/train/testtestLARGE100Min3
# working directory: /Users/camellia/Desktop/OneDrive_Surrey/dental_disease/notebooks

DEFAULT_REL_PATH = os.path.join(os.curdir, 'runs', 'train')  # './runs/train'
DEFAULT_EXP_PREFIX = 'exp'
rel_path = os.path.join(os.curdir, 'runs', 'train')
metadata = {'baseline': {'name': 'yolo', 'exp_id': 'testtestLARGE100Min3'},
            'contribution': {'name': 'cyolo', 'exp_id': 'testtestLARGE100Min3'}}


# todo: change the id to the baseline and contribution accordingly


# input: the 'file name', rel_path from working directory to the file name
def read_exp_results(exp_name, rel_path='runs/train'):
    exp_path = os.path.join(rel_path, exp_name)
    res_file_name = os.path.join(exp_path, 'results.csv')
    results_df = pd.read_csv(res_file_name)
    pred = json.load(open(os.path.join(exp_path, '_predictions.json')))
    hyp = yaml.safe_load(open(os.path.join(exp_path, 'hyp.yaml')))
    opt = yaml.safe_load(open(os.path.join(exp_path, 'opt.yaml')))
    cm_path = os.path.join(exp_path, 'confusion_matrix_abs_0.25.png')
    cm = plt.imread(cm_path)
    return {'results': results_df, 'hyp': hyp, 'opt': opt, 'pred': pred, 'cm': cm}  # a dict



# call preds from the output of read_exp_results()
def get_im_boxes_map(preds):
    im_boxes_map = defaultdict(list)
    for pred in tqdm(preds):
        im_id = pred['image_id']
        im_boxes_map[im_id].append({k: v for k, v in pred.items() if k != 'image_id'})
    im_boxes_map = dict(im_boxes_map)
    im_names = list(sorted(im_boxes_map))
    return im_boxes_map, im_names


# should read both yolo and cyolo together but not now
# read exp for cyolo
exp_no = metadata['contribution']['exp_id']
exp_results = read_exp_results('testtestLARGE100Min3')
#exp_results.keys()  # todo: checking
cyolo_preds = exp_results['pred']
cyolo_im_boxes_map, cyolo_im_names = get_im_boxes_map(cyolo_preds)


# path extract
data_yaml_filename = exp_results['opt']['data']
data_dict = yaml.safe_load(open(data_yaml_filename, 'r'))
data_path = data_dict['path']
train_path = os.path.join(data_path, data_dict['train'])
val_path = os.path.join(data_path, data_dict['val'])
test_path = os.path.join(data_path, data_dict['test'])
#data_dict  # todo: checking
class_names = data_dict['names']
class_name_abbr_map = {'bone-loss': 'BL', 'dental-caries': 'DC'}  # todo: other classes add here

# read exp for yolo
# exp_no = metadata['baseline']['exp_id']
# exp_results = read_exp_results(exp_no, rel_path = rel_path)
# yolo_preds = exp_results['pred']
# yolo_im_boxes_map, yolo_im_names = get_im_boxes_map(yolo_preds)


# input: path should be directly goes to ...train/val/expert
# dental_disease/data/datasets/data_name/volunteers/train
extract_path = '../data/datasets/All_Volunteers_Calc_Removed_100min_test_Crowdsourced/volunteers/train'
# todo: !! can be modified based on data_path

def extract_names(extract_path=extract_path):
    names = []
    for filename in os.listdir(extract_path):
        with open(os.path.join(extract_path, filename), 'r') as f:
            name = f.read()
            name = name.split('\n')
            for item in name:
                if item != '':
                    names.append(item)
    result = []
    for item in names:
        if item not in result:
            result.append(item)

    return result

data_file = 'All_Volunteers_Calc_Removed_100min_test_Crowdsourced'
vol_names = extract_names(extract_path)
vol_name_id_map = dict(zip(vol_names, range(1, 1 + len(vol_names))))
im_prefs = list(sorted({x['image_id'].split('.')[0] for x in cyolo_preds}))
vol_entries = []
for v in vol_names:
    labels_path = os.path.join(data_dict['path'], 'All_Volunteers_Calc_Removed_100min_test_Crowdsourced', 'labels', v)
    for pref in im_prefs:
        label_file_path = os.path.join(labels_path, pref + '.txt')
        try:
            with open(label_file_path, 'r') as f:
                lines = [tuple(map(lambda x: round(float(x), 3), x.strip().split())) for x in f.readlines()]
            for c, x, y, w, h in lines:
                c = int(c)
                entry = {'image_id': pref,
                         'category_id': c,
                         'bbox': [x, y, w, h],
                         'volunteer': v}
                vol_entries.append(entry)
        except FileNotFoundError:
            print(label_file_path, 'not found')
            entry = {'image_id': pref,
                     'category_id': None,
                     'bbox': None,
                     'volunteer': v}
            vol_entries.append(entry)

vol_im_boxes_map, vol_im_names = get_im_boxes_map(vol_entries)


def get_im_boxes(im_name, mode):
    if mode == 'y':
        return yolo_im_boxes_map[im_name]
    elif mode == 'cy':
        return cyolo_im_boxes_map[im_name]
    elif mode == 'v':
        return vol_im_boxes_map[im_name.split('.')[0]]
    elif mode == 'e':
        return exp_im_boxes_map[im_name.split('.')[0]]


def plot_predboxes_per_radiograph(im_filepath, im_boxes, ax, conf_thres, mode):
    img = mpimg.imread(im_filepath)
    ht, wd, _ = img.shape

    ax.axis('off')
    img_plot = ax.imshow(img)

    box_linewidth = 2
    box_color_map = {'bone-loss': 'r', 'dental-caries': 'y'}
    use_abbr_labels = True
    text_on_box_color = 'w'
    text_on_box_fontsize = 10
    text_on_box_ha = 'left'
    text_on_box_va = 'center'

    for im_box in im_boxes:
        box_class = im_box['category_id']
        bbox = im_box['bbox']
        if box_class is None or bbox is None:
            print(im_filepath, 'has not been tagged by volunteer', im_box['volunteer'], '. Ignoring...')
            continue
        score_mode = True
        vol_mode = False
        try:
            box_score = im_box['score']
        except KeyError:
            score_mode = False
        try:
            vol_name = im_box['volunteer']
            vol_id = vol_name_id_map[vol_name]
            vol_mode = True
        except KeyError:
            pass
        text_on_box = class_names[box_class]
        if score_mode:
            if box_score < conf_thres[text_on_box]:
                continue
        box_color = box_color_map[text_on_box]
        if use_abbr_labels:
            text_on_box = class_name_abbr_map[text_on_box]
        if score_mode:
            text_on_box = f'{text_on_box} ({int(box_score * 100)}%)'
        if vol_mode:
            text_on_box = f'{text_on_box} (V{vol_id})'
        x, y, w, h = bbox
        if mode in {'v', 'e'}:
            x, y, w, h = x * wd, y * ht, w * wd, h * ht
            px, py, pw, ph = x - w / 2, y - h / 2, w, h
        else:
            px, py, pw, ph = x, y, w, h
        box = mpatch.Rectangle((px, py), pw, ph, linewidth=box_linewidth, edgecolor=box_color, facecolor='none')
        ax.add_artist(box)
        _ = ax.annotate(text_on_box, (px + w / 2, py - 2 * text_on_box_fontsize),
                        backgroundcolor=box_color,
                        color=text_on_box_color, weight='bold',
                        fontsize=text_on_box_fontsize,
                        ha=text_on_box_ha,
                        va=text_on_box_va)


def extract_names(dir=dir):
    names = []
    for filename in os.listdir(dir):
        with open(os.path.join(dir, filename), 'r') as f:
            name = f.read()
            name = name.split('\n')
            for item in name:
                if item != '':
                    names.append(item)
    result = []
    for item in names:
        if item not in result:
            result.append(item)

    return result


exp_results = read_exp_results('testtestLARGE100Min3')

data_yaml_filename = exp_results['opt']['data']
data_dict = yaml.safe_load(open(data_yaml_filename, 'r'))
data_path = data_dict['path']
train_path = os.path.join(data_path, data_dict['train'])
val_path = os.path.join(data_path, data_dict['val'])
test_path = os.path.join(data_path, data_dict['test'])

class_names = data_dict['names']
class_name_abbr_map = {'bone-loss': 'BL', 'dental-caries': 'DC'}

metadata = {'baseline': {'name': 'yolo', 'exp_id': 'testtestLARGE100Min3'},
            'contribution': {'name': 'cyolo', 'exp_id': 'testtestLARGE100Min3'}}

exp_no = metadata['baseline']['exp_id']
yolo_preds = exp_results['pred']
yolo_im_boxes_map, yolo_im_names = get_im_boxes_map(yolo_preds)

exp_no = metadata['contribution']['exp_id']
cyolo_preds = exp_results['pred']
cyolo_im_boxes_map, cyolo_im_names = get_im_boxes_map(cyolo_preds)

# volunteer mapping
vol_names = extract_names(dir=extract_path)
vol_name_id_map = dict(zip(vol_names, range(1, 1 + len(vol_names))))
im_prefs = list(sorted({x['image_id'].split('.')[0] for x in yolo_preds}))
vol_entries = []
for v in vol_names:
    labels_path = os.path.join(data_dict['path'], 'master', 'labels', v)
    for pref in im_prefs:
        label_file_path = os.path.join(labels_path, pref + '.txt')
        try:
            with open(label_file_path, 'r') as f:
                lines = [tuple(map(lambda x: round(float(x), 3), x.strip().split())) for x in f.readlines()]
            for c, x, y, w, h in lines:
                c = int(c)
                entry = {'image_id': pref,
                         'category_id': c,
                         'bbox': [x, y, w, h],
                         'volunteer': v}
                vol_entries.append(entry)
        except FileNotFoundError:
            print(label_file_path, 'not found')
            entry = {'image_id': pref,
                     'category_id': None,
                     'bbox': None,
                     'volunteer': v}
            vol_entries.append(entry)

vol_im_boxes_map, vol_im_names = get_im_boxes_map(vol_entries)

# expert mapping
exp_name = 'expert'
im_prefs = list(sorted({x['image_id'].split('.')[0] for x in yolo_preds}))
exp_entries = []
labels_path = os.path.join(data_dict['path'], 'master', 'labels', exp_name)
for pref in im_prefs:
    label_file_path = os.path.join(labels_path, pref + '.txt')
    try:
        with open(label_file_path, 'r') as f:
            lines = [tuple(map(lambda x: round(float(x), 3), x.strip().split())) for x in f.readlines()]
        for c, x, y, w, h in lines:
            c = int(c)
            entry = {'image_id': pref,
                     'category_id': c,
                     'bbox': [x, y, w, h]}
            exp_entries.append(entry)
    except FileNotFoundError:
        print(label_file_path, 'not found')
        entry = {'image_id': pref,
                 'category_id': None,
                 'bbox': None}
        exp_entries.append(entry)
exp_im_boxes_map, exp_im_names = get_im_boxes_map(exp_entries)

# extract image shape
pref = list(exp_im_boxes_map)[0]  # Or select your image of choice
boxes = exp_im_boxes_map[pref]
im_filepath = os.path.join(data_dict['path'], data_dict['val'], pref + '.Camellia.jpg')
img = mpimg.imread(im_filepath)
ht, wd, _ = img.shape

# Change this to select the images you need in the plot. Note that there are no suffixes.
image_prefixes_to_plot = ['IS20180614_161235_0835_000000CD',
                          'IS20190605_100053_0811_00001CCE']

scale = 15

radiographs_toplot_list = []
modes = ['v', 'e', 'y', 'cy']
mode_names = {'v': 'Volunteers',
              'e': 'Expert',
              'y': 'YOLO',
              'cy': 'CYOLO'}
default_vol_name = 'Camellia'
try:
    im_ids = [yolo_im_names.index(f'{i}.{default_vol_name}') for i in image_prefixes_to_plot]
except ValueError:
    paths = [os.path.join(val_path, f'{i}.{default_vol_name}.jpg') for i in image_prefixes_to_plot]
    raise Exception(
        f"Some image wasn't found in the IID validation dataset for YOLO. Ensure all the following paths have images:\n" + '\n'.join(
            paths))
for mode in modes:
    list_per_mode = []
    for im_id in im_ids:
        im_name = yolo_im_names[im_id]
        im_filename = f'{im_name}.jpg'
        im_filepath = os.path.join(val_path, im_filename)
        im_boxes = get_im_boxes(im_name, mode)
        list_per_mode.append((im_filepath, im_boxes))
    radiographs_toplot_list.append(list_per_mode)

# plotting:
inner_grid = (1, 2)
conf_thres = {'bone-loss': 0.25, 'dental-caries': 0.05}

for m, mode in enumerate(modes):
    fig, axs = plt.subplots(*inner_grid, figsize=(1.25 * scale, scale / 2))
    # subfigs = fig.subfigures(*outer_grid)
    fig.suptitle(f'{mode_names[mode]}')
    fig.subplots_adjust(wspace=0, hspace=0)
    for innerind, ax in enumerate(axs.flat):
        im_filepath, im_boxes = radiographs_toplot_list[m][innerind]
        plot_predboxes_per_radiograph(im_filepath, im_boxes, ax, conf_thres, mode)

# Annotating a custom image

im_data_path = '../data/datasets/master/images/'
la_data_path = '../data/datasets/master/labels/Jonathan'
im_names = os.listdir(im_data_path)
im_prefs = [x.rsplit('.', 1)[0] for x in im_names]
for i in [10]:
    im_pref = im_prefs[i]
    print(im_pref)

    im_filepath = os.path.join(im_data_path, im_pref + '.jpg')
    img = mpimg.imread(im_filepath)
    ht, wd, _ = img.shape
    print(wd, ht)

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.axis('off')
    img_plot = ax.imshow(img)

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.axis('off')
    img_plot = ax.imshow(img)
    la_filepath = os.path.join(la_data_path, im_pref + '.txt')
    with open(la_filepath, 'r') as f:
        labels = [tuple(map(float, x.strip().split())) for x in f.readlines()]
    for label in labels:
        c = class_names[int(label[0])]
        x = label[1] * wd
        y = label[2] * ht
        w = label[3] * wd
        h = label[4] * ht
        c, x, y, w, h
        px, py, pw, ph = x - w / 2, y - h / 2, w, h
        box = mpatch.Rectangle((px, py), pw, ph, linewidth=4, edgecolor='r', facecolor='none')
        ax.add_artist(box)
        text_on_box = c
        _ = ax.annotate(text_on_box, (px + 2 * w, py - 30),
                        backgroundcolor='r',
                        color='w', weight='bold',
                        fontsize=20,
                        ha='right',
                        va='center')
