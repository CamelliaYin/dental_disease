
from shutil import copyfile
from tqdm.notebook import tqdm
import numpy as np
import math
import shutil
import os

SINGLETOY_YOLO_MODE =   {'sz': 'to', 'ml': 'si',  'tr': 'ner', 'va': 'ner', 'te': 'er'}
SINGLEFULL_YOLO_MODE =  {'sz': 'fu', 'ml': 'si',  'tr': 'ner', 'va': 'ner', 'te': 'er'}
ALLTOY_YOLO_MODE =      {'sz': 'to', 'ml': 'all', 'tr': 'ner', 'va': 'ner', 'te': 'er'}
ALLFULL_YOLO_MODE =     {'sz': 'fu', 'ml': 'all', 'tr': 'ner', 'va': 'ner', 'te': 'er'}

SINGLETOY_CYOLO_MODE =  {'sz': 'to', 'ml': 'si',  'tr': 'nec', 'va': 'ner', 'te': 'er'}
SINGLEFULL_CYOLO_MODE = {'sz': 'fu', 'ml': 'si',  'tr': 'nec', 'va': 'ner', 'te': 'er'}
ALLTOY_CYOLO_MODE =     {'sz': 'to', 'ml': 'all', 'tr': 'nec', 'va': 'ner', 'te': 'er'}
ALLFULL_CYOLO_MODE =    {'sz': 'fu', 'ml': 'all', 'tr': 'nec', 'va': 'ner', 'te': 'er'}

DATA_MODES = {'sty': SINGLETOY_YOLO_MODE,
              'sfy': SINGLEFULL_YOLO_MODE,
              'aty': ALLTOY_YOLO_MODE,
              'afy': ALLFULL_YOLO_MODE,
              'stcy': SINGLETOY_CYOLO_MODE,
              'sfcy': SINGLEFULL_CYOLO_MODE,
              'atcy': ALLTOY_CYOLO_MODE,
              'afcy': ALLFULL_CYOLO_MODE}


def get_data_mode_textual_name(mode='aty'):
    data_mode = DATA_MODES[mode]
    w1 = 'single' if data_mode['ml'] == 'si' else 'all'
    w2 = 'toy' if data_mode['sz'] == 'to' else 'full'
    w3 = 'yolo' if data_mode['tr'] == 'ner' else 'cyolo'
    assert data_mode['tr'] in ['ner', 'nec'] and \
           data_mode['va'] == 'ner' and \
           data_mode['te'] == 'er', "Undefined textual name for this data mode."
    textual_name = f'{w1}{w2}_{w3}'
    return textual_name


def get_data_mode_technical_name(mode='aty'):
    data_mode = DATA_MODES[mode]
    key_seq = ['sz', 'ml', 'tr', 'va', 'te']
    technical_name = '.'.join([f'{x}_{data_mode[x]}' for x in key_seq])
    return technical_name


def get_split_ratios(*args):
    if len(args) == 1:
        train = args[0]
        test = 1.0 - train
        val = 0.0
    elif len(args) == 2:
        train, test = args
        val = 1.0 - (train+test)
    elif len(args) == 3:
        train, val, test = args
    assert all([0 <= x <= 1 for x in [train, val, test]]), "split ratios should be between 0 and 1"
    assert train+val+test, "split ratios should add up to 1"
    return round(train, 2), round(val, 2), round(test, 2)


def get_split_ids(n, ratios, seed=None):
    if seed is not None:
        np.random.seed(seed)
    ids = np.arange(n)
    np.random.shuffle(ids)
    n_train = min(math.ceil(n * ratios[0]), n)
    n_val = min(math.ceil(n * ratios[1]), n)
    n_test = n - n_train - n_val
    train, val, test = ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]
    return train, val, test

def prepare_data(mode, train_ratios, data_path):
    data_mode = DATA_MODES[mode]
    toy_mode = data_mode['sz'] == 'to'
    single_mode = data_mode['ml'] == 'si'
    expert_modes = [data_mode[x].startswith('e') for x in ['tr', 'va', 'te']]
    crowdsourced_modes = [data_mode[x].endswith('c') for x in ['tr', 'va', 'te']]
    
    MASTER_NAME = 'master'
    SEED = 0 # Set as None for no seed.
    EXPERT_NAMES = {'Jonathan'}
    SINGLE_VOLUNTEER_NAME = 'Camellia'
    NUM_TOY = 10

    IID_NAME = 'iid'
    BCC_NAME = 'bcc'
    
    TOY_MODE = toy_mode
    
    SINGLE_MODE = single_mode

    data_home_path = data_path
    tr = train_ratios

    SUFFIX = None
    if tr[0]:
        SUFFIX = 't'
        if tr[1]:
            SUFFIX += 'v'
        if tr[2]:
            SUFFIX += 't'

    if SUFFIX is not None:
        IID_NAME = IID_NAME + '-' + SUFFIX
        BCC_NAME = BCC_NAME + '-' + SUFFIX

    silent = True
    # def prepare_data(data_home_path, tr):
    master_data_path = os.path.join(data_home_path, MASTER_NAME)
    im_path = os.path.join(master_data_path, 'images')
    la_path = os.path.join(master_data_path, 'labels')
    users = list(sorted([x for x in os.listdir(la_path) if (not x.startswith('.'))]))
    experts = [x for x in users if x in EXPERT_NAMES]
    assert len(experts) == 1, 'Not implemented for more than one experts'
    volunteers = [x for x in users if x not in EXPERT_NAMES]
    if SINGLE_MODE:
        volunteers = [SINGLE_VOLUNTEER_NAME]
        if SINGLE_VOLUNTEER_NAME not in volunteers:
            raise Exception('Choose a different single volunteer.')
    im_name_exts = list(sorted([os.path.splitext(x) for x in os.listdir(im_path) if (not x.startswith('.'))]))

    if TOY_MODE:
        im_name_exts = im_name_exts[:NUM_TOY]

    n_images = len(im_name_exts)
    if isinstance(tr, float):
        tr = (tr,)
    tr_ids, val_ids, te_ids = get_split_ids(n_images, get_split_ratios(*tr), seed=SEED)

    tr_im_name_exts = [im_name_exts[i] for i in tr_ids]
    val_im_name_exts = [im_name_exts[i] for i in val_ids]
    te_im_name_exts = [im_name_exts[i] for i in te_ids]

    mode_metadata_dict = {'train': tr_im_name_exts,
                        'val': val_im_name_exts,
                        'test': te_im_name_exts}



    data_name = IID_NAME if not TOY_MODE else f'toy_{IID_NAME}'
    if SINGLE_MODE:
        data_name = 'single_'+data_name
        
    iid_data_path = os.path.join(data_home_path, data_name)
    if os.path.exists(iid_data_path):
        raise Exception('Path already exists. Not recreating data')
        shutil.rmtree(iid_data_path)
    os.mkdir(iid_data_path)
    iid_im_path = os.path.join(iid_data_path, 'images')
    if not os.path.exists(iid_im_path):
        os.mkdir(iid_im_path)
    iid_la_path = os.path.join(iid_data_path, 'labels')
    if not os.path.exists(iid_la_path):
        os.mkdir(iid_la_path)

    for mode, metadata in mode_metadata_dict.items():
        if mode == 'train':
            labellers = volunteers
        elif mode == 'val':
            if SUFFIX == 'tt':
                continue
            labellers = experts if SUFFIX == 'tv' else volunteers
        elif mode == 'test':
            if SUFFIX == 'tv':
                continue
            labellers = experts
        else:
            raise Exception('Mode not identified.')
        iid_im_mode_dir_path = os.path.join(iid_im_path, mode)
        iid_la_mode_dir_path = os.path.join(iid_la_path, mode)
        if not os.path.exists(iid_im_mode_dir_path):
            os.mkdir(iid_im_mode_dir_path)
        if not os.path.exists(iid_la_mode_dir_path):
            os.mkdir(iid_la_mode_dir_path)
        for name, im_ext in tqdm(metadata):
            im_file_name = f'{name}{im_ext}'
            la_file_name = f'{name}.txt'
            im_file_path = os.path.join(im_path, im_file_name)
            for v in labellers:
                iid_name = f'{name}.{v}'
                iid_im_file_name = f'{iid_name}{im_ext}'
                iid_im_file_path = os.path.join(iid_im_mode_dir_path, iid_im_file_name)
                if not silent:
                    print(f'Copying {im_file_path} --> {iid_im_file_path}')
                copyfile(im_file_path, iid_im_file_path)

                iid_la_file_name = f'{iid_name}.txt'
                iid_la_file_path = os.path.join(iid_la_mode_dir_path, iid_la_file_name)
                la_file_path = os.path.join(la_path, v, la_file_name)
                if not os.path.exists(la_file_path):
                    continue
                if not silent:
                    print(f'Copying {la_file_path} --> {iid_la_file_path}')
                copyfile(la_file_path, iid_la_file_path)



    data_name = BCC_NAME if not TOY_MODE else f"toy_{BCC_NAME}"
    if SINGLE_MODE:
        data_name = 'single_'+data_name
    bcc_data_path = os.path.join(data_home_path, data_name)
    if os.path.exists(bcc_data_path):
        raise Exception('Path already exists. Not recreating data.')
        shutil.rmtree(bcc_data_path)
    os.mkdir(bcc_data_path)
    bcc_im_path = os.path.join(bcc_data_path, 'images')
    if not os.path.exists(bcc_im_path):
        os.mkdir(bcc_im_path)
    bcc_la_path = os.path.join(bcc_data_path, 'labels')
    if not os.path.exists(bcc_la_path):
        os.mkdir(bcc_la_path)
    bcc_vol_path = os.path.join(bcc_data_path, 'volunteers')
    if not os.path.exists(bcc_vol_path):
        os.mkdir(bcc_vol_path)

    for mode, metadata in mode_metadata_dict.items():
        if mode == 'train':
            labellers = volunteers
        elif mode == 'val':
            if SUFFIX == 'tt':
                continue
            labellers = experts if SUFFIX == 'tv' else volunteers
        elif mode == 'test':
            if SUFFIX == 'tv':
                continue
            labellers = experts
        else:
            raise Exception('Mode not identified.')
        bcc_im_mode_dir_path = os.path.join(bcc_im_path, mode)
        bcc_la_mode_dir_path = os.path.join(bcc_la_path, mode)
        bcc_vol_mode_dir_path = os.path.join(bcc_vol_path, mode)
        if not os.path.exists(bcc_im_mode_dir_path):
            os.mkdir(bcc_im_mode_dir_path)
        if not os.path.exists(bcc_la_mode_dir_path):
            os.mkdir(bcc_la_mode_dir_path)
        if not os.path.exists(bcc_vol_mode_dir_path):
            os.mkdir(bcc_vol_mode_dir_path)
        for name, im_ext in tqdm(metadata):
            im_file_name = f'{name}{im_ext}'
            la_file_name = f'{name}.txt'
            im_file_path = os.path.join(im_path, im_file_name)
            bcc_name = name
            bcc_im_file_name = f'{bcc_name}{im_ext}'
            bcc_im_file_path = os.path.join(bcc_im_mode_dir_path, bcc_im_file_name)
            if not silent:
                print(f'Copying {im_file_path} --> {bcc_im_file_path}')
            copyfile(im_file_path, bcc_im_file_path)
            bcc_la_file_name = f'{bcc_name}.txt'
            bcc_la_file_path = os.path.join(bcc_la_mode_dir_path, bcc_la_file_name)
            lab_list = []
            with open(bcc_la_file_path, 'w') as bcc_file:
                for lab in labellers:
                    lab_la_file_path = os.path.join(la_path, lab, la_file_name)
                    try:
                        with open(lab_la_file_path) as lab_file:
                            lines = lab_file.readlines()
                            bcc_file.write(''.join(lines))
                            lab_list.extend([lab]*len(lines))
                    except FileNotFoundError:
                        print(f'File {lab_la_file_path} not found.')
            bcc_vol_file_name = f'{name}.txt'
            bcc_vol_file_path = os.path.join(bcc_vol_mode_dir_path, bcc_vol_file_name)
    #         print(lab_list)
            with open(bcc_vol_file_path, 'w') as bcc_vol_file:
                bcc_vol_file.write('\n'.join(lab_list) + ('\n' if len(lab_list) > 0 else ''))


if __name__ == '__main__':
    prepare_data()
