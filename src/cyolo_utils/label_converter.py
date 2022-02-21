import pdb

import numpy as np
import torch
from collections import defaultdict

MAX_GRID_CELLS = 99999
BACKGROUND_CLASS_ID = 2
BKGD_WH_IS_MEAN = False

def targetize(z):
    targeted_y = []
    for i, e in enumerate(z):
        targeted_y.append(torch.cat([i*torch.ones(e.shape[0], 1), e], 1))
    return targeted_y


def find_containing_grid_cell(grid_x_frac, grid_y_frac, x, y, mode='row-major'):
    if grid_x_frac == 0 or grid_y_frac == 0:
        raise Exception(
            "Cannot divide into infinite grids. Ensure grid fractions are > 0.")
    n_x_cells, n_y_cells = np.ceil(1 / grid_x_frac).astype(int), np.ceil(1 / grid_y_frac).astype(int)
    if n_x_cells*n_y_cells > MAX_GRID_CELLS:
        raise Exception("Grid cells maxed out.")
    x_cell_id = np.floor(x/grid_x_frac).astype(int) if x < 1 else n_x_cells-1
    y_cell_id = np.floor(y/grid_y_frac).astype(int) if y < 1 else n_y_cells-1
    if mode.startswith('row'):
        return (y_cell_id)*n_x_cells + x_cell_id
    elif mode.startswith('col'):
        return (x_cell_id)*n_y_cells + y_cell_id
    else:
        raise Exception("Incorrect mode for finding grid cell.")


def test_find_containing_grid_cell():
    # Should throw exceptions in all the following cases:
    # assert find_containing_grid_cell(0, 0, 0, 0) == 0
    # assert find_containing_grid_cell(1, 0, 0, 0) == 0
    # assert find_containing_grid_cell(0, 1, 0, 0) == 0
    # assert find_containing_grid_cell(0.001, 0.001, 0, 0) == 0
    assert find_containing_grid_cell(0.01, 0.01, 0, 0) == 0
    assert find_containing_grid_cell(0.01, 0.01, 1, 1) == 9999

    assert find_containing_grid_cell(0.5, 0.5, 0, 0) == 0
    assert find_containing_grid_cell(0.5, 0.5, 0, 1) == 2
    assert find_containing_grid_cell(0.5, 0.5, 1, 0) == 1
    assert find_containing_grid_cell(0.5, 0.5, 1, 1) == 3
    assert find_containing_grid_cell(0.5, 0.5, 0, 0.5) == 2
    assert find_containing_grid_cell(0.5, 0.5, 0.5, 0) == 1
    assert find_containing_grid_cell(0.5, 0.5, 0.5, 0.5) == 3
    assert find_containing_grid_cell(0.5, 0.5, 0, 0.499) == 0
    assert find_containing_grid_cell(0.5, 0.5, 0.499, 0) == 0
    assert find_containing_grid_cell(0.5, 0.5, 0.499, 0.499) == 0


def find_grid_center(grid_x_frac, grid_y_frac, gc, mode='row-major'):
    if grid_x_frac == 0 or grid_y_frac == 0:
        raise Exception(
            "Cannot divide into infinite grids. Ensure grid fractions are > 0.")
    n_x_cells, n_y_cells = np.ceil(1 / grid_x_frac).astype(int), np.ceil(1 / grid_y_frac).astype(int)
    if n_x_cells*n_y_cells > MAX_GRID_CELLS:
        raise Exception("Grid cells maxed out.")
    if mode.startswith('row'):
        x_cell_id, y_cell_id = gc % n_x_cells, np.floor(gc / n_x_cells).astype(int)
    elif mode.startswith('col'):
        x_cell_id, y_cell_id = np.floor(gc / n_y_cells).astype(int), gc % n_y_cells
    else:
        raise Exception("Incorrect mode for finding grid center.")
    x = x_cell_id*grid_x_frac + grid_x_frac/2
    y = y_cell_id*grid_y_frac + grid_y_frac/2
    return x, y


def test_find_grid_center():
    assert find_grid_center(0.01, 0.01, 0) == (0.005, 0.005)
    assert find_grid_center(0.01, 0.01, 9999) == (1-0.005, 1-.005)

    assert find_grid_center(0.5, 0.5, 0) == (0.25, 0.25)
    assert find_grid_center(0.5, 0.5, 1) == (0.75, 0.25)
    assert find_grid_center(0.5, 0.5, 2) == (0.25, 0.75)
    assert find_grid_center(0.5, 0.5, 3) == (0.75, 0.75)

def find_union_cstargets(cstargets):
    n_images = len(cstargets[0])
    y_cs_union = []
    for i in range(n_images):
        y_cs_union.append(torch.cat([torch.tensor(x[i]) for x in cstargets]))
    return y_cs_union

# When YOLO predicts bounding boxes in a given epoch, this has to be fed to BCC,
# which is only possible if the dimensions match. While YOLO gives a boundingbox-class tensor (x, y, w, h, c, c1, c2),
# what BCC needs is a gridcell-class tensor, and that too in logits.
def yolo2bcc_newer(y_yolo, imgsz, silent = True):
    # transform probability from o,c1,c2 to p0,p1,p2
    wh = y_yolo[:, ..., 2:4] / imgsz  # width and height
    conf = y_yolo[:, ..., 4]  # first class C1

    p = y_yolo[..., 4]  # also first class C1?
    sigma_t = y_yolo[..., 5:]  # the rest of the classes C2, C3, ...
    sigma_prime_t = sigma_t / sigma_t.sum(axis=2).unsqueeze(-1)  # divides the C2, C3,... by the sum of itself for each instance
    class_prob = sigma_prime_t * p.unsqueeze(-1)  # multiplies C2, C3, ... by the first class prob C1
    bkgd_prob = 1 - p  # inverts the probabilities for the first class C1
    # single_bcc_toy.yaml, first epoch, min bkgd_prob is 0.91 which is no good. bg is dominated
    bcc_prob = torch.cat([class_prob, bkgd_prob.unsqueeze(-1)],-1)  # concatanates C1 to new C2, C3, ... in the form of C2, C3, ..., C1
    if not silent:
        mins = [round(x, 6) for x in list(bcc_prob.min(1).values.min(0).values.cpu().detach().numpy())]
        maxs = [round(x, 6) for x in list(bcc_prob.max(1).values.max(0).values.cpu().detach().numpy())]
        print('Minimum probs (c1, c2, bkgd):', mins)
        print('Maximum probs (c1, c2, bkgd):', maxs)
    bcc_logits = torch.log(bcc_prob)  # finds the logarithm of each element.
    # note that they are log normalised probs from yolo output, i.e. log(transformprob)
    return bcc_logits, wh, conf

def yolo2bcc_new(y_yolo, imgsz):
    y_bcc = torch.log(y_yolo[:, ..., 5:]/y_yolo[:,...,5:].sum(2).unsqueeze(-1))
    wh = y_yolo[:, ..., 2:4]/imgsz
    conf = y_yolo[:, ..., 4]
    return y_bcc, wh, conf

def yolo2bcc(yolo_labels, intermediate_yolo_mode = False, torchMode = False):
    BB, G, Nc = [yolo_labels[x] for x in ['labels', 'G', 'Nc']]
    flattened = (len(BB.shape) != 4)
    if not flattened:
        Ng, Na, Nb, _ = BB.shape
        Nbs = [Nb for _ in G]
    else:
        Ng = G.shape[0]
        Nbs = (np.ceil(1.0/G)**2).astype(int)
        Na = int(np.ceil(BB.shape[0]/Nbs.sum()))
    
    S = (1/G).astype(int)
    C = list(range(Nc))
    effective_id = 0
    gc_bb_map = defaultdict(set)
    for g in range(Ng):  # per grid choice
        g_frac = G[g]
        for a in range(Na):  # per anchor box
            gc_id = 0
            Nb = Nbs[g]
            for b in range(Nb):  # per bounding box
                if not flattened:
                    bb = BB[g, a, b, :]
                else:
                    bb = BB[effective_id, :]
                c, x, y, w, h = bb
                if intermediate_yolo_mode:
                    gc = gc_id
                    gc_id += 1
                else:
                    gc = find_containing_grid_cell(g_frac, g_frac, x, y)
                gc_bb_map[(g, a, gc)].add(b)
                effective_id += 1

    bcc_labels = []
    wh_map = {}
    effective_id = 0
    for g in range(Ng): # grid-choices    
        g_frac = G[g]
        cells_per_side = np.ceil(1/g_frac).astype(int)
        for a in range(Na):  # anchor-boxes
            for gc in range(cells_per_side**2):  # grid-cells
                bb_ids = gc_bb_map[(g, a, gc)]
                if len(bb_ids) > 1:
                    print(f"More than one bounding boxes for the same " +
                                    f"(grid-choice ({g}: {G[g]}), anchor-choice ({a}), grid-cell ({gc})) combination")
                if len(bb_ids) == 0:
                    c = BACKGROUND_CLASS_ID
                if len(bb_ids) == 1:
                    (bb_id,) = bb_ids
                    if not flattened:
                        c, _, _, w, h = BB[g, a, bb_id, :]
                    else:
                        c, _, _, w, h = BB[effective_id, :]
                    wh_map[(g, a, gc)] = (w, h)
                bcc_labels.append(c)
                effective_id += 1
    labels = torch.tensor(bcc_labels, dtype=int) if torchMode else np.array(bcc_labels, dtype=int)
    return {'labels': labels, 'wh_map': wh_map, 'Na': Na, 'G': G, 'Nc': Nc}

# After a BCC step is run on a YOLO output (i.e., we have q_t with us), we need to convert it to
# a format acceptable by YOLO as a target. I.e., we need to convert a grid-class tensor to x,y,w,h,c format.
# You give qt, grid sizes G, number of anchors Na, volunteer-image-gridchoice-gridcellid-w-h tensor as input,
# and get a YOLO compatible output.
def qt2yolo_optimized(qt, G, Na, vigcwh, torchMode=False, device=None):
    Ng = G.shape[0]
    num_images = qt.shape[0]
    y_bcc = []
    for i in range(num_images):
        cs = torch.argmax(qt[i, :], 1)
        st = 0
        for g in range(Ng):
            g_frac = G[g]
            S_g = np.ceil(1/g_frac).astype(int)
            n_cells = S_g*S_g
            
            ig_indices = torch.logical_and(vigcwh[:, 1] == i, vigcwh[:, 2] == g) #(78x1)
            vigcwh_ig = vigcwh[ig_indices, :] #select the bbox which is drawn by volunteer in image i
            wh_ig = vigcwh_ig[:, -2:]
            wh_ig_mean = wh_ig.mean(axis=0) # w,h mean within image
            wh_init_multiplier = wh_ig_mean if BKGD_WH_IS_MEAN and wh_ig.shape[0] > 0 else -1 # always -1 dont know why in use
            wh = wh_init_multiplier*torch.ones(n_cells, 2) # 6400 x 2
            tagged_gc_ids = vigcwh_ig[:, 3].unique().int()
            for gc in tagged_gc_ids:
                igc_indices = torch.logical_and(ig_indices, vigcwh[:,3] == gc)
                vigcwh_igc = vigcwh[igc_indices]
                wh_igc = vigcwh_igc[:, -2:]
                wh_igc_mean = wh_igc.mean(axis=0)
                wh[gc, :] = wh_ig_mean if wh_igc.shape[0] == 0 else wh_ig_mean
            for a in range(Na):
                z = torch.linspace(g_frac/2, 1-g_frac/2, S_g).repeat(S_g, 1).unsqueeze(-1)
                xy = torch.cat((z.permute(1, 0, 2), z), 2).permute(1, 0, 2).reshape(n_cells, 2)
                c = cs[st:st+n_cells]
                icxywh = torch.cat(((i*torch.ones(n_cells, 1)).to(device), c.unsqueeze(-1).to(device), xy.to(device), wh.to(device)), 1)
                y_bcc.append(icxywh)
                st += n_cells
    qt_yolo = torch.cat(y_bcc)
    return qt_yolo

# given the previous function qt2yolo_optimized() is computing (image, class, 4location) for each bbox
# now I want to change the single class setting to soft labels that is iamge, c1,c2,c3, 4locations
# Start from easy task by removing the class
def qt2yolo_soft(qt, G, Na, vigcwh, torchMode=False, device=None):
    Ng = G.shape[0]
    num_images = qt.shape[0]
    y_bcc = []
    for i in range(num_images):
        st = 0
        for g in range(Ng):
            g_frac = G[g]
            S_g = np.ceil(1 / g_frac).astype(int)
            n_cells = S_g * S_g

            ig_indices = torch.logical_and(vigcwh[:, 1] == i, vigcwh[:, 2] == g)
            vigcwh_ig = vigcwh[ig_indices, :]
            wh_ig = vigcwh_ig[:, -2:]
            wh_ig_mean = wh_ig.mean(axis=0)
            wh_init_multiplier = wh_ig_mean if BKGD_WH_IS_MEAN and wh_ig.shape[0] > 0 else -1 #why -1 all the time
            wh = wh_init_multiplier * torch.ones(n_cells, 2)
            tagged_gc_ids = vigcwh_ig[:, 3].unique().int()
            for gc in tagged_gc_ids:
                igc_indices = torch.logical_and(ig_indices, vigcwh[:, 3] == gc)
                vigcwh_igc = vigcwh[igc_indices]
                wh_igc = vigcwh_igc[:, -2:]
                wh_igc_mean = wh_igc.mean(axis=0)
                wh[gc, :] = wh_ig_mean if wh_igc.shape[0] == 0 else wh_ig_mean
            for a in range(Na):
                z = torch.linspace(g_frac / 2, 1 - g_frac / 2, S_g).repeat(S_g, 1).unsqueeze(-1)
                xy = torch.cat((z.permute(1, 0, 2), z), 2).permute(1, 0, 2).reshape(n_cells, 2)
                icxywh = torch.cat(
                    ((i * torch.ones(n_cells, 1)).to(device), xy.to(device), wh.to(device)),1)
                y_bcc.append(icxywh)
                st += n_cells
    qt_yolo_soft = torch.cat(y_bcc)
    return qt_yolo_soft


def qt2yolo(qt, G, Na, wh_yolo, torchMode=False, device=None):
    Ng = G.shape[0]
    y_bcc = []
    num_images = qt.shape[0]
    for i in range(num_images):
        effective_id = 0
        cs = torch.argmax(qt[i, :], 1) if torchMode else np.argmax(qt[i, :], 1)
        for g in range(Ng):
            g_frac = G[g]
            S_g = np.ceil(1/g_frac).astype(int)
            for a in range(Na):
                for gc in range(S_g*S_g):
                    x, y = find_grid_center(g_frac, g_frac, gc)
                    w, h = wh_yolo[i][effective_id]
                    c = cs[effective_id]
                    y_bcc.append([i, c, x, y, w, h])
                    effective_id += 1
    return torch.tensor(y_bcc).to(device) if torchMode else np.array(y_bcc)


def bcc2yolo(bcc_labels, return_flattened=False):
    G, Na, Nc, wh_map = [bcc_labels[x] for x in ['G', 'Na', 'Nc', 'wh_map']]
    Ng = G.shape[0]
    S = (1/G).astype(int)
    C = list(range(Nc))
    effective_id = 0
    yolo_labels = []
    for g in range(Ng):
        g_labels = []
        g_frac = G[g]
        cells_per_side = np.ceil(1/g_frac).astype(int)
        for a in range(Na):
            a_labels = []
            for gc in range(cells_per_side**2):  # grid-cells
                c = bcc_labels['labels'][effective_id]
                effective_id += 1
                if c == BACKGROUND_CLASS_ID:
                    continue
                x, y = find_grid_center(g_frac, g_frac, gc)
                # print(effective_id, c)
                w, h = bcc_labels['wh_map'][(g, a, gc)]
                gc_label = [c, x, y, w, h]
                a_labels.append(gc_label)
            g_labels.append(a_labels)
        yolo_labels.append(g_labels)
    if return_flattened:
        y = np.concatenate([np.concatenate(x) for x in yolo_labels])
    else:
        y = np.array(yolo_labels)
    return {'labels': y, 'G': G, 'Nc': Nc}


def init_yolo_labels(BB, Na, G, flatten=False, torchMode = False):
    Ng = G.shape[0]
    base_lib = torch if torchMode else np
    if not flatten:
        labels = base_lib.tile(BB, (Ng, Na, 1, 1))
    else:
        labels = base_lib.tile(BB, (Ng*Na, 1))
    return labels

def equals(a, b):
    if type(a) != type(b):
        return False
    if type(a) == dict:
        if len(a) != len(b) or a.keys() != b.keys():
            return False
        return all([equals(a[k], b[k]) for k in a.keys()])
    if type(a) == list:
        if len(a) != len(b):
            return False
        return all([equals(a[i], b[i]) for i in range(len(a))])
    if type(a) == np.ndarray:
        return np.array_equal(a, b)
    return a == b


def test_bcc2yolo():
    input, exp_output = generate_test_pairs(mode = 'bcc2yolo')
    output = bcc2yolo(input)
    assert equals(output, exp_output), "BCC2YOLO test case failed"

def test_yolo2bcc():
    input, exp_output = generate_test_pairs(mode = 'yolo2bcc')
    output = yolo2bcc(input)
    assert equals(output, exp_output), "YOLO2BCC test case failed"

def generate_test_pairs(mode = 'yolo2bcc'):
    G = np.array([1.0/4, 1.0/2])
    Na = 3
    Nc = 2
    y_yolo_orig = np.array(
        [
            [0, 0.2, 0.3, 0.2, 0.4],
            [1, 0.5, 0.4, 0.2, 0.2],
            [0, 0.7, 0.7, 0.3, 0.3]
        ])
    y_yolo_derived = np.array([[0   , 0.25 , 0.25 , 0.2  , 0.4  ],
          [1   , 0.75 , 0.25 , 0.2  , 0.2  ],
          [0   , 0.75 , 0.75 , 0.3  , 0.3  ]])
    y_bcc = np.array(
        [-1, -1, -1, -1,  0, -1,  1, -1, -1, -1,  0, -1, -1, -1, -1, -1,  0,
          1, -1,  0, -1, -1, -1, -1,  0, -1,  1, -1, -1, -1,  0, -1, -1, -1,
         -1, -1,  0,  1, -1,  0, -1, -1, -1, -1,  0, -1,  1, -1, -1, -1,  0,
         -1, -1, -1, -1, -1,  0,  1, -1,  0])
    wh_map = {(0, 0, 4): (0.2, 0.4),
            (0, 0, 6): (0.2, 0.2),
            (0, 0, 10): (0.3, 0.3),
            (0, 1, 0): (0.2, 0.4),
            (0, 1, 1): (0.2, 0.2),
            (0, 1, 3): (0.3, 0.3),
            (1, 0, 4): (0.2, 0.4),
            (1, 0, 6): (0.2, 0.2),
            (1, 0, 10): (0.3, 0.3),
            (1, 1, 0): (0.2, 0.4),
            (1, 1, 1): (0.2, 0.2),
            (1, 1, 3): (0.3, 0.3),
            (2, 0, 4): (0.2, 0.4),
            (2, 0, 6): (0.2, 0.2),
            (2, 0, 10): (0.3, 0.3),
            (2, 1, 0): (0.2, 0.4),
            (2, 1, 1): (0.2, 0.2),
            (2, 1, 3): (0.3, 0.3)}
    yolo_labels_orig = {'labels': init_yolo_labels(
        y_yolo_orig, Na, G), 'G': G, 'Nc': Nc}
    yolo_labels_derived = {'labels': init_yolo_labels(
        y_yolo_derived, Na, G), 'G': G, 'Nc': Nc}
    bcc_labels = {'labels': y_bcc, 'Na': Na, 'G': G, 'Nc': 2,'wh_map': wh_map}

    if mode == 'yolo2bcc':
        input, output = yolo_labels_orig, bcc_labels
    elif mode == 'bcc2yolo':
        input, output = bcc_labels, yolo_labels_derived
    return input, output

def generate_point_in_grid_cell(grid_ratio, nc, grid_cell_id):
    x, y = find_grid_center(grid_ratio, grid_ratio, grid_cell_id)
    w, h = np.random.rand(2)
    c = np.random.randint(0, nc)
    return np.array([c, x, y, w, h])

def main():
    # For one image, we have
    # Nb=3 bounding boxes
    # BB = np.array(
    #     [
    #         [0, 0.2, 0.3, 0.2, 0.4],
    #         [1, 0.5, 0.4, 0.2, 0.2],
    #         [0, 0.7, 0.7, 0.3, 0.3]
    #     ])

    # Ng=2 grid choices
    G = np.array([1.0/4, 1.0/2])

    # Na=3 anchor-box choices
    Na = 3

    # Classes
    Nc = 2
    points = []
    for g in range(G.shape[0]):
        gr = G[g]
        S = np.ceil(1.0/gr).astype(int)
        for a in range(Na):
            for gc in range(S*S):
                p = generate_point_in_grid_cell(gr, Nc, gc)
                points.append(p)
    points = np.array(points)
    yolo_labels = {'labels': points, 'G': G, 'Nc': Nc}

    y_l = yolo_labels
    old_b_l = {'labels': np.zeros(1)}
    old_y_l = y_l.copy()
    for i in range(5):
        b_l = yolo2bcc(y_l, intermediate_yolo_mode=True)
        # Beqflag = np.array_equal(b_l['labels'], old_b_l['labels'])
    #     Yeqflag = np.array_equal(y_l['labels'], old_y_l['labels'])
    #     print(f'          Y{i} =' if i == 0 else f'Y{i} = B2Y(B{i-1}) =',
    #           y_l['labels'].shape, f'\tB{i} = Y2B(Y{i}) =', b_l['labels'].shape,
    #           f'\tB{i} == B{i-1}?: {Beqflag}' if i > 0 else '',
    #           f'\tY{i} == Y{i-1}?: {Yeqflag}' if i > 0 else '')
    #     old_y_l = y_l.copy()
        y_l = bcc2yolo(b_l, return_flattened=True)
        print(y_l)
    #     old_b_l = b_l.copy()



def main1():
    # For one image, we have
    # Nb=3 bounding boxes
    BB = np.array(
        [
            [0, 0.2, 0.3, 0.2, 0.4],
            [1, 0.5, 0.4, 0.2, 0.2],
            [0, 0.7, 0.7, 0.3, 0.3]
        ])

    # Ng=2 grid choices
    G = np.array([1.0/4, 1.0/2])

    # Na=3 anchor-box choices
    Na = 3

    # Classes
    Nc = 2

    yolo_labels = {'labels': init_yolo_labels(
        BB, Na, G, flatten=False), 'G': G, 'Nc': Nc}

    y_l = yolo_labels
    old_b_l = {'labels': np.zeros(1)}
    old_y_l = y_l.copy()
    for i in range(5):
        b_l = yolo2bcc(y_l, intermediate_yolo_mode=True)
        Beqflag = np.array_equal(b_l['labels'], old_b_l['labels'])
        Yeqflag = np.array_equal(y_l['labels'], old_y_l['labels'])
        print(f'          Y{i} =' if i == 0 else f'Y{i} = B2Y(B{i-1}) =',
              y_l['labels'].shape, f'\tB{i} = Y2B(Y{i}) =', b_l['labels'].shape,
              f'\tB{i} == B{i-1}?: {Beqflag}' if i > 0 else '',
              f'\tY{i} == Y{i-1}?: {Yeqflag}' if i > 0 else '')
        old_y_l = y_l.copy()
        y_l = bcc2yolo(b_l)
        old_b_l = b_l.copy()


if __name__ == '__main__':
    main()
