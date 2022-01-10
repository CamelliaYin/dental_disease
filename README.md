# Crowd-YOLO

This is the official repository for Crowd-YOLO, that is based on the [Bayesian Classifier Combination neural network (BCCNet)](https://github.com/OlgaIsupova/BCCNet) and the [YOLO (You Only Look Once)](https://github.com/ultralytics/yolov5) object detection algorithm.
We use the version 5 of all available YOLO algorithms/implementations (a.k.a. YOLOv5).
The latest version of our paper could be found here (@TODO: LINK HERE).

## Directory structure
```
├── data [The main data directory where all the YAML files lie.]
│   ├── cyolo.yaml [An example of a YAML file for data for CYOLO.]
│   ├── datasets [The directory where all the actual data lies.]
│   │   ├── bcc-tvt [Example of volunteer-based data used for CYOLO]
│   │   │   ├── images [The directory where all images reside]
│   │   │   │   ├── test [Test images]
│   │   │   │   │   └── IS20191111_161011_0377_0000387B.jpg [A test image]
│   │   │   │   ├── train [Train images]
│   │   │   │   │   └── IS20180614_161235_0835_000000CD.jpg [A train image]
│   │   │   │   └── val [Validation images]
│   │   │   │       └── IS20191125_105634_0062_000039C9.jpg [A validation image]
│   │   │   ├── labels [The directory where all labels lie. Labels from all volunteers are vertically concatenated.]
│   │   │   │   ├── test [Test labels]
│   │   │   │   │   └── IS20191111_161011_0377_0000387B.txt [A test label file]
│   │   │   │   ├── train [Train labels]
│   │   │   │   │   └── IS20180614_161235_0835_000000CD.txt [A train label file]
│   │   │   └── volunteers [The directory where all volunteer information lies]
│   │   │       ├── test [Volunteer info corresponding to test labels.]
│   │   │       │   └── IS20191111_161011_0377_0000387B.txt [A volunteer-correspondence file]
│   │   │       ├── train [Volunteer info corresponding to train labels.]
│   │   │       │   └── IS20180614_161235_0835_000000CD.txt [A volunteer-correspondence file]
│   │   │       └── val [Volunteer info corresponding to validation labels.]
│   │   │           └── IS20191125_105634_0062_000039C9.txt [A volunteer-correspondence file]
│   │   ├── iid-tvt [Example of a volunteer-flattened data used for original YOLO]
│   │   │   ├── images [Each image present in the master data is replicated for each volunteer (iid)]
│   │   │   │   ├── test
│   │   │   │   │   └── IS20191111_161011_0377_0000387B.Jonathan.jpg [Test image for expert Jonathan]
│   │   │   │   ├── train
│   │   │   │   │   └── IS20180614_161235_0835_000000CD.Camellia.jpg [Train image for volunteer Camellia]
│   │   │   │   └── val
│   │   │   │       └── IS20191125_105634_0062_000039C9.Camellia.jpg [Validation image for volunteer Camellia]
│   │   │   └── labels [Labels corresponding to image-volunteer combination]
│   │   │       ├── test
│   │   │       │   └── IS20191111_161011_0377_0000387B.Jonathan.txt
│   │   │       ├── train
│   │   │       │   └── IS20180614_161235_0835_000000CD.Camellia.txt
│   │   │       └── val
│   │   │           └── IS20191125_105634_0062_000039C9.Camellia.txt
│   │   ├── master [The directory where ALL the data lies. All other data is created from this using the data_preparer.py utility]
│   │   │   ├── images [All available images in the data.]
│   │   │   │   └── IS20180614_161235_0835_000000CD.jpg
│   │   │   └── labels [All available labels from all labellers]
│   │   │       ├── Camellia [Directory for labeller Camellia]
│   │   │       │   └── IS20180614_161235_0835_000000CD.txt
│   │   │       ├── Conghui [Directory for labeller Conghui]
│   │   │       │   └── IS20180614_161235_0835_000000CD.txt
│   │   │       ├── HaoWen [Directory for labeller HaoWen]
│   │   │       │   └── IS20180614_161235_0835_000000CD.txt
│   │   │       ├── Jonathan [Directory for labeller Jonathan]
│   │   │       │   └── IS20180614_161235_0835_000000CD.txt
│   │   │       └── Xiongjie [Directory for labeller Xiongjie]
│   │   │           └── IS20180614_161235_0835_000000CD.txt
│   ├── hyps [Hyperparameter settings preset by the original YOLO code]
│   │   ├── hyp.finetune.yaml [Settings recommended for finetuning the model]
│   │   ├── hyp.finetune_objects365.yaml
│   │   ├── hyp.scratch-p6.yaml
│   │   └── hyp.scratch.yaml [Settings recommended for training the model from scratch]
│   └── yolo.yaml [An example of data YAML for original YOLO]
├── notebooks [All the notebooks lie here.]
│   ├── reproduce_results.ipynb [Reproduce all results using this notebook.]
│   ├── results_plot_govind.ipynb [A subset of reproduce_results.ipynb]
│   ├── runs [A directory where all results from each run of the algorithm lie.]
│   └── yolov5s.pt [A pre-trained YOLO model]
├── requirements.txt
└── src
    ├── README.md [YOLO's original README]
    ├── cyolo_utils [Utilities added by us, over original YOLO ones.]
    │   ├── data_preparer.py [Prepare data using the master file.]
    │   ├── label_converter.py [Utilities to manipulate target labels (from YOLO to BCC and vice-versa.)]
    │   ├── label_filter.py [Not being used as of now, but used to filter target labels.]
    │   ├── results_plotter.py [Utilities to plot results via the reproduce_results.ipynb notebook]
    │   └── train_with_bcc.py [Prediction utilities for BCC]
    ├── data_preparer.py [Used by the reproduce_results.ipynb notebook to prepare data.]
    ├── detect.py [YOLO's native directory]
    ├── export.py [YOLO's native directory]
    ├── hubconf.py [YOLO's native directory]
    ├── lib
    │   └── BCCNet [The BCCNet library as coded by Olga, et al.; although some refactoring is done.]
    │       ├── LICENSE
    │       ├── NNArchitecture
    │       │   └── lenet5_mnist.py [Model used to demonstrate BCC]
    │       ├── README.md
    │       ├── SyntheticCrowdsourcing
    │       │   └── synthetic_crowd_volunteers.py
    │       ├── VariationalInference
    │       │   ├── VB_iteration.py
    │       │   ├── VB_iteration_yolo.py [The YOLO version of VB_iteration.py]
    │       │   └── confusion_matrix.py
    │       ├── demo.ipynb
    │       ├── demo.py
    │       └── utils
    │           └── utils_dataset_processing.py
    ├── models [YOLO's original code directory]
    ├── results_metadata.json [Could be used to compare results.]
    ├── sanity_check_train.py [Used for sanity check via Single volunteer data]
    ├── train.py [The main file that is run for training CYOLO]
    ├── train_backup.py [A backup of the original train.py]
    ├── utils [Internal YOLO utils]
    └── val.py [Internal YOLO file, made changes to custom print confusion matrices and intermediate results.]
```

## Usage

### To reproduce results in the paper

- Use the `reproduce_results.ipynb` Notebook.

## Citation

If you use this code in your pipeline, please cite our work as follows:

@TODO: Add a citation to the paper.
