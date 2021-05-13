# tf-object-detection-tutorial

This is based on the tutorial available here (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/).  

The goal is to be able to detect rabit warrens in drone imagary, so the outcome of this work will be a workflow for processing drone footage.  At this stage I think the workflow would look like this:  

- Downsize video via dropping frames (optional)  
- Reduce size of all frames (optional)  
- Processing  
  - run model across frames and predict objects  
  - report positive frames in a summary spreadsheet  
  - rejoin frames with boundary boxes drawn around objects  
- training  
  - Manually identify positive frames to train on  
  - Annotate the frames using labellmg (https://tzutalin.github.io/labelImg/)  
  - split the data into training and test sets  
  - convert xml and images to tf records  
  - train the model (ie this tutorial)  
  - assess the model with the test set  

For this work I will train on cats just to ensure that the model works on a single category.  I then will use the same pipeline to train on warrens.  

## Tutorial on cats  

The cats dataset was part of the xx package, I am only going to annotate xxx, I will follow the workflow here but I also should try it with the MD model which should be a good starting point for cat identification as well as perhaps warrens given they glow in the dark of the IR cameras used on these drones.  

### Setting up base  
```
# tensorflow  
conda create -n tensorflow pip python=3.8
conda activate tensorflow
pip install --ignore-installed --upgrade tensorflow==2.2.0

# verify install should output info on CPU/GPU avail   
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

# obj detection API (do this from the base of the repo)   
mkdir Tensorflow
cd Tensorflow
git clone https://github.com/tensorflow/models.git
ls models -1
## output
#AUTHORS
#CODEOWNERS
#community
#CONTRIBUTING.md
#ISSUES.md
#LICENSE
#official
#orbit
#README.md
#research
##

# The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. 
# 1. Download models from https://github.com/protocolbuffers/protobuf/releases
# 2. Download the protoc-*-*.zip ie protoc-3.17.0-linux-x86_64.zip
# 3. Extract contents
mkdir ~/software/google_protobuf
unzip ~/Downloads/protoc-3.17.0-linux-x86_64.zip -d ~/software/google_protobuf/
ls ~/software/google_protobuf
## output
#bin
#include
#readme.txt
##
# 4. Add the directory to the path  
vim ~/.bashrc  
# add the following line 
export PATH="$PATH:/home/dpidave/software/google_protobuf/bin"
# reload the env
source ~/.bashrc
# reactivate the conda env
conda activate tensorflow
# test
cd models/research/
for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.
# I got nothing in the output, but running protoc should bring up manual  

# The tensorflow API req pycocotools, download git ZIP and make, then copy
# to <repo_base>/TensorFlow/models/research/
# 1. Goto https://github.com/cocodataset/cocoapi download ZIP
# 2. unzip
# 3. from the repo
cd PythonAPI/
# MAKE SURE YOU ARE USING THE CONDA TENSORFLOW ENV
# make sure pip is python 3.8 from conda enbv
pip --version 
pip install cython
pip install pycocotools
# req sudo
sudo make
cp -r pycocotools ~/training_resources/tf-object-detection-tutorial/Tensorflow/models/research/
# check that is there in the dest
cd ~/training_resources/tf-object-detection-tutorial/Tensorflow/models/research/
ls pycocotools/
# should output files

# install object detection API
# From within TensorFlow/models/research/
cp object_detection/packages/tf2/setup.py .
python -m pip install .
# the above step will break the internet!
# to test, from within TensorFlow/models/research/
python object_detection/builders/model_builder_tf2_test.py
## output lots of OKs!
# OK 
# OK 
# OK 
```

### Training a custom object detector!  
1. Preparing the workspace  
```
# from the base repo, this training_demo will be our training folder
mkdir -p TensforFlow/workspace/training_demo
cd Tensorflow/workspace/training_demo/
# annotations will store .csv files and tfrecords
mkdir annotations/
# exported versions of our trained model(s)
exported-models/
# .xml and JPG images from labelImg
mkdir -p images/test
mkdir -p images/train
# contain a sub-folder for each of training job. 
# Each subfolder contain the training pipeline config *.config
mkdir models
# downloaded pre-trained models, ie starting checkpoints for transfer learn
pre-trained-models/
# optional readme telling the user what the hell you did.  
touch README.md
tree
.
├── annotations
├── exported-models
├── images
│   ├── test
│   └── train
├── models
├── pre-trained-models
└── README.md

```

2. Preparing the dataset
Check your images, find all images files that have a target category ie cats in this case, copy them to `training_demo/images` folder. Use `labelImg` to draw boundary boxes around each positive detection and save the result, this should create a corresponding xml file for each JPG.  
```
ls images
o
