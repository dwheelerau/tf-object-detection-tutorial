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

For this work I will train on kangaroos just to ensure that the model works on a single category.  I then will use the same pipeline to train on warrens.  

## Tutorial on kangaroos  

The kangaroos dataset was part of the xx package, I am only going to annotate xxx, I will follow the workflow here but I also should try it with the MD model which should be a good starting point for kangaroo identification as well as perhaps warrens given they glow in the dark of the IR cameras used on these drones.  

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
# AAH
#### If you run into numpy errors do latter ###
pip uninstall pycocotools
# then
python setup.py install
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
Check your images, find all images files that have a target category ie kangaroos in this case, copy them to `training_demo/images` folder. Use `labelImg` to draw boundary boxes around each positive detection and save the result, this should create a corresponding xml file for each JPG.  

FYI the shortcut for the boundary box tool is `w` and next is `d`.  

```
tree images/
images/
├── 00001.jpg
├── 00001.xml
├── 00003.jpg
├── 00003.xml
├── 00004.jpg
├── 00004.xml
├── 00005.jpg
...
```
Copy a random 10% of images and xml into the `test` folder and the rest into the `training` folder. I did this manually, but there is a helper script in `./scripts/partion_dataset.py` that will do this automatically.    
```
ls -l images/train/*.jpg | wc -l
100
ls -l images/train/*.xml | wc -l
100
ls -l images/test/*.jpg | wc -l
34
ls -l images/test/*.xml | wc -l
34
```

Now a new directory for helper scripts in `TensorFlow/scripts`.
```
mkdir -l Tensorflow/scripts
# scripts for preprocess training inputs
mkdir -p Tensorflow/scripts/preprocessing
# we will save a copy of the partition_dataset.py in here
```

Now we need to create a label map, it might look like this.  
```
item {
    id: 1
    name: 'cat'
}

item {
    id: 2
    name: 'dog'
}
```
So we will create ours for kangaroos  
```
vim Tensorflow/workspace/training_demo/annotations/label_map.pbtxt
```
Mine looks like this.
```
#cat Tensorflow/workspace/training_demo/annotations/label_map.pbtxt
item {
    id: 1
    name: 'kangaroo'
}
```

Convert `xml` to `.record` files. See `Tensorflow/scripts/preprocessing/generate_tfrecord.py`
```
# install pandas
conda install pandas
cd TensorFlow/scripts/preprocessing/
# training tfrecords
python generate_tfrecord.py -x ../../workspace/training_demo/images/train -l ../../workspace/training_demo/annotations/label_map.pbtxt -o ../../workspace/training_demo/annotations/train.record
# test records
python generate_tfrecord.py -x ../../workspace/training_demo/images/test -l ../../workspace/training_demo/annotations/label_map.pbtxt -o ../../workspace/training_demo/annotations/test.record
Successfully created the TFRecord file: ../../workspace/training_demo/annotations/test.record
# check
tree ../../workspace/training_demo/annotations/
├── label_map.pbtxt
├── test.record
└── train.record
```

## Configure the training
For this we will be using transfer learning, if you want to train a model from scrach see `https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md`. 

The model we shall be using in our examples is the SSD ResNet50 V1 FPN 640x640 model, since it provides a relatively good trade-off between performance and speed. However, there exist a number of other models you can use, all of which are listed in TensorFlow 2 Detection Model Zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

1. Download the pre-trained model (http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)  into `training_demo/pre-trained-models`. While we are at it we can grab another model frames work, efficentDet from (http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz) and put that in the same directory.    
```
 tree -L 2 Tensorflow/workspace/training_demo/pre-trained-models/
Tensorflow/workspace/training_demo/pre-trained-models/
├── efficientdet_d1_coco17_tpu-32
│   ├── checkpoint
│   ├── pipeline.config
│   └── saved_model
└── ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
    ├── checkpoint
    ├── pipeline.config
    └── saved_model
```

2. Configure the training pipeline  
Under the training_demo/models create a new directory named my_ssd_resnet50_v1_fpn and copy the training_demo/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config file inside the newly created directory.
```
mkdir Tensorflow/workspace/training_demo/models/my_ssd_resnet50_v1_fpn
cp Tensorflow/workspace/training_demo/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config Tensorflow/workspace/training_demo/models/my_ssd_resnet50_v1_fpn/
```
Now modify this file to reflect  
1. the number of classes: `num_classes: 1`  
2. `batch_size: 8` # Increase/Decrease this value depending on the
3. `fine_tune_checkpoint: "pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0" # Path to checkpoint of pre-trained model`
4. `fine_tune_checkpoint_type: "detection" `  
5. `use_bfloat16: false`  # false if not using TPU 
6. `label_map_path: "annotations/label_map.pbtxt"`   
7. `input_path: "annotations/train.record"`  
8. `metrics_set: "coco_detection_metrics"`  
9. `use_moving_averages: false`
10. `label_map_path: "annotations/label_map.pbtxt"`  
11. `input_path: "annotations/test.record" `  

See the config file comments for more details on this.  

It is worth noting here that the changes to lines 178 to 179 above are optional. These should only be used if you installed the COCO evaluation tools (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tf-models-install-coco), as outlined in the COCO API installation section, and you intend to run evaluation (see Evaluating the Model (Optional)).

So my file looks like this.  

```
# cat Tensorflow/workspace/training_demo/models/my_ssd_resnet50_v1_fpn/pipeline.config 
model {
  ssd {
    num_classes: 1
    image_resizer {
      fixed_shape_resizer {
        height: 640
        width: 640
      }
    }
    feature_extractor {
      type: "ssd_resnet50_v1_fpn_keras"
      depth_multiplier: 1.0
      min_depth: 16
      conv_hyperparams {
        regularizer {
          l2_regularizer {
            weight: 0.00039999998989515007
          }
        }
        initializer {
          truncated_normal_initializer {
            mean: 0.0
            stddev: 0.029999999329447746
          }
        }
        activation: RELU_6
        batch_norm {
          decay: 0.996999979019165
          scale: true
          epsilon: 0.0010000000474974513
        }
      }
      override_base_feature_extractor_hyperparams: true
      fpn {
        min_level: 3
        max_level: 7
      }
    }
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    box_predictor {
      weight_shared_convolutional_box_predictor {
        conv_hyperparams {
          regularizer {
            l2_regularizer {
              weight: 0.00039999998989515007
            }
          }
          initializer {
            random_normal_initializer {
              mean: 0.0
              stddev: 0.009999999776482582
            }
          }
          activation: RELU_6
          batch_norm {
            decay: 0.996999979019165
            scale: true
            epsilon: 0.0010000000474974513
          }
        }
        depth: 256
        num_layers_before_predictor: 4
        kernel_size: 3
        class_prediction_bias_init: -4.599999904632568
      }
    }
    anchor_generator {
      multiscale_anchor_generator {
        min_level: 3
        max_level: 7
        anchor_scale: 4.0
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        scales_per_octave: 2
      }
    }
    post_processing {
      batch_non_max_suppression {
        score_threshold: 9.99999993922529e-09
        iou_threshold: 0.6000000238418579
        max_detections_per_class: 100
        max_total_detections: 100
        use_static_shapes: false
      }
      score_converter: SIGMOID
    }
    normalize_loss_by_num_matches: true
    loss {
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      classification_loss {
        weighted_sigmoid_focal {
          gamma: 2.0
          alpha: 0.25
        }
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    encode_background_as_zeros: true
    normalize_loc_loss_by_codesize: true
    inplace_batchnorm_update: true
    freeze_batchnorm: false
  }
}
train_config {
  batch_size: 8
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  sync_replicas: true
  optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.03999999910593033
          total_steps: 25000
          warmup_learning_rate: 0.013333000242710114
          warmup_steps: 2000
        }
      }
      momentum_optimizer_value: 0.8999999761581421
    }
    use_moving_average: false
  }
  fine_tune_checkpoint: "/home/dpidave/training_resources/tf-object-detection-tutorial/Tensorflow/workspace/training_demo/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0"
  num_steps: 25000
  startup_delay_steps: 0.0
  replicas_to_aggregate: 8
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: "detection"
  use_bfloat16: false
  fine_tune_checkpoint_version: V2
}
train_input_reader {
  label_map_path: "/home/dpidave/training_resources/tf-object-detection-tutorial/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "/home/dpidave/training_resources/tf-object-detection-tutorial/Tensorflow/workspace/training_demo/annotations/train.record"
  }
}
eval_config {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "/home/dpidave/training_resources/tf-object-detection-tutorial/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "/home/dpidave/training_resources/tf-object-detection-tutorial/Tensorflow/workspace/training_demo/annotations/test.record"
  }
}
```

As always its a good idea to check your paths by copy paste and `ls -l` them to make sure they are correct.  

## Training the model  
Open a new terminal and activate the env again, then copy `TensorFlow/models/research/object_detection/model_main_tf2.py` to `TensorFlow/workspace/training_demo/'  
```
# activate new env with conda activate tensorflow
cp Tensorflow/models/research/object_detection/model_main_tf2.py Tensorflow/workspace/training_demo/
cd Tensorflow/workspace/training_demo/
python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config
```

