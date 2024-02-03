########
#
# train-usgs-tegu-detector-yolov8.py
#
# This file documents the model training process, starting from where prepare-yolo-training-set.py
# leaves off.  Training happens at the yolov5 CLI, and the exact command line arguments are documented
# in the "Train" cell.
#
# Later cells in this file also:
#
# * Run the YOLOv5 validation scripts
# * Convert YOLOv5 val results to MD .json format
# * Use the MD visualization pipeline to visualize results
# * Use the MD inference pipeline to run the trained model
#
########

#%% Train

# Tips:
#
# https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results


## Environment prep

"""
mamba create --name yolov8 pip python==3.11 -y
mamba activate yolov8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
"""

## Training

"""
mkdir -p ~/tmp/usgs-tegus/yolov8-training
cd ~/tmp/usgs-tegus/yolov8-training
export PYTHONPATH=
LD_LIBRARY_PATH=
mamba activate yolov8

BATCH_SIZE=-1
IMAGE_SIZE=640
EPOCHS=300
DATA_YAML_FILE=/home/user/data/usgs-kissel-training-yolo/dataset.yaml
BASE_MODEL=yolov8x.pt
PROJECT=/home/user/tmp/usgs-tegus-yolov8-training
TRAINING_RUN_NAME=usgs-tegus-yolov8x-2023.10.26-b${BATCH_SIZE}-img${IMAGE_SIZE}-e${EPOCHS}
NAME=${TRAINING_RUN_NAME}

yolo detect train data=${DATA_YAML_FILE} batch=${BATCH_SIZE} model=${BASE_MODEL} epochs=${EPOCHS} imgsz=${IMAGE_SIZE} project=${PROJECT} name=${NAME} device=0,1
"""

# When training on multiple GPUs, batch=-1 is ignored, and the default batch size (16) is used

## Resuming training

"""
cd ~/git
export PYTHONPATH=
LD_LIBRARY_PATH=
mamba activate yolov8
yolo train resume model=/home/user/tmp/usgs-tegus-yolov8-training/usgs-tegus-yolov8x-2023.10.26-b-1-img640-e300/weights/last.pt
"""
pass


#%% Make plots during training

import os
import pandas as pd
import matplotlib.pyplot as plt

results_file = os.path.expanduser('/home/user/tmp/usgs-tegus/yolov8-training/runs/detect/train/results.csv')
                                  
df = pd.read_csv(results_file)
df = df.rename(columns=lambda x: x.strip())
    
fig,ax = plt.subplots()

df.plot(x = 'epoch', y = 'val/box_loss', ax = ax) 
df.plot(x = 'epoch', y = 'val/dfl_loss', ax = ax, secondary_y = True) 

df.plot(x = 'epoch', y = 'train/box_loss', ax = ax) 
df.plot(x = 'epoch', y = 'train/dfl_loss', ax = ax, secondary_y = True) 

plt.show()

fig,ax = plt.subplots()

df.plot(x = 'epoch', y = 'val/cls_loss', ax = ax) 
df.plot(x = 'epoch', y = 'train/cls_loss', ax = ax) 

plt.show()

fig,ax = plt.subplots()

df.plot(x = 'epoch', y = 'metrics/precision(B)', ax = ax)
df.plot(x = 'epoch', y = 'metrics/recall(B)', ax = ax)
df.plot(x = 'epoch', y = 'metrics/mAP50(B)', ax = ax)
df.plot(x = 'epoch', y = 'metrics/mAP50-95(B)', ax = ax)

plt.show()

# plt.close('all')

# 200
"""
Model summary (fused): 268 layers, 68138013 parameters, 0 gradients, 257.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95
                   all      10382      11704      0.822      0.673      0.758      0.629
     american_cardinal      10382        185      0.853      0.805      0.856      0.706
              blue_jay      10382         61      0.991      0.656      0.858      0.696
                  crow      10382       4565      0.924      0.926      0.961      0.874
          green_iguana      10382          2          1          0      0.496      0.298
          grey_catbird      10382        358      0.785      0.867      0.874      0.701
                 human      10382        242       0.77      0.846      0.821      0.766
  northern_mockingbird      10382         71      0.545      0.761      0.655      0.561
            other_bird      10382        944      0.836      0.464      0.658      0.506
          other_mammal      10382        811      0.921      0.695      0.851      0.784
              ovenbird      10382         22      0.467      0.682      0.592      0.488
               raccoon      10382       1561      0.906      0.937      0.964      0.903
                rodent      10382       1453       0.91      0.853      0.936      0.715
                 snake      10382         25      0.638       0.28      0.257      0.209
                  tegu      10382        211       0.84      0.573      0.682      0.421
        turkey_vulture      10382       1193      0.945      0.757      0.913      0.801
"""

# 300
"""
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95
                   all      10382      11704      0.869      0.659      0.773      0.643
     american_cardinal      10382        185      0.896      0.778      0.863      0.742
              blue_jay      10382         61      0.957      0.722      0.886      0.774
                  crow      10382       4565      0.946      0.916      0.964      0.886
          green_iguana      10382          2          1          0      0.498     0.0997
          grey_catbird      10382        358      0.818      0.841      0.848      0.714
                 human      10382        242      0.757      0.814      0.782      0.742
  northern_mockingbird      10382         71      0.683      0.704      0.699      0.621
            other_bird      10382        944      0.866      0.494      0.675      0.536
          other_mammal      10382        811      0.948       0.65      0.834      0.775
              ovenbird      10382         22       0.69      0.727      0.702      0.599
               raccoon      10382       1561      0.918      0.907       0.96      0.909
                rodent      10382       1453      0.931      0.786      0.933      0.733
                 snake      10382         25      0.766       0.28      0.312      0.246
                  tegu      10382        211      0.888      0.565      0.729      0.458
        turkey_vulture      10382       1193      0.972      0.699      0.906      0.807
"""

#%% Back up trained weights

"""
mkdir -p ~/models/usgs-tegus
# TRAINING_RUN_NAME="usgs-tegus-yolov8x-2023.10.25-b-1-img640-e200"
# TRAINING_OUTPUT_FOLDER="/home/user/tmp/usgs-tegus/yolov8-training/runs/detect/train/weights"

TRAINING_RUN_NAME="usgs-tegus-yolov8x-2023.10.26-b-1-img640-e300"
TRAINING_OUTPUT_FOLDER="/home/user/tmp/usgs-tegus-yolov8-training/usgs-tegus-yolov8x-2023.10.26-b-1-img640-e300/weights/"

cp ${TRAINING_OUTPUT_FOLDER}/best.pt ~/models/usgs-tegus/${TRAINING_RUN_NAME}-best.pt
cp ${TRAINING_OUTPUT_FOLDER}/last.pt ~/models/usgs-tegus/${TRAINING_RUN_NAME}-last.pt
"""

pass


#%% Validation with YOLOv8

import os

model_base = os.path.expanduser('~/models/usgs-tegus')
# training_run_names = ['usgs-tegus-yolov8x-2023.10.25-b-1-img640-e200']
# project_name = os.path.expanduser('~/tmp/usgs-tegus-val-v8')

training_run_names = ['usgs-tegus-yolov8x-2023.10.26-b-1-img640-e300']
project_name = os.path.expanduser('~/tmp/usgs-tegus-val-v8-300')

data_folder = os.path.expanduser('~/data/usgs-kissel-training-yolo')
image_size = 640

# Note to self: validation batch size appears to have no impact on mAP
# (it shouldn't, but I verified that explicitly)
batch_size_val = 16

data_file = os.path.join(data_folder,'dataset.yaml')
assert os.path.isfile(data_file)
augment_flags = [True,False]

assert os.path.isfile(data_file)

commands = []

n_devices = 2

from collections import defaultdict
device_to_command = defaultdict(list)

# training_run_name = training_run_names[0]
for training_run_name in training_run_names:
    
    for augment in augment_flags:
        
        model_file_base = os.path.join(model_base,training_run_name)
        model_files = [model_file_base + s for s in ('-last.pt','-best.pt')]
        
        # model_file = model_files[0]
        for model_file in model_files:
            
            assert os.path.isfile(model_file)
            
            model_short_name = os.path.basename(model_file).replace('.pt','')
            
            cuda_index = len(commands) % n_devices
            cuda_string = 'CUDA_VISIBLE_DEVICES={}'.format(cuda_index)                        
            
            if augment:
                aug_string = 'aug'
            else:
                aug_string = 'noaug'
                
            if augment:
                augment_string = 'augment'
            else:
                augment_string = ''
                
            cmd = cuda_string + \
                ' yolo val {} model={} imgsz={} batch={} data={} project={} name={}'.format(
                augment_string,model_file,image_size,batch_size_val,data_file,
                project_name,model_short_name)        
            cmd += ' save_hybrid save_json'
            
            commands.append(cmd)
            
            device_to_command[cuda_index].append(cmd)
            
        # ...for each model
    
    # ...augment on/off        
    
# ...for each training run    

for cuda_index in device_to_command:
    commands = device_to_command[cuda_index]
    for s in commands:
        print(s + '\n')
        

#%% Results

"""
last-aug
"""

"""
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                   all      10382      11704          1          1      0.995      0.995
     american_cardinal      10382        185          1          1      0.995      0.995
              blue_jay      10382         61          1          1      0.995      0.995
                  crow      10382       4565          1          1      0.995      0.995
          green_iguana      10382          2          1          1      0.995      0.995
          grey_catbird      10382        358          1          1      0.995      0.995
                 human      10382        242          1          1      0.995      0.995
  northern_mockingbird      10382         71          1          1      0.995      0.995
            other_bird      10382        944          1          1      0.995      0.995
          other_mammal      10382        811          1          1      0.995      0.995
              ovenbird      10382         22          1          1      0.995      0.995
               raccoon      10382       1561          1          1      0.995      0.995
                rodent      10382       1453          1          1      0.995      0.995
                 snake      10382         25          1          1      0.995      0.995
                  tegu      10382        211          1          1      0.995      0.995
        turkey_vulture      10382       1193          1          1      0.995      0.995
"""
        
"""
best-aug
"""


"""
last-noaug
"""

"""
best-noaug
"""


#%% Convert YOLO val .json results to MD .json format

# pip install jsonpickle humanfriendly tqdm skicit-learn

import os
import glob
from data_management import yolo_output_to_md_output

class_mapping_file = os.path.expanduser('~/data/usgs-kissel-training-yolo/dataset.yaml')
                        
# base_folder = os.path.expanduser('~/tmp/usgs-tegus-val-v8')
base_folder = os.path.expanduser('~/tmp/usgs-tegus-val-v8-300')
run_folders = os.listdir(base_folder)
run_folders = [os.path.join(base_folder,s) for s in run_folders]
run_folders = [s for s in run_folders if os.path.isdir(s)]

image_base = os.path.expanduser('~/data/usgs-kissel-training-yolo/val')
image_files = glob.glob(image_base + '/*.jpg')

prediction_files = []

# run_folder = run_folders[0]
for run_folder in run_folders:
    prediction_files_this_folder = glob.glob(run_folder+'/*predictions.json')
    assert len(prediction_files_this_folder) <= 1
    if len(prediction_files_this_folder) == 1:
        prediction_files.append(prediction_files_this_folder[0])        

md_format_prediction_files = []

# prediction_file = prediction_files[2]
for prediction_file in prediction_files:

    run_name = prediction_file.split('/')[-2]
    run_dir = os.path.dirname(prediction_file)
    
    output_file = os.path.join(run_dir,run_name + '_md-format.json')
    assert output_file != prediction_file
    
    yolo_output_to_md_output.yolo_json_output_to_md_output(
        yolo_json_file=prediction_file,
        image_folder=image_base,
        output_file=output_file,
        yolo_category_id_to_name=class_mapping_file,                              
        detector_name=run_name,
        image_id_to_relative_path=None,
        offset_yolo_class_ids=False)    
    
    md_format_prediction_files.append(output_file)

# ...for each prediction file


#%% Visualize results with the MD visualization pipeline

postprocessing_output_folder = os.path.expanduser('~/tmp/usgs-tegus-yolov8-300-previews')

from md_utils import path_utils

from api.batch_processing.postprocessing.postprocess_batch_results import (
    PostProcessingOptions, process_batch_results)

# prediction_file = md_format_prediction_files[0]
for prediction_file in md_format_prediction_files:
        
    assert '_md-format.json' in prediction_file
    base_task_name = os.path.basename(prediction_file).replace('_md-format.json','')

    options = PostProcessingOptions()
    options.image_base_dir = image_base
    options.include_almost_detections = True
    options.num_images_to_sample = 10000
    options.confidence_threshold = 0.4
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
    options.ground_truth_json_file = None
    options.separate_detections_by_category = True
    # options.sample_seed = 0
    
    options.parallelize_rendering = False
    options.parallelize_rendering_n_cores = 16
    options.parallelize_rendering_with_threads = False
    
    output_base = os.path.join(postprocessing_output_folder,
        base_task_name + '_{:.3f}'.format(options.confidence_threshold))
    
    os.makedirs(output_base, exist_ok=True)
    print('Processing to {}'.format(output_base))
    
    options.api_output_file = prediction_file
    options.output_dir = output_base
    ppresults = process_batch_results(options)
    html_output_file = ppresults.output_html_file
    path_utils.open_file(html_output_file)

# ...for each prediction file
