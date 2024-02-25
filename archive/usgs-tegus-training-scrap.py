########
#
# usgs-tegus-training-scrap.py
#
# Scrap code that used to live at the bottom of train-usgs-tegu-detector.py;
# moved here for cleanliness, since it's not directly related to training.
#
#######

#%% Validation with YOLOv5

import os

model_base = os.path.expanduser('~/models/usgs-tegus')
training_run_names = ['usgs-tegus-yolov5x-231003-b8-img1280-e3002']

data_folder = os.path.expanduser('~/data/usgs-kissel-training-yolo')
image_size = 1280

# Note to self: validation batch size appears to have no impact on mAP
# (it shouldn't, but I verified that explicitly)
batch_size_val = 8

project_name = os.path.expanduser('~/tmp/usgs-tegus-val')
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
                
            cmd = cuda_string + \
                ' python val.py --img {} --batch-size {} --weights {} --project {} --name {}-{} --data {} --save-txt --save-json --save-conf --exist-ok'.format(
                image_size,batch_size_val,model_file,project_name,model_short_name,aug_string,data_file)        
            if augment:
                cmd += ' --augment'
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
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all      10382      11704      0.738      0.794      0.785      0.679
     american_cardinal      10382        185      0.796      0.854      0.854      0.754
              blue_jay      10382         61      0.878      0.885      0.897      0.784
                  crow      10382       4565      0.899      0.951       0.96      0.891
          green_iguana      10382          2      0.894        0.5      0.498      0.349
          grey_catbird      10382        358      0.724      0.916      0.863       0.75
                 human      10382        242      0.612      0.851      0.799      0.743
  northern_mockingbird      10382         71      0.507      0.859      0.699      0.644
            other_bird      10382        944      0.705      0.706      0.735       0.61
          other_mammal      10382        811      0.848      0.781      0.862      0.803
              ovenbird      10382         22      0.445      0.727      0.756      0.643
               raccoon      10382       1561      0.845      0.972      0.953      0.891
                rodent      10382       1453      0.817      0.945      0.932      0.756
                 snake      10382         25      0.426      0.356      0.279       0.24
                  tegu      10382        211      0.769      0.754      0.774      0.511
        turkey_vulture      10382       1193        0.9      0.849      0.911      0.811
"""

"""
best-aug
"""

"""
               Class     Images  Instances          P          R      mAP50   mAP50-95
                   all      10382      11704      0.728      0.801        0.8      0.698
     american_cardinal      10382        185       0.77      0.876      0.875      0.763
              blue_jay      10382         61       0.93      0.868      0.919      0.765
                  crow      10382       4565      0.897      0.956      0.967       0.89
          green_iguana      10382          2      0.947        0.5      0.495      0.495
          grey_catbird      10382        358      0.697      0.919       0.89      0.766
                 human      10382        242       0.65      0.851      0.826      0.782
  northern_mockingbird      10382         71      0.505      0.859      0.727      0.651
            other_bird      10382        944      0.739      0.711      0.755      0.615
          other_mammal      10382        811      0.853      0.767      0.852      0.798
              ovenbird      10382         22      0.365      0.727      0.741      0.632
               raccoon      10382       1561      0.859      0.971      0.967      0.914
                rodent      10382       1453      0.775      0.952      0.944      0.755
                 snake      10382         25      0.254       0.44      0.305      0.268
                  tegu      10382        211      0.761      0.768      0.811      0.547
        turkey_vulture      10382       1193      0.917      0.849      0.927      0.829
"""

"""
last-noaug
"""

"""
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all      10382      11704      0.793      0.766      0.779      0.675
     american_cardinal      10382        185      0.831      0.832       0.85      0.748
              blue_jay      10382         61      0.961      0.799      0.868       0.77
                  crow      10382       4565      0.932      0.944      0.961      0.893
          green_iguana      10382          2      0.894        0.5      0.498      0.348
          grey_catbird      10382        358      0.803      0.897       0.87      0.759
                 human      10382        242      0.668       0.81      0.793       0.73
  northern_mockingbird      10382         71      0.581      0.831      0.698      0.639
            other_bird      10382        944      0.793      0.657      0.732       0.61
          other_mammal      10382        811      0.892      0.737      0.851      0.794
              ovenbird      10382         22      0.556      0.727      0.715      0.619
               raccoon      10382       1561      0.871      0.965      0.952      0.886
                rodent      10382       1453      0.895      0.924      0.931      0.765
                 snake      10382         25      0.432      0.336      0.285      0.243
                  tegu      10382        211      0.847      0.701      0.776      0.501
        turkey_vulture      10382       1193      0.937      0.829      0.912      0.817
"""

"""
best-noaug
"""

"""
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all      10382      11704      0.782      0.776      0.795      0.695
     american_cardinal      10382        185      0.856      0.859       0.88       0.76
              blue_jay      10382         61      0.942      0.802      0.861      0.737
                  crow      10382       4565      0.929      0.952      0.967      0.891
          green_iguana      10382          2      0.947        0.5      0.505      0.498
          grey_catbird      10382        358      0.763      0.905      0.903      0.785
                 human      10382        242      0.698      0.835      0.819      0.768
  northern_mockingbird      10382         71      0.604      0.831       0.71      0.631
            other_bird      10382        944      0.802      0.641      0.744      0.611
          other_mammal      10382        811      0.902      0.727      0.855      0.801
              ovenbird      10382         22      0.446      0.727      0.721      0.635
               raccoon      10382       1561      0.877      0.944      0.966      0.911
                rodent      10382       1453      0.858      0.934      0.931      0.741
                 snake      10382         25      0.344        0.4      0.316       0.28
                  tegu      10382        211      0.823      0.754      0.816      0.545
        turkey_vulture      10382       1193      0.941      0.823      0.923      0.825
"""

#%% Convert YOLO val .json results to MD .json format

# pip install jsonpickle humanfriendly tqdm skicit-learn

import os
import glob
from data_management import yolo_output_to_md_output

class_mapping_file = os.path.expanduser('~/data/usgs-kissel-training-yolo/dataset.yaml')
                        
base_folder = os.path.expanduser('~/tmp/usgs-tegus-val')
run_folders = os.listdir(base_folder)
run_folders = [os.path.join(base_folder,s) for s in run_folders]
run_folders = [s for s in run_folders if os.path.isdir(s)]

image_base = os.path.expanduser('~/data/usgs-kissel-training-yolo/val')
image_files = glob.glob(image_base + '/*.jpg')

prediction_files = []

# run_folder = run_folders[0]
for run_folder in run_folders:
    prediction_files_this_folder = glob.glob(run_folder+'/*_predictions.json')
    assert len(prediction_files_this_folder) <= 1
    if len(prediction_files_this_folder) == 1:
        prediction_files.append(prediction_files_this_folder[0])        

md_format_prediction_files = []

# prediction_file = prediction_files[2]
for prediction_file in prediction_files:

    # ~/tmp/usgs-tegus-val/usgs-tegus-yolov5x-231003-b8-img1280-e3002-best-aug/usgs-tegus-yolov5x-231003-b8-img1280-e3002-best_predictions.json'
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

postprocessing_output_folder = os.path.expanduser('~/tmp/usgs-tegus-previews')

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
    options.num_images_to_sample = 7500
    options.confidence_threshold = 0.1
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
