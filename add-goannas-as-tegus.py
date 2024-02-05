########
#
# add-goannas-as-tegus.py
#
# Use the goanna dataset described here:
#
# https://github.com/agentmorris/unsw-goannas
#
# ...to enhance the tegu training data in the usgs-tegus training set.
#
# Notes to self:
#
# The UNSW dataset includes 84,361 goanna images.
#
# The USGS dataset includes 1260 tegu images.
#
# The tegu class ID is 14 (although we will double-check this against the dataset.yml file).
#
########

#%% Imports and constants

import os
import shutil

from md_utils.path_utils import find_images
from tqdm import tqdm

tegu_yolo_base = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-yolo')
goanna_yolo_base = os.path.expanduser('~/data/unsw-alting-yolo')

assert os.path.isdir(tegu_yolo_base) and os.path.isdir(goanna_yolo_base)

# We will validate these against the dataset.yml files below
input_goanna_category = 2
output_tegu_category = 14
do_writes = True

# Double-check that images have been properly resized for efficient training
validate_image_sizes = False
expected_image_width = 1600


#%% Verify class IDs

from data_management.yolo_output_to_md_output import read_classes_from_yolo_dataset_file

tegu_dataset_file = os.path.join(tegu_yolo_base,'dataset.yaml')
assert os.path.isfile(tegu_dataset_file)

goanna_dataset_file = os.path.join(goanna_yolo_base,'dataset.yml')
assert os.path.isfile(goanna_dataset_file)

tegu_classes = read_classes_from_yolo_dataset_file(tegu_dataset_file)
goanna_classes = read_classes_from_yolo_dataset_file(goanna_dataset_file)

assert tegu_classes[output_tegu_category] == 'tegu'
assert goanna_classes[input_goanna_category] == 'goanna'


#%% Enumerate goanna images

unsw_yolo_images = find_images(goanna_yolo_base,return_relative_paths=True,recursive=True)
print('{} total images in the goanna yolo folder'.format(len(unsw_yolo_images)))

# Make sure all the annotation files exist
for fn_relative in tqdm(unsw_yolo_images):
    fn_relative_txt = os.path.splitext(fn_relative)[0] + '.txt'
    fn_abs_txt = os.path.join(goanna_yolo_base,fn_relative_txt)
    assert os.path.isfile(fn_abs_txt)

# Sample filename:
#
# train/dingo#BrendanAltingMLDP2023Images#PS11#CamA#PS11__CamA__2023-04-23__07-39-38(5).JPG

goanna_yolo_images = []
for fn_relative in unsw_yolo_images:
    tokens = fn_relative.split('/')
    assert len(tokens) == 2
    assert tokens[0] in ('train','val')
    species = tokens[1].split('#')[0]
    if species == 'goanna':
        goanna_yolo_images.append(fn_relative)
        
print('\nFound {} goanna images'.format(len(goanna_yolo_images)))        

    
#%% Copy goanna images

from md_visualization import visualization_utils as vis_utils

for fn_relative_unsw in tqdm(goanna_yolo_images):

    tokens = fn_relative_unsw.split('/')
    assert len(tokens) == 2
    split = tokens[0]
    file_basename = tokens[1]
    assert split in ('train','val')
    species = tokens[1].split('#')[0]
    assert species == 'goanna'
    
    source_file_abs = os.path.join(goanna_yolo_base,fn_relative_unsw)
    assert os.path.isfile(source_file_abs)
    
    if validate_image_sizes:
        im = vis_utils.load_image(source_file_abs)
        assert im.size[0] == expected_image_width
        
    target_split_folder_abs = os.path.join(tegu_yolo_base,split)
    assert os.path.isdir(target_split_folder_abs)
    
    # Create, for example:
    #
    # usgs-kissel-training-yolo/val/unsw-images/goanna#PSML2023-06#PS12#CamA#PS12__CamA__2023-04-26__14-14-52(12).JPG
    target_file_abs = os.path.join(target_split_folder_abs,'unsw-images',file_basename)
    
    # Re-write the annotation file
    txt_file_source = os.path.splitext(source_file_abs)[0] + '.txt'
    assert os.path.isfile(txt_file_source)

    with open(txt_file_source,'r') as f:
        annotation_lines = f.readlines()
    
    # annotation_line = annotation_lines[0]
    output_annotation_lines = []
    for annotation_line in annotation_lines:
        tokens = annotation_line.split()
        assert len(tokens) == 5
        input_category = int(tokens[0])
        assert input_category == input_goanna_category
        output_annotation_line = str(output_tegu_category) + ' ' + ' '.join(tokens[1:])
        output_annotation_lines.append(output_annotation_line)
                
    txt_file_dst = os.path.splitext(target_file_abs)[0] + '.txt'
    
    if do_writes:
        os.makedirs(os.path.dirname(txt_file_dst),exist_ok=True)
        with open(txt_file_dst,'w') as f:
            for output_annotation_line in output_annotation_lines:
                f.write(output_annotation_line + '\n')
        shutil.copyfile(source_file_abs,target_file_abs)

# ...for every goanna file