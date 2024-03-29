########
#
# prepare-yolo-training-set.py
#
# Given the COCO-formatted training set, prepare the final YOLO training data:
#
# * Split into train/val (trivial, since the original folders are already sorted into train/val)
# * Sample blanks (from the USGS tegu training set)
# * Preview the train/val files to make sure everything looks OK
# * Convert to YOLO format
#
########

#%% Imports and constants

import os
import json
import random
import shutil

from data_management import coco_to_yolo
from md_utils.path_utils import insert_before_extension

input_folder_base = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-resized')
input_folder_train = os.path.join(input_folder_base,'train')
input_folder_val = os.path.join(input_folder_base,'val')

output_folder_base = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-yolo')
yolo_dataset_file = os.path.join(output_folder_base,'dataset.yaml')
os.makedirs(output_folder_base,exist_ok=True)

input_coco_file = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-resized.json')

split_names = ('train','val')

# The fraction of blank images we'll sample
#
# The original dataset has 60,590 boxes and ~39k total blank images
#
# Sample around 5k blanks, which we'll complement with blanks from LILA later
blank_sample_p = 0.125

random.seed(0)


#%% Consistency checks

assert os.path.isdir(input_folder_train) and os.path.isdir(input_folder_val)
assert os.path.isfile(input_coco_file)


#%% Split the original COCO file into train/val files

# This is not strictly necessary, it's just handy to have COCO files later for the 
# train and val data separately.

total_images_written = 0
total_annotations_written = 0

for split_name in split_names:
    
    subset_file = insert_before_extension(input_coco_file,split_name,separator='-')
    assert subset_file != input_coco_file
    
    with open(input_coco_file,'r') as f:
        d = json.load(f)
    
    n_images_original = len(d['images'])
    n_annotations_original = len(d['annotations'])
    
    images_to_keep = []
    
    for im in d['images']:
        assert im['file_name'].startswith('train') or im['file_name'].startswith('val')
        if im['file_name'].startswith(split_name):
            images_to_keep.append(im)
    assert len(images_to_keep) < len(d['images'])
    
    image_ids_to_keep = set([im['id'] for im in images_to_keep])
    
    assert len(image_ids_to_keep) == len(images_to_keep)
    assert len(images_to_keep) != 0
    
    annotations_to_keep = []
    
    for ann in d['annotations']:
        if ann['image_id'] in image_ids_to_keep:
            annotations_to_keep.append(ann)

    assert len(annotations_to_keep) != len(d['annotations'])
    assert len(annotations_to_keep) != 0
    
    d['images'] = images_to_keep
    d['annotations'] = annotations_to_keep
    
    with open(subset_file,'w') as f:
        json.dump(d,f,indent=1)
        
    print('Wrote {} of {} images ({} of {} annotations) to {}'.format(
        len(images_to_keep),n_images_original,
        len(annotations_to_keep),n_annotations_original,
        subset_file))

    total_annotations_written += len(annotations_to_keep)
    total_images_written += len(images_to_keep)

# ...for each split
    
print('Wrote a total of {} images and {} annotations'.format(
    total_images_written,total_annotations_written))


#%% Preview the new files

from data_management.databases import integrity_check_json_db
from md_visualization import visualize_db
from md_utils import path_utils

html_files = []
# split_name = split_names[0]
for split_name in split_names:
    
    ## Validate
    
    input_folder = input_folder_base
    subset_file = insert_before_extension(input_coco_file,split_name,separator='-')
    assert os.path.isfile(input_coco_file)
    
    options = integrity_check_json_db.IntegrityCheckOptions()
        
    options.baseDir = input_folder
    options.bCheckImageSizes = True
    options.bCheckImageExistence = True
    options.bFindUnusedImages = True
    options.bRequireLocation = False
    
    sorted_categories, _, error_info = \
        integrity_check_json_db.integrity_check_json_db(subset_file,options)    
        
    ## Preview
    
    options = visualize_db.DbVizOptions()
    options.parallelize_rendering = True
    options.viz_size = (900, -1)
    options.num_to_visualize = 5000
    options.htmlOptions['maxFiguresPerHtmlFile'] = 1000
    
    html_file,_ = visualize_db.visualize_db(subset_file,\
      os.path.expanduser('~/tmp/labelme_to_coco_preview-{}'.format(split_name)),
      input_folder,options)    
   
    html_files.append(html_file)

for s in html_files:    
    path_utils.open_file(html_file)

# import clipboard; clipboard.copy(html_files[0])
# import clipboard; clipboard.copy(html_files[1])


#%% Convert the train/val sets to separate YOLO datasets, sampling blanks as we go

class_list_files = []
blank_category_codes = set()

yolo_conversion_dry_run = False

# split_name = split_names[0]
for split_name in split_names:    
    
    blank_files = []
    included_blank_files = []
    subset_file = insert_before_extension(input_coco_file,split_name,separator='-')        
    split_output_folder = os.path.join(output_folder_base,split_name)
    
    print('\nCreating YOLO-formatted dataset in {}'.format(split_output_folder))
    
    images_to_exclude = []
    category_names_to_exclude = ['other','unknown']
    
    with open(subset_file,'r') as f:
        d = json.load(f)
    image_filenames = [im['file_name'] for im in d['images']]
        
    # image_fn = image_filenames[0]
    for image_fn in image_filenames:
        
        # E.g.:
        # 
        # 'train/blanks_and_very_small_things/blanks_and_very_small_things#
        # AnCa#110.04_C104#2017-2019#C104_110.04#(15) 01MAR - 20MAR18 ARB EVS#MFDC8278.JPG'
        
        assert image_fn.startswith(split_name)
        category_folder = image_fn.split('/')[1]
        
        # For some reason I had "empty" in "category_names_to_exclude", which doesn't make sense.
        # "empty" is a category in the .json file, but this is referring to the category folders,
        # where the closest thing is "blanks_and_very_small_things".
        # 
        # I'm just assert'ing here to make sure 2023-me didn't know something that 2024-me doesn't.
        assert category_folder != 'empty'
        
        if category_folder in category_names_to_exclude:
            images_to_exclude.append(image_fn)
            
        elif category_folder == 'blanks_and_very_small_things':
            
            category_code = image_fn.split('#')[1]
            blank_category_codes.add(category_code)
            
            # For now, exclude everything in the "blanks_and_very_smal_things" category that isn't blank: 
            # insects, very small reptiles, etc.
            if category_code != 'blank':
                images_to_exclude.append(image_fn)            
            blank_files.append(image_fn)
            
            p = random.random()
            if p < blank_sample_p:
                included_blank_files.append(image_fn)
            else:
                images_to_exclude.append(image_fn)
    
    # input_image_folder = input_folder_base; output_folder = split_output_folder; input_file = subset_file
    return_info = coco_to_yolo.coco_to_yolo(input_folder_base,split_output_folder,subset_file,
                     source_format='coco_camera_traps',
                     overwrite_images=False,
                     create_image_and_label_folders=False,
                     class_file_name='classes.txt',
                     allow_empty_annotations=False, # doesn't matter for coco_camera_traps data
                     clip_boxes=True,
                     images_to_exclude=images_to_exclude,
                     category_names_to_exclude=category_names_to_exclude,
                     write_output=(not yolo_conversion_dry_run))
    
    class_list_files.append(return_info['class_list_filename'])
    
    print('Included {} of {} blank files ({:.2f}%)'.format(
        len(included_blank_files),len(blank_files),
        100 * len(included_blank_files)/len(blank_files)))
    
# ...for each split

# Make sure the two datasets (train/val) have identical class lists
class_list = None
for fn in class_list_files:
    with open(fn,'r') as f:
        current_class_list = f.readlines()
        current_class_list = [s.strip() for s in current_class_list]
    if class_list is None:
        class_list = current_class_list
    else:        
        assert class_list == current_class_list
            

#%% Generate the YOLOv5 dataset.yaml file

coco_to_yolo.write_yolo_dataset_file(yolo_dataset_file,
                                     dataset_base_dir=output_folder_base,
                                     class_list=class_list,
                                     train_folder_relative='train',
                                     val_folder_relative='val',
                                     test_folder_relative=None)


#%% Copy the dataset files for Bounding Box editor

# split_name = 'val'
for split_name in split_names:    
    
    print('Preparing {} for BBE'.format(split_name))
    split_output_folder = os.path.join(output_folder_base,split_name)
    target_class_list_file = os.path.join(split_output_folder,'object.data')
    shutil.copyfile(class_list_files[0],target_class_list_file)


#%% Optional extra resizing pass

import os
from md_visualization.visualization_utils import resize_image_folder

input_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-yolo-1600')
output_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-yolo')

resize_image_folder(input_folder,output_folder,
                    target_width=1280,verbose=False,quality=85,no_enlarge_width=True,
                    pool_type='process',n_workers=12)

# Copy annotation files

from md_utils.path_utils import recursive_file_list
import shutil
from tqdm import tqdm

all_files_relative = recursive_file_list(input_folder,return_relative_paths=True)
annotation_files_relative = [fn for fn in all_files_relative if fn.endswith('.txt')]
print('Found {} annotation files (of {})'.format(
    len(annotation_files_relative),len(all_files_relative)))

# fn_relative = annotation_files_relative[-1]
for fn_relative in tqdm(annotation_files_relative):
    source_fn_abs = os.path.join(input_folder,fn_relative)
    target_fn_abs = os.path.join(output_folder,fn_relative)
    os.makedirs(os.path.dirname(target_fn_abs),exist_ok=True)
    shutil.copyfile(source_fn_abs,target_fn_abs)

    
#%% Summarize folder content

import os
from md_utils.path_utils import find_images

data_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-yolo-1600')

images = find_images(data_folder,recursive=True)


print('Found {} images in {}'.format(len(images),data_folder))

usgs_images = [fn for fn in images if (('unsw' not in fn) and ('lila-blank' not in fn))]
usgs_blanks = [fn for fn in usgs_images if 'blanks' in fn]
usgs_tegus = [fn for fn in usgs_images if 'tegu#' in fn]
print('Found {} USGS images ({} blank) ({} tegus)'.\
      format(len(usgs_images),len(usgs_blanks), len(usgs_tegus)))

lila_blanks = [fn for fn in images if 'lila-blank' in fn]
print('Found {} LILA-blank images'.format(len(lila_blanks)))

unsw_images = [fn for fn in images if 'unsw' in fn]
print('Found {} UNSW images'.format(len(unsw_images)))
for s in unsw_images:
    assert 'goanna' in s
    assert 'blank' not in s

print('')

for split_name in ('train','val'):
    split_folder = os.path.join(data_folder,split_name)
    assert os.path.isdir(split_folder)
    split_images = find_images(split_folder,recursive=True)
    print('Found {} images for split {}'.format(len(split_images),split_name))


