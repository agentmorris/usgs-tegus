########
#
# prepare-yolo-training-set.py
#
# Given the COCO-formatted training set, prepare the final YOLO training data:
#
# * Split into train/val
# * Preview the train/val files to make sure everything looks OK
# * Convert to YOLO format
#
########

#%% Imports and constants

import os
import json

from data_management import coco_to_yolo
from md_utils.path_utils import insert_before_extension

input_folder_base = os.path.expanduser('~/data/usgs-kissel-training-resized')
input_folder_train = os.path.join(input_folder_base,'train')
input_folder_val = os.path.join(input_folder_base,'val')

output_folder_base = os.path.expanduser('~/data/usgs-kissel-training-yolo')
yolo_dataset_file = os.path.join(output_folder_base,'dataset.yaml')
os.makedirs(output_folder_base,exist_ok=True)

input_coco_file = os.path.expanduser('~/data/usgs-tegus.resized.json')

split_names = ('train','val')

blank_sample_p = 0.05


#%% Consistency checks

assert os.path.isdir(input_folder_train) and os.path.isdir(input_folder_val)
assert os.path.isfile(input_coco_file)


#%% Split the original COCO file into train/val files

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
    
    html_file,_ = visualize_db.process_images(subset_file,\
      os.path.expanduser('~/tmp/labelme_to_coco_preview-{}'.format(split_name)),
      input_folder,options)    
    
    path_utils.open_file(html_file)


#%% Convert the train/val sets to separate YOLO datasets, sampling the blanks as we go

from data_management import coco_to_yolo # noqa

import random
random.seed(1)

class_list_files = []
blank_category_codes = set()
# split_name = split_names[0]

for split_name in split_names:    
    
    blank_files = []
    included_blank_files = []
    subset_file = insert_before_extension(input_coco_file,split_name,separator='-')        
    split_output_folder = os.path.join(output_folder_base,split_name)
    
    print('\nCreating YOLO-formatted dataset in {}'.format(split_output_folder))
    
    images_to_exclude = []
    category_names_to_exclude = ['empty','other','unknown']
    
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
        if category_folder in category_names_to_exclude:
            images_to_exclude.append(image_fn)
        elif category_folder == 'blanks_and_very_small_things':
            category_code = image_fn.split('#')[1]
            blank_category_codes.add(category_code)
            # For now, exclude everything that isn't blank: insects, very small reptiles, etc.
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
                     allow_empty_annotations=False, # don't matter for coco_camera_traps data
                     clip_boxes=True,
                     images_to_exclude=images_to_exclude,
                     category_names_to_exclude=category_names_to_exclude,
                     write_output=True)
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


#%% Prepare symlinks for Bounding Box Editor

# split_name = 'val'
for split_name in split_names:    
    
    print('Creating symlinks for {}'.format(split_name))
    split_output_folder = os.path.join(output_folder_base,split_name)
    source_folder = split_output_folder
    images_folder = source_folder + '-images-symlinks'
    labels_folder = source_folder + '-labels-symlinks'
    coco_to_yolo.create_yolo_symlinks(source_folder,images_folder,labels_folder,
                                      class_list_file=class_list_files[0],
                                      class_list_output_name='object.data',
                                      force_lowercase_image_extension=True)


#%% Scrap

if False:
    
    pass

    #%% Read the code --> common mapping, as an interactive convenient

    spp_to_common_file = os.path.expanduser('~/data/usgs-kissel/usgs-kissel_spp_to_common.json')

    with open(spp_to_common_file,'r') as f:
        spp_to_common = json.load(f)


    #%% Print category code mappings for the blanks_and_very_small_things category
    
    for spp in blank_category_codes:
        print('{}: {}'.format(spp,spp_to_common[spp]))          

