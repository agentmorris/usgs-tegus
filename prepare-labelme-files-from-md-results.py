########
#
# prepare-labelme-files-from-md-results.py
#
# Prepopulate labelme annotation files based on MegaDetector results, then - after review
# and cleanup in labelme - convert to COCO format and generate HTML previews of the labels.
#
# This script is not intended to be run top to bottom; it's a notebook that I used to prepare
# and label one category at a time.
#
########

#%% Constants and imports

import os
import json

from tqdm import tqdm
import clipboard

from api.batch_processing.postprocessing.md_to_labelme import get_labelme_dict_for_image
from data_management.databases import integrity_check_json_db
from data_management.labelme_to_coco import labelme_to_coco
from data_management.resize_coco_dataset import resize_coco_dataset
from detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP
from md_visualization import visualize_db
from md_visualization.visualization_utils import open_image
from md_utils import path_utils
from md_utils.path_utils import find_images
from md_utils.path_utils import recursive_file_list

output_file_pre_resize = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training.json')
output_file_resized = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-resized.json')
md_file = os.path.expanduser('~/postprocessing/usgs-kissel/usgs-kissel-2023-09-11-aug-v5a.0.0/combined_api_outputs/usgs-kissel-2023-09-11-aug-v5a.0.0_detections.filtered_rde_0.075_0.850_25_0.200.json')
image_base = '/media/user/Tegu camera trap photos/Longterm and Live Trap'

# This is the folder created by usgs-tegus-training-data-prep
input_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training')
assert os.path.isdir(input_folder)

resized_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-resized')
os.makedirs(resized_folder,exist_ok=True)


#%% Load MD results

with open(md_file,'r') as f:
    md_results = json.load(f)
    
print('Loaded MD results for {} images'.format(len(md_results['images'])))


#%% Map filenames to results

fn_to_results = {}

for im in tqdm(md_results['images']):
    fn_to_results[im['file']] = im


#%% Choose a category to annotate

target_folder = os.path.join(input_folder,'train/tegu'); confidence_threshold=None
# target_folder = os.path.join(input_folder,'train/american_cardinal'); confidence_threshold=0.3
# target_folder = os.path.join(input_folder,'train/crow'); confidence_threshold=0.4
# target_folder = os.path.join(input_folder,'train/green_iguana'); confidence_threshold=0.1
# target_folder = os.path.join(input_folder,'train/grey_catbird'); confidence_threshold=0.2
# target_folder = os.path.join(input_folder,'train/human'); confidence_threshold=0.4
# target_folder = os.path.join(input_folder,'train/northern_mockingbird'); confidence_threshold=0.2
# target_folder = os.path.join(input_folder,'train/other'); confidence_threshold=0.2
# target_folder = os.path.join(input_folder,'train/other_bird'); confidence_threshold=0.2
# target_folder = os.path.join(input_folder,'train/other_mammal'); confidence_threshold=0.4
# target_folder = os.path.join(input_folder,'train/raccoon'); confidence_threshold=0.3
# target_folder = os.path.join(input_folder,'train/rodent'); confidence_threshold=0.3
# target_folder = os.path.join(input_folder,'trian/snake'); confidence_threshold=0.2
# target_folder = os.path.join(input_folder,'train/unknown'); confidence_threshold=0.1
# target_folder = os.path.join(input_folder,'train/blanks_and_very_small_things)'; confidence_threshold=None

assert os.path.isdir(target_folder)


#%% Resize files (one category folder)

cmd = 'mogrify -resize 1600x {}/*.JPG'.format(target_folder)
print(cmd); clipboard.copy(cmd)


#%% Prepare labelme files

destination_image_files = find_images(target_folder,return_relative_paths=True)
source_file_to_destination_image_file = {}

category_id_to_name = md_results['detection_categories']
info = md_results['info']

json_files_written = []
json_files_skipped = []

overwrite_json_files = False

# destination_image_file_relative = destination_image_files[0]
for destination_image_file_relative in tqdm(destination_image_files):

    source_file_relative = '#'.join(destination_image_file_relative.split('#')[3:]).replace('#','/')
    im = fn_to_results[source_file_relative]
    
    source_file_abs = os.path.join(image_base,source_file_relative)
    assert os.path.isfile(source_file_abs)
    
    destination_image_file_abs = os.path.join(target_folder,destination_image_file_relative)
    pil_im = open_image(destination_image_file_abs)
    im['width'] = pil_im.width
    im['height'] = pil_im.height
    
    labelme_dict = get_labelme_dict_for_image(im,destination_image_file_relative,
                                              category_id_to_name,info,confidence_threshold=confidence_threshold)
    
    target_file_relative = os.path.splitext(destination_image_file_relative)[0] + '.json'
    target_file_abs = os.path.join(target_folder,target_file_relative)
    if (not overwrite_json_files) and (os.path.isfile(target_file_abs)):
        # print('Bypassing write to existing .json file {}'.format(target_file_abs))
        json_files_skipped.append(target_file_abs)
    else:
        json_files_written.append(target_file_abs)
        with open(target_file_abs,'w') as f:
            json.dump(labelme_dict,f,indent=1)            

# ...for each image

print('\nWrote {} .json files (skipped {} that already existed)'.format(
    len(json_files_written),len(json_files_skipped)))


#%% Label (a new batch)

cmd = 'python labelme {} --labels animal --linewidth 8 --last_updated_file ~/labelme-last-updated.txt'.format(
    target_folder)
print(cmd); clipboard.copy(cmd)


#%% Resume labeling

cmd = 'python labelme {} --labels animal --linewidth 8 --last_updated_file ~/labelme-last-updated.txt --resume_from_last_update'.format(
    target_folder)
print(cmd); clipboard.copy(cmd)


#%% Validate labelme files

# If you set 'target_folder' to point to the base folder, this cell will operate on the whole dataset
#
# target_folder = os.path.join(input_folder,'train/tegu')
# target_folder = input_folder

target_folder_images = find_images(target_folder,return_relative_paths=True,recursive=True)
target_folder_jsons = recursive_file_list(target_folder,return_relative_paths=True)
target_folder_jsons = [fn for fn in target_folder_jsons if fn.endswith('.json')]

target_folder_jsons_set = set(target_folder_jsons)
target_folder_images_set_no_extension = set([os.path.splitext(fn)[0] for fn in target_folder_images])

tokens_not_expected_to_have_annotations = ['blanks_and_very_small_things']

def is_annotation_expected(fn):
    for s in tokens_not_expected_to_have_annotations:
        if s in fn:
            return False
    return True        
    
# Missing .json files are a problem, it means we didn't annotate this image
for image_fn_relative in target_folder_images:
    if not is_annotation_expected(image_fn_relative):
        continue
    expected_json = os.path.splitext(image_fn_relative)[0] + '.json'
    if expected_json not in target_folder_jsons_set:
        print('Could not find .json file for image {}'.format(image_fn_relative))
        raise Exception('Missing .json for image {}'.format(image_fn_relative))

# This is OK, it happens any time we delete the images after initially saving an annotation
#
# Leaving the loop here just so I don't ask later whether I should be checking for this,
# but it is a no-op.
for fn in target_folder_jsons:
    expected_image_no_extension = os.path.splitext(fn)[0]
    if expected_image_no_extension not in target_folder_images_set_no_extension:
        # print('Could not find image file for .json {}'.format(fn))
        pass

# image_fn_relative = target_folder_images[0]
total_shapes = 0

multi_box_files = []
empty_files = []
illegal_box_files = set()

expected_label = 'animal'
    
for image_fn_relative in tqdm(target_folder_images):

    if 'human' in image_fn_relative:
        expected_label = 'person'
    else:
        expected_label = 'animal'
    
    if not is_annotation_expected(image_fn_relative):
        continue
    
    expected_json_relative = os.path.splitext(image_fn_relative)[0] + '.json'
    expected_json_abs = os.path.join(target_folder,expected_json_relative)
    image_fn_abs = os.path.join(target_folder,image_fn_relative)
    with open(expected_json_abs,'r') as f:
        d = json.load(f)
    if len(d['shapes']) > 1:
        multi_box_files.append(image_fn_relative)
    elif len(d['shapes']) == 0:
        empty_files.append(image_fn_relative)
    for shape in d['shapes']:
        if shape['label'] != expected_label:
            illegal_box_files.add(image_fn_relative)
            print('Illegal label {} in {}'.format(shape['label'],image_fn_abs))
        assert len(shape['points']) == 2
        
illegal_box_files = sorted(list(illegal_box_files))
        
print('{} files total'.format(len(target_folder_images)))
print('{} files have multiple boxes'.format(len(multi_box_files)))
print('{} files have no boxes'.format(len(empty_files)))
print('{} files have illegal boxes'.format(len(illegal_box_files)))

# clipboard.copy(os.path.join(target_folder,empty_files[0]))
# clipboard.copy(os.path.join(target_folder,illegal_box_files[0]))


#%% Delete images with no boxes

empty_files_abs = [os.path.join(target_folder,fn) for fn in empty_files]
print('Deleting {} files'.format(len(empty_files_abs)))

for fn_abs in empty_files_abs:
    os.remove(fn_abs)

    
#%% Convert labelme to COCO, preview (single category)

target_folder_images = find_images(target_folder,return_relative_paths=True,recursive=True)

# image_paths_to_include = multi_box_files
image_paths_to_include = target_folder_images

category_id_to_category_name = DEFAULT_DETECTOR_LABEL_MAP
output_file = os.path.expanduser('~/tmp/single_category_label_validation.json')
output_dict = labelme_to_coco(input_folder=target_folder,
                              output_file=output_file,
                              category_id_to_category_name=category_id_to_category_name,
                              info_struct=None,
                              relative_paths_to_include=image_paths_to_include,
                              relative_paths_to_exclude=None,
                              use_folders_as_labels=False,
                              recursive=False, 
                              no_json_handling='error',
                              validate_image_sizes=True,
                              right_edge_quantization_threshold=0.015)


##%% Validate

options = integrity_check_json_db.IntegrityCheckOptions()
    
options.baseDir = target_folder
options.bCheckImageSizes = True
options.bCheckImageExistence = True
options.bFindUnusedImages = True
options.bRequireLocation = False

sortedCategories, _, errorInfo = integrity_check_json_db.integrity_check_json_db(output_file,options)    


##%% Preview

options = visualize_db.DbVizOptions()
options.parallelize_rendering = True
options.include_filename_links = True
options.show_full_paths = True
options.viz_size = (1280,-1)
options.htmlOptions['maxFiguresPerHtmlFile'] = 2000

html_file,_ = visualize_db.visualize_db(output_file,os.path.expanduser('~/tmp/labelme_to_coco_preview'),
                            target_folder,options)

path_utils.open_file(html_file)
# clipboard.copy(html_file)


#%% Convert labelme to COCO, preview (whole dataset)

category_id_to_category_name = DEFAULT_DETECTOR_LABEL_MAP
output_dict = labelme_to_coco(input_folder=input_folder,
                              output_file=output_file_pre_resize,
                              category_id_to_category_name=None,
                              empty_category_name='empty',
                              empty_category_id=None,
                              info_struct=None,
                              relative_paths_to_include=None,
                              relative_paths_to_exclude=None,
                              use_folders_as_labels=True,
                              recursive=True,                              
                              # At this point, anything without a .json file is considered empty;
                              # we'll make sure later that these are all in the "blanks" folder
                              no_json_handling='empty',
                              validate_image_sizes=True,
                              right_edge_quantization_threshold=0.015)


##%% Make sure all the empty images are where they belong

with open(output_file_pre_resize,'r') as f:
    coco_db = json.load(f)

# This is only true because of the behavior of labelme_to_coco, it's not fundamental to the format
assert coco_db['categories'][0]['name'] == 'empty'
assert coco_db['categories'][0]['id'] == 0
empty_category_id = 0

image_id_to_image = {im['id']:im for im in coco_db['images']}

n_blanks = 0

# ann = coco_db['annotations'][0]
for ann in coco_db['annotations']:
    image_id = ann['image_id']
    im = image_id_to_image[image_id]
    if ann['category_id'] == empty_category_id:
        assert 'bbox' not in ann
        assert 'blank' in im['file_name']
        n_blanks += 1
    else:
        assert 'bbox' in ann
        assert 'blank' not in im['file_name']

print('Validated {} blank images'.format(n_blanks))


##%% Validate

options = integrity_check_json_db.IntegrityCheckOptions()
    
options.baseDir = input_folder
options.bCheckImageSizes = True
options.bCheckImageExistence = True
options.bFindUnusedImages = True
options.bRequireLocation = False

sortedCategories, _, errorInfo = \
    integrity_check_json_db.integrity_check_json_db(output_file_pre_resize,options)    


##%% Preview

options = visualize_db.DbVizOptions()
options.parallelize_rendering = True
options.include_filename_links = True
options.show_full_paths = True
options.viz_size = (1280,-1)
options.num_to_visualize = 5000
options.htmlOptions['maxFiguresPerHtmlFile'] = 2000


html_file,_ = visualize_db.visualize_db(output_file_pre_resize,
                                        os.path.expanduser('~/tmp/labelme_to_coco_preview'),
                                        input_folder,options)

path_utils.open_file(html_file)
# clipboard.copy(html_file)


#%% Resize the database (images and .json file)

resized_dataset = resize_coco_dataset(
                    input_folder=input_folder,
                    input_filename=output_file_pre_resize,
                    output_folder=resized_folder,
                    output_filename=output_file_resized,
                    target_size=(1600,-1),
                    correct_size_image_handling='copy',
                    right_edge_quantization_threshold=0.015)


#%% Validate and preview the resized images

with open(output_file_resized,'r') as f:
    coco_db = json.load(f)

# This is only true because of the behavior of labelme_to_coco, it's not fundamental to the format
assert coco_db['categories'][0]['name'] == 'empty'
assert coco_db['categories'][0]['id'] == 0
empty_category_id = 0


##%% Make sure all the empty images are where they belong

image_id_to_image = {im['id']:im for im in coco_db['images']}

n_blanks = 0

# ann = coco_db['annotations'][0]
for ann in coco_db['annotations']:
    image_id = ann['image_id']
    im = image_id_to_image[image_id]
    if ann['category_id'] == empty_category_id:
        assert 'bbox' not in ann
        assert 'blank' in im['file_name']
        n_blanks += 1
    else:
        assert 'bbox' in ann
        assert 'blank' not in im['file_name']

print('Validated {} blank images'.format(n_blanks))


##%% Make sure all the images are the right size

assert all([im['width'] == 1600 for im in coco_db['images']])


##%% Count train and val annotations

n_total_annotations = 0
n_val_annotations = 0
n_train_annotations = 0

# ann = coco_db['annotations'][0]
for ann in coco_db['annotations']:
    if 'bbox' not in ann:
        continue
    assert 'blank' not in ann['image_id']
    n_total_annotations += 1
    if 'train/' in ann['image_id']:
        n_train_annotations += 1
    if 'val/' in ann['image_id']:
        n_val_annotations += 1
        
print('{} total annotations'.format(n_total_annotations))
print('{} val annotations'.format(n_val_annotations))
print('{} train annotations'.format(n_train_annotations))


##%% Validate

options = integrity_check_json_db.IntegrityCheckOptions()
    
options.baseDir = resized_folder
options.bCheckImageSizes = True
options.bCheckImageExistence = True
options.bFindUnusedImages = True
options.bRequireLocation = False

sortedCategories, _, errorInfo = \
    integrity_check_json_db.integrity_check_json_db(output_file_resized,options)    

n_total_annotations_ic = 0
for c in sortedCategories:
    if c['name'] != 'empty':
        n_total_annotations_ic += c['_count']

assert n_total_annotations_ic == n_total_annotations


##%% Preview

options = visualize_db.DbVizOptions()
options.parallelize_rendering = True
options.include_filename_links = True
options.show_full_paths = True
options.viz_size = (1280,-1)
options.num_to_visualize = 5000
options.htmlOptions['maxFiguresPerHtmlFile'] = 1000

html_file,_ = visualize_db.visualize_db(output_file_resized,
                                        os.path.expanduser('~/tmp/labelme_to_coco_resized'),
                                        resized_folder,options)

path_utils.open_file(html_file)
# clipboard.copy(html_file)


#%% Scrap

if False:
    
    pass

    #%% Back up labelme .json files

    from datetime import datetime
    image_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training')
    backup_folder = os.path.expanduser('~/data/usgs-tegus/usgs-labeling-backups')
    backup_folder += '/{}'.format(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
    commands = []
    commands.append('pushd "{}"'.format(image_folder))
    commands.append("find . -name '*.json' | cpio -pdm \"{}\"".format(backup_folder))
    commands.append('popd')
    commands.append('')

    for s in commands:
        print(s)
        
    import clipboard; clipboard.copy('\n'.join(commands))


    #%% Check some empty files
    
    empty_files_abs = [os.path.join(target_folder,fn) for fn in empty_files]
    fn = empty_files_abs[20]
    os.path.isfile(fn)
    from md_utils.path_utils import open_file; open_file(fn)


    #%% Integrity-check the COCO .json file from a cold start
    
    target_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training')
    output_file = os.path.expanduser('~/data/usgs-tegus/usgs-tegus.json')
    
    options = integrity_check_json_db.IntegrityCheckOptions()
        
    options.baseDir = target_folder
    options.bCheckImageSizes = True
    options.bCheckImageExistence = True
    options.bFindUnusedImages = True
    options.bRequireLocation = False

    sortedCategories, _, errorInfo = \
        integrity_check_json_db.integrity_check_json_db(output_file,options)


    #%% Compare two nominally-identical .json files produced by this script at different times
    
    fn0 = os.path.expanduser('~/data/usgs-tegus/archive/usgs-tegus.json')
    fn1 = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training.json')
    
    assert os.path.isfile(fn0)
    assert os.path.isfile(fn1)
    
    with open(fn0,'r') as f:
        d0 = json.load(f)
    with open(fn1,'r') as f:
        d1 = json.load(f)
        
    image_files_0 = [im['file_name'] for im in d0['images']]
    image_files_1 = [im['file_name'] for im in d1['images']]
    assert set(image_files_0) == set(image_files_1)
    
    options = integrity_check_json_db.IntegrityCheckOptions()    
    options.baseDir = None
    options.bCheckImageSizes = False
    options.bCheckImageExistence = False
    options.bFindUnusedImages = False
    options.bRequireLocation = False

    sortedCategories0, _, errorInfo = integrity_check_json_db.integrity_check_json_db(fn0,options)
    sortedCategories1, _, errorInfo = integrity_check_json_db.integrity_check_json_db(fn1,options)
    
    name_to_count_0 = {c['name']:c['_count'] for c in sortedCategories0}
    name_to_count_1 = {c['name']:c['_count'] for c in sortedCategories1}
    assert name_to_count_0 == name_to_count_1