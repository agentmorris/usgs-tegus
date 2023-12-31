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
from md_visualization.visualization_utils import open_image


#%% Load MD results

# These 
md_file = os.path.expanduser('~/postprocessing/usgs-kissel/usgs-kissel-2023-09-11-aug-v5a.0.0/combined_api_outputs/usgs-kissel-2023-09-11-aug-v5a.0.0_detections.filtered_rde_0.075_0.850_25_0.200.json')

image_base = '/media/user/Tegu camera trap photos/Longterm and Live Trap'

with open(md_file,'r') as f:
    md_results = json.load(f)
    

#%% Map filenames to results

fn_to_results = {}

for im in tqdm(md_results['images']):
    fn_to_results[im['file']] = im
    

#%% Choose a category

target_folder = '/home/user/data/usgs-kissel-training/train/tegu'; confidence_threshold=None
# target_folder = '/home/user/data/usgs-kissel-training/train/american_cardinal'; confidence_threshold=0.3
# target_folder = '/home/user/data/usgs-kissel-training/train/crow'; confidence_threshold=0.4
# target_folder = '/home/user/data/usgs-kissel-training/train/green_iguana'; confidence_threshold=0.1
# target_folder = '/home/user/data/usgs-kissel-training/train/grey_catbird'; confidence_threshold=0.2
# target_folder = '/home/user/data/usgs-kissel-training/train/human'; confidence_threshold=0.4
# target_folder = '/home/user/data/usgs-kissel-training/train/northern_mockingbird'; confidence_threshold=0.2
# target_folder = '/home/user/data/usgs-kissel-training/train/other'; confidence_threshold=0.2
# target_folder = '/home/user/data/usgs-kissel-training/train/other_bird'; confidence_threshold=0.2
# target_folder = '/home/user/data/usgs-kissel-training/train/other_mammal'; confidence_threshold=0.4
# target_folder = '/home/user/data/usgs-kissel-training/train/raccoon'; confidence_threshold=0.3
# target_folder = '/home/user/data/usgs-kissel-training/train/rodent'; confidence_threshold=0.3
# target_folder = '/home/user/data/usgs-kissel-training/trian/snake'; confidence_threshold=0.2
# target_folder = '/home/user/data/usgs-kissel-training/train/unknown'; confidence_threshold=0.1
# target_folder = '/home/user/data/usgs-kissel-training/val/blanks_and_very_small_things'; confidence_threshold=None

assert os.path.isdir(target_folder)


#%% Resize files

cmd = 'mogrify -resize 1600x {}/*.JPG'.format(target_folder)
print(cmd)
import clipboard; clipboard.copy(cmd)


#%% Prepare labelme files

from api.batch_processing.postprocessing.md_to_labelme import get_labelme_dict_for_image
from md_utils.path_utils import find_images

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
print(cmd)
import clipboard; clipboard.copy(cmd)


#%% Resume labeling

cmd = 'python labelme {} --labels animal --linewidth 8 --last_updated_file ~/labelme-last-updated.txt --resume_from_last_update'.format(
    target_folder)
print(cmd)
import clipboard; clipboard.copy(cmd)


#%% Validate labelme files

from md_utils.path_utils import find_images

target_folder_images = find_images(target_folder,return_relative_paths=True)
target_folder_jsons = [fn for fn in os.listdir(target_folder) if fn.endswith('.json')]

target_folder_jsons_set = set(target_folder_jsons)
target_folder_images_set_no_extension = set([os.path.splitext(fn)[0] for fn in target_folder_images])

# This is a problem, it means we didn't annotate this image
for fn in target_folder_images:
    expected_json = os.path.splitext(fn)[0] + '.json'
    if expected_json not in target_folder_jsons_set:
        print('Could not find .json file for image {}'.format(fn))
        raise Exception('Missing .json')

# This is OK, it happens any time we delete the images after initially saving an annotation
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
if 'human' in target_folder:
    expected_label = 'person'
    
for image_fn_relative in target_folder_images:
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

# import clipboard; clipboard.copy(os.path.join(target_folder,empty_files[3]))
# import clipboard; clipboard.copy(os.path.join(target_folder,illegal_box_files[0]))


#%% Delete images with no boxes

empty_files_abs = [os.path.join(target_folder,fn) for fn in empty_files]
print('Deleting {} files'.format(len(empty_files_abs)))

for fn_abs in empty_files_abs:
    os.remove(fn_abs)

    
#%% Convert labelme to COCO, preview

# image_paths_to_include = multi_box_files
image_paths_to_include = target_folder_images

from detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP
from data_management.labelme_to_coco import labelme_to_coco
category_id_to_category_name = DEFAULT_DETECTOR_LABEL_MAP
output_file = output_file = os.path.expanduser('~/tmp/label_validation.json')
output_dict = labelme_to_coco(target_folder,output_file,
                              category_id_to_category_name=category_id_to_category_name,
                              relative_paths_to_include=image_paths_to_include)


##%% Validate

from data_management.databases import integrity_check_json_db

options = integrity_check_json_db.IntegrityCheckOptions()
    
options.baseDir = target_folder
options.bCheckImageSizes = True
options.bCheckImageExistence = True
options.bFindUnusedImages = True
options.bRequireLocation = False

sortedCategories, _, errorInfo = integrity_check_json_db.integrity_check_json_db(output_file,options)    


##%% Preview

from md_visualization import visualize_db
options = visualize_db.DbVizOptions()
options.parallelize_rendering = True
options.include_filename_links = True
options.show_full_paths = True
options.viz_size = (1280,-1)

html_file,_ = visualize_db.process_images(output_file,os.path.expanduser('~/tmp/labelme_to_coco_preview'),
                            target_folder,options)


from md_utils import path_utils # noqa
path_utils.open_file(html_file)


#%% Back up .json files

from datetime import datetime
image_folder = os.path.expanduser('~/data/usgs-kissel-training')
backup_folder = os.path.expanduser('~/data/usgs-labeling-backups')
backup_folder += '/{}'.format(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
commands = []
commands.append('pushd "{}"'.format(image_folder))
commands.append("find . -name '*.json' | cpio -pdm \"{}\"".format(backup_folder))
commands.append('popd')
commands.append('')

for s in commands:
    print(s)
    
import clipboard; clipboard.copy('\n'.join(commands))


#%% Scrap

if False:
    
    pass

    #%% Check some empty files
    
    empty_files_abs = [os.path.join(target_folder,fn) for fn in empty_files]
    fn = empty_files_abs[20]
    os.path.isfile(fn)
    from md_utils.path_utils import open_file; open_file(fn)


    #%% Preview the COCO .json file from a cold start
    
    target_folder = '/home/user/data/usgs-tegus/usgs-kissel-training'
    output_file = '/home/user/data/usgs-tegus/usgs-tegus.json'
    
    from data_management.databases import integrity_check_json_db

    options = integrity_check_json_db.IntegrityCheckOptions()
        
    options.baseDir = target_folder
    options.bCheckImageSizes = True
    options.bCheckImageExistence = True
    options.bFindUnusedImages = True
    options.bRequireLocation = False

    sortedCategories, _, errorInfo = integrity_check_json_db.integrity_check_json_db(output_file,options)
