########
#
# review-usgs-tegu-results.py
#
# Create data review pages for USGS tegu validation data
#
########

#%% Imports and constants

import os
import sys
import json
import stat

from tqdm import tqdm

from ultralytics import YOLO

from data_management import yolo_output_to_md_output
from md_utils.path_utils import open_file

# Import yolov5 tools for printing model information

# Remove all YOLOv5 folders from the PYTHONPATH, to make sure the ultralytics
# package load the correct YOLOv5 repo.
keep_path = []
for s in sys.path:
    if 'git/yolov5' in s:
        print('Removing {} from PYTHONPATH'.format(s))
    else:
        keep_path.append(s)
sys.path = keep_path

yolov5_dir = os.path.expanduser('~/git/yolov5-training')
assert os.path.isdir(yolov5_dir)
if yolov5_dir not in sys.path:
    sys.path.append(yolov5_dir)

# data_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-yolo-1600-usgs-only')
# data_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-resized')
data_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training')
assert os.path.isdir(data_folder)
val_folder = os.path.join(data_folder,'val')
assert os.path.isdir(val_folder)

# val_file_coco = os.path.join(data_folder,'usgs-kissel-training-resized-val.json')
val_file_coco = os.path.join(data_folder,'usgs-kissel-training-val-only.json')
assert os.path.isfile(val_file_coco)

results_base_folder = os.path.expanduser('~/tmp/usgs-tegus/model-comparison')
preview_base_folder = os.path.join(results_base_folder,'preview')
os.makedirs(preview_base_folder,exist_ok=True)

n_gpus = 2
augment = False


#%% Define candidate models

candidate_models = {}

candidate_models['default'] = {}
candidate_models['default']['confidence_thresholds'] = {'default':0.5,'tegu':0.45}
candidate_models['default']['rendering_confidence_thresholds'] = {'default':0.4,'tegu':0.05}
candidate_models['default']['model_file'] = None

model_base_folder = '/mnt/c/users/dmorr/models/usgs-tegus'
# model_base_folder = os.path.expanduser('~/models/usgs-tegus')
# model_base_folder = None
assert os.path.isdir(model_base_folder)

# classes_data_type

## USGS-only yolov5x6

model_name = 'all-classes_usgs-only_yolov5x6'
candidate_models[model_name] = {}
candidate_models[model_name]['model_file'] = \
    os.path.join(model_base_folder,
                 'usgs-tegus-yolov5x-231003-b8-img1280-e3002/weights/usgs-tegus-yolov5x-231003-b8-img1280-e3002-best-stripped.pt')
candidate_models[model_name]['image_size'] = 1280
candidate_models[model_name]['model_type'] = 'yolov5'

## USGS-only yolov8

model_name = 'all-classes_usgs-only_yolov8x'
candidate_models[model_name] = {}
candidate_models[model_name]['model_file'] = \
    os.path.join(model_base_folder,
                 'usgs-tegus-yolov8x-2023.10.26-b-1-img640-e300/weights/usgs-tegus-yolov8x-2023.10.26-b-1-img640-e300-best.pt')
candidate_models[model_name]['image_size'] = 640
candidate_models[model_name]['model_type'] = 'yolov8'
candidate_models[model_name]['confidence_thresholds'] = {'default':0.5,'tegu':0.25}

## yolov5s models

model_name = 'tegu-human_usgs-goannas-lilablanks_yolov5s'
candidate_models[model_name] = {}
candidate_models[model_name]['model_file'] = \
    os.path.join(model_base_folder,
                 'usgs-tegus-tegu_human_w_goanna_lilablanks-im448-e300-b128-yolov5s/weights/usgs-tegus-tegu_human_w_goanna_lilablanks-im448-e300-b128-yolov5s-best.pt')
candidate_models[model_name]['image_size'] = 448
candidate_models[model_name]['confidence_thresholds'] = {'default':0.1,'tegu':0.05}
candidate_models[model_name]['model_type'] = 'yolov5'

model_name = 'tegu-human_usgs-only_yolov5s'
candidate_models[model_name] = {}
candidate_models[model_name]['model_file'] = \
    os.path.join(model_base_folder,
                 'usgs-tegus-tegu_human-im448-e250-b64-yolov5s/weights/usgs-tegus-tegu_human-im448-e250-b64-yolov5s-best.pt')
candidate_models[model_name]['image_size'] = 448    
candidate_models[model_name]['confidence_thresholds'] = {'default':0.1,'tegu':0.05}
candidate_models[model_name]['model_type'] = 'yolov5'

## lilablanks

model_name = 'all-classes_usgs-lilablanks_yolov5x6-20240223'
candidate_models[model_name] = {}
candidate_models[model_name]['model_file'] = \
    os.path.join(model_base_folder,
                 'usgs-tegus-yolov5-lilablanks-20240205101724-b8-img1280-e3006/checkpoint-20240223/usgs-tegus-yolov5-lilablanks-20240205101724-b8-img1280-e3006-best-cp-20240223-stripped.pt')
candidate_models[model_name]['image_size'] = 1280
candidate_models[model_name]['model_type'] = 'yolov5'

model_name = 'all-classes_usgs-lilablanks_yolov5x6-20240306'
candidate_models[model_name] = {}
candidate_models[model_name]['model_file'] = \
    os.path.join(model_base_folder,
                 'usgs-tegus-yolov5-lilablanks-20240205101724-b8-img1280-e3006/checkpoint-20240306/usgs-tegus-yolov5-lilablanks-20240205101724-b8-img1280-e3006-best-cp-20240306-stripped.pt')
candidate_models[model_name]['image_size'] = 1280
candidate_models[model_name]['model_type'] = 'yolov5'

model_name = 'all-classes_usgs-lilablanks_yolov5x6-20240317-training-complete'
candidate_models[model_name] = {}
candidate_models[model_name]['model_file'] = \
    os.path.join(model_base_folder,
                 'usgs-tegus-yolov5-lilablanks-20240205101724-b8-img1280-e3006/checkpoint-training-complete-20230317/usgs-tegus-yolov5-lilablanks-20240205101724-b8-img1280-e3006-best-cp-training-complete-20230317-stripped.pt')
candidate_models[model_name]['image_size'] = 1280
candidate_models[model_name]['model_type'] = 'yolov5'


## goannas+lilablanks

model_name = 'all-classes_usgs-goannas-lilablanks_yolov5x6-20240223'
candidate_models[model_name] = {}
candidate_models[model_name]['model_file'] = \
    os.path.join(model_base_folder,
                 'usgs-tegus-yolov5-lilablanks_goannas-20240205105940-b8-img1280-e3003/checkpoint-20240223/usgs-tegus-yolov5-lilablanks_goannas-20240205105940-b8-img1280-e3003-best-cp-20240223-stripped.pt')    
candidate_models[model_name]['image_size'] = 1280    
candidate_models[model_name]['confidence_thresholds'] = {'default':0.5,'tegu':0.475}
candidate_models[model_name]['model_type'] = 'yolov5'

model_name = 'all-classes_usgs-goannas-lilablanks_yolov5x6-20240306'
candidate_models[model_name] = {}
candidate_models[model_name]['model_file'] = \
    os.path.join(model_base_folder,
                 'usgs-tegus-yolov5-lilablanks_goannas-20240205105940-b8-img1280-e3003/checkpoint-20240306/usgs-tegus-yolov5-lilablanks_goannas-20240205105940-b8-img1280-e3003-best-cp-20240306-stripped.pt')
candidate_models[model_name]['image_size'] = 1280    
candidate_models[model_name]['confidence_thresholds'] = {'default':0.5,'tegu':0.475}
candidate_models[model_name]['model_type'] = 'yolov5'

model_name = 'all-classes_usgs-goannas-lilablanks_yolov5x6-20240317-training-complete'
candidate_models[model_name] = {}
candidate_models[model_name]['model_file'] = \
    os.path.join(model_base_folder,
                 'usgs-tegus-yolov5-lilablanks_goannas-20240205105940-b8-img1280-e3003/checkpoint-training-complete-20240317/usgs-tegus-yolov5-lilablanks_goannas-20240205105940-b8-img1280-e3003-best-cp-training-complete-20240317-stripped.pt')
candidate_models[model_name]['image_size'] = 1280    
candidate_models[model_name]['confidence_thresholds'] = {'default':0.5,'tegu':0.45}
candidate_models[model_name]['model_type'] = 'yolov5'

model_filenames = set()


#%% Fill in default information for each model

def fill_defaults(model_name):
    
    if model_name == 'default':
        return

    model_info = candidate_models[model_name]
    
    if 'results_file' not in model_info:
        results_file = os.path.join(results_base_folder,model_name + '.json')
        model_info['results_file'] = results_file
    
    model_type = model_info['model_type']
    
    if model_type is not None and model_type != 'synthetic':
        
        model_filename = model_info['model_file']
        assert '\\' not in model_filename
        assert 'last' not in model_filename
        assert model_filename not in model_filenames
        model_filenames.add(model_filename)
        
        assert os.path.isfile(model_info['model_file'])
        assert model_type in model_info['model_file']
        
    if 'confidence_thresholds' not in model_info:
        model_info['confidence_thresholds'] = \
            candidate_models['default']['confidence_thresholds']
    if 'rendering_confidence_thresholds' not in model_info:
        model_info['rendering_confidence_thresholds'] = \
            candidate_models['default']['rendering_confidence_thresholds']

    if 'dataset_file' not in model_info:
        dataset_file_name = os.path.join(os.path.dirname(model_filename),'dataset.yaml')
        model_info['dataset_file'] = dataset_file_name
        
    assert os.path.isfile(model_info['dataset_file'])

    
for model_name in candidate_models.keys():    
    fill_defaults(model_name)
    
model_names = [s for s in candidate_models.keys() if s != 'default']


#%% Validate model class names against dataset files

# model_name = model_names[-1]; print(model_name)
for model_name in model_names:
    
    model_info = candidate_models[model_name]
    
    model_type = model_info['model_type']
    assert model_type in ('yolov5','yolov8')
    
    model_file = model_info['model_file']
    assert os.path.isfile(model_file)
    
    model = YOLO(model_file)
    model_class_id_to_name = model.names
    
    image_size = model_info['image_size']
    _ = int(image_size)
    
    dataset_file_name = model_info['dataset_file']
    dataset_file_class_id_to_name = \
        yolo_output_to_md_output.read_classes_from_yolo_dataset_file(dataset_file_name)
    assert len(model_class_id_to_name) == len(dataset_file_class_id_to_name)
    
    for class_id in model_class_id_to_name:
        assert model_class_id_to_name[class_id] == dataset_file_class_id_to_name[class_id]
    
    yolo_dataset_file = model_info['dataset_file']
    assert os.path.isfile(yolo_dataset_file) and yolo_dataset_file.endswith('.yaml')


#%% YOLO --> COCO conversion (if necessary)

force_yolo_to_coco_conversion = False

val_file_coco_from_yolo = os.path.join(data_folder,'dataset-val-converted-from-yolo.json')    

if force_yolo_to_coco_conversion:
    
    from data_management.yolo_to_coco import yolo_to_coco
    
    _ = yolo_to_coco(input_folder = val_folder,
                     class_name_file = os.path.join(data_folder,'dataset.yaml'),
                     output_file = val_file_coco_from_yolo,
                     empty_image_handling = 'empty_annotations')
    
else:
    
    pass
    # assert os.path.isfile(val_file_coco_from_yolo)
    
    
#%% Create val-only json file

force_val_only_json_file_creation = False

ground_truth_file = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training/usgs-kissel-training.json')
val_file_coco = ground_truth_file.replace('.json','-val-only.json')

if force_val_only_json_file_creation:
    
    assert os.path.isfile(ground_truth_file)
    
    with open(ground_truth_file,'r') as f:
        d = json.load(f)
        
    images_to_keep = []
    
    # im = d['images'][0]
    for im in d['images']:
        if 'train/' not in im['file_name']:
            images_to_keep.append(im)
        if 'width' in im:
            del im['width']
        if 'height' in im:
            del im['height']
            
    annotations_to_keep = []
    
    # ann = d['annotations'][0]
    for ann in d['annotations']:
        if 'train/' not in ann['image_id']:
            annotations_to_keep.append(ann)
    
    print('Kept {} of {} images'.format(len(images_to_keep),len(d['images'])))
    print('Kept {} of {} annotations'.format(len(annotations_to_keep),len(d['annotations'])))
    
    d['images'] = images_to_keep
    d['annotations'] = annotations_to_keep
    
    with open(val_file_coco,'w') as f:
        json.dump(d,f,indent=1)

else:
    
    assert os.path.isfile(val_file_coco)
    
    
#%% Validate ground truth data

with open(val_file_coco,'r') as f:
    d = json.load(f)

from collections import defaultdict
image_id_to_annotations = defaultdict(list)

for ann in d['annotations']:
    image_id_to_annotations[ann['image_id']].append(ann)
    
category_id_to_name = {c['id']:c['name'] for c in d['categories']}  
category_name_to_id = {c['name']:c['id'] for c in d['categories']}  
empty_category_id = category_name_to_id['empty']

for im in tqdm(d['images']):
    
    assert im['id'] in image_id_to_annotations
    fn_relative = im['file_name']
    fn_abs = os.path.join(data_folder,fn_relative)
    assert os.path.isfile(fn_abs)
    
    annotations_this_image = image_id_to_annotations[im['id']]    
    if 'blanks' in fn_relative:
        assert len(annotations_this_image) == 1
        assert annotations_this_image[0]['category_id'] == empty_category_id
    else:
        for ann in annotations_this_image:
            assert ann['category_id'] != empty_category_id


#%% Run each model on the validation data

# mamba activate yolov5
# export PYTHONPATH=/mnt/c/git/MegaDetector
# export PYTHONPATH=/home/user/git/MegaDetector

yolo_working_folder = os.path.expanduser('~/git/yolov5-training')

gpu_to_commands = defaultdict(list)

# model_name = model_names[0]
for i_model,model_name in enumerate(model_names):
    
    model_info = candidate_models[model_name]
    
    model_type = model_info['model_type']
    assert model_type in ('yolov5','yolov8')
    
    model_file = model_info['model_file']
    assert os.path.isfile(model_file)
    
    image_size = model_info['image_size']
    _ = int(image_size)
    
    yolo_dataset_file = model_info['dataset_file']
    assert os.path.isfile(yolo_dataset_file) and yolo_dataset_file.endswith('.yaml')
        
    results_file = model_info['results_file']
    
    cmd = 'python run_inference_with_yolov5_val.py "{}" "{}" "{}" --model_type {}'.format(
        model_file,
        val_folder,
        results_file,
        model_type)
    
    if model_type == 'yolov5':
        cmd += ' --yolo_working_folder {}'.format(yolo_working_folder)
        
    cmd += ' --overwrite_handling overwrite'
    cmd += ' --yolo_dataset_file {}'.format(yolo_dataset_file)
    cmd += ' --image_size {}'.format(image_size)
    
    if not augment:
        cmd += ' --augment_enabled 0'

    if n_gpus > 1:
        gpu_index = i_model % n_gpus
        cmd = 'CUDA_VISIBLE_DEVICES={} '.format(gpu_index) + cmd
    else:
        gpu_index = 0
        
    gpu_to_commands[gpu_index].append(cmd)    

# ...for each model

output_script_base = os.path.join(results_base_folder,'run_all_models_on_val_data.sh')

if n_gpus > 1:
    from md_utils.path_utils import insert_before_extension
    for gpu_index in range(0,n_gpus):
        cmd_file = insert_before_extension(output_script_base,'gpu_' + str(gpu_index).zfill(2))
        with open(cmd_file,'w') as f:
            for c in gpu_to_commands[gpu_index]:
                f.write(c + '\n')
        st = os.stat(cmd_file)
        os.chmod(cmd_file, st.st_mode | stat.S_IEXEC)
else:
    output_script = output_script_base
    assert len(gpu_to_commands) == 1
    with open(output_script,'w') as f:
        for c in gpu_to_commands[0]:
            f.write(c + '\n')        
    st = os.stat(output_script)
    os.chmod(output_script, st.st_mode | stat.S_IEXEC)

# import clipboard; cmd = commands[1]; print(cmd); clipboard.copy(cmd)


#%% Run the script(s)

# ...


#%% Confirm that all the output files got written

# ...and that they all contain results for the same files.

images_in_results = None

# model_name = model_names[0]
for model_name in model_names:
    
    model_info = candidate_models[model_name]
    assert os.path.isfile(model_info['results_file'])

    with open(model_info['results_file'],'r') as f:
        model_results = json.load(f)
        
    images_set = set([im['file'] for im in model_results['images']])
    
    if images_in_results is None:
        images_in_results = images_set
    else:
        assert images_in_results == images_set
    
    
#%% Remove "val/" from the ground truth file

val_file_coco_no_val_folder = val_file_coco.replace('.json','_no_val_folder.json')

force_val_folder_removal = False

if force_val_folder_removal:
    
    with open(val_file_coco,'r') as f:
        d = json.load(f)
    
    for im in d['images']:
        assert im['file_name'].startswith('val/')
        im['file_name'] = im['file_name'].replace('val/','',1)
        assert 'val/' not in im['file_name']
    
    with open(val_file_coco_no_val_folder,'w') as f:
        json.dump(d,f,indent=1)

else:

    assert os.path.isfile(val_file_coco_no_val_folder)    


#%% Merge the best results

from api.batch_processing.postprocessing.merge_detections import \
    MergeDetectionsOptions, merge_detections
    
merge_options = MergeDetectionsOptions()

merge_options.source_confidence_thresholds = [0.05]

# We'd rather have redundant detections in this case
merge_options.target_confidence_threshold = 0.9

merge_options.categories_to_include = None
merge_options.merge_empty_only = False
merge_options.iou_threshold = 0.65
merge_options.overwrite = True

target_model_name_for_merging = 'all-classes_usgs-only_yolov5x6'
source_model_name_for_merging = 'all-classes_usgs-goannas-lilablanks_yolov5x6-20240223'    

target_results_file = candidate_models[target_model_name_for_merging]['results_file']
source_results_file = candidate_models[source_model_name_for_merging]['results_file']

with open(source_results_file,'r') as f:
    source_d = json.load(f)

with open(target_results_file,'r') as f:
    target_d = json.load(f)

remapping_required = \
    (source_d['detection_categories'] != target_d['detection_categories'])

if remapping_required:
    
    # If the two best results sets use slightly different categories, remap
    remapped_source_file = source_results_file.replace('.json','_remapped.json')
    assert remapped_source_file != source_results_file
    
    from api.batch_processing.postprocessing.remap_detection_categories import \
        remap_detection_categories
    
    remap_detection_categories(input_file=source_results_file,
                               output_file=remapped_source_file,
                               target_category_map=target_results_file,
                               extra_category_handling='drop_if_unused',
                               overwrite=False)

    source_results_file = remapped_source_file
    with open(source_results_file,'r') as f:
        source_d = json.load(f)
    assert source_d['detection_categories'] == target_d['detection_categories']
        
merged_results_file = os.path.join(results_base_folder,'merged_detections_00.json')

merge_detections(source_files=[source_results_file],
                 target_file=target_results_file,
                 output_file=merged_results_file,
                 options=merge_options)


#%% Create a new entry in the model list for this case

model_name = 'all-classes_merged_results'
target_model_dataset_yaml_file = candidate_models[target_model_name_for_merging]['dataset_file']

candidate_models[model_name] = {}
candidate_models[model_name]['model_file'] = 'merged results from multiple files'
candidate_models[model_name]['results_file'] = merged_results_file
candidate_models[model_name]['image_size'] = 1280    
candidate_models[model_name]['confidence_thresholds'] = {'default':0.5,'tegu':0.6}
candidate_models[model_name]['model_type'] = 'synthetic'
candidate_models[model_name]['dataset_file'] = target_model_dataset_yaml_file
fill_defaults(model_name)

model_names = [s for s in candidate_models.keys() if s != 'default']


#%% Render confusion matrices for each model

from api.batch_processing.postprocessing.render_detection_confusion_matrix \
    import render_detection_confusion_matrix
    
html_image_list_options = {'maxFiguresPerHtmlFile':3000}
target_image_size = (1280,-1)
force_render_images = True

# i_model = -1; model_name = model_names[i_model]
#
# model_name = 'all-classes_usgs-goannas-lilablanks_yolov5x6-20240317-training-complete'
# i_model = model_names.index(model_name)
for i_model,model_name in enumerate(model_names):

    print('Processing results from model {} of {}'.format(
        i_model,len(model_names)))
    
    model_info = candidate_models[model_name]

    preview_folder = os.path.join(preview_base_folder,model_name)
    confusion_matrix_results = render_detection_confusion_matrix(
        ground_truth_file=val_file_coco_no_val_folder,
        results_file=model_info['results_file'],
        image_folder=val_folder,
        preview_folder=preview_folder,
        force_render_images=force_render_images, 
        confidence_thresholds=model_info['confidence_thresholds'],
        rendering_confidence_thresholds=model_info['rendering_confidence_thresholds'],
        target_image_size=target_image_size,
        parallelize_rendering=True,
        parallelize_rendering_n_cores=10,
        parallelize_rendering_with_threads=True,
        job_name=model_name,
        model_file=model_info['model_file'],
        empty_category_name='empty',
        html_image_list_options=html_image_list_options)

    model_info['confusion_matrix_results'] = confusion_matrix_results
    
# ...for each model


#%% Open results

# model_name = model_names[-1]
for model_name in model_names:

    model_info = candidate_models[model_name]
    cm_info = model_info['confusion_matrix_results']
    html_file = cm_info['html_file']
    open_file(html_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')


#%% Zip model files

from md_utils.path_utils import parallel_zip_files

model_files = [m['model_file'] for m in candidate_models.values() if m['model_file'] is not None]
model_files = [fn for fn in model_files if os.path.isfile(fn)]
    
parallel_zip_files(model_files, use_threads=False)
