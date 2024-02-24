########
#
# prepare-mini-traning-set.py
#
# Prepare a reduced-size, reduced-class-set version of the training data set, in
# this case for embedded YOLOv5s training.
#
########

#%% Imports and constants

import os 
from md_visualization import visualization_utils as vis_utils
from md_utils.path_utils import find_images

data_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-yolo-1600')
all_images_relative = find_images(data_folder,recursive=True, return_relative_paths=True)

usgs_images = [fn for fn in all_images_relative if (('unsw' not in fn) and ('lila-blank' not in fn))]


#%% Resize images

target_width = 600

input_folder = data_folder
output_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-yolo-{}-tegu_human_w_goanna_lila'.format(
    target_width))

output_yolo_dataset_file = os.path.join(output_folder,'dataset.yaml')
input_yolo_dataset_file = os.path.join(input_folder,'dataset.yaml')

_ = vis_utils.resize_image_folder(input_folder,output_folder,
                    target_width=target_width,verbose=False,quality=85,no_enlarge_width=True,
                    pool_type='process',n_workers=12,image_files_relative=all_images_relative)


#%% Write dataset file

from data_management import coco_to_yolo
class_list = ['tegu','human']

coco_to_yolo.write_yolo_dataset_file(output_yolo_dataset_file,
                                     dataset_base_dir=output_folder,
                                     class_list=class_list,
                                     train_folder_relative='train',
                                     val_folder_relative='val',
                                     test_folder_relative=None)

class_list_files = [
    os.path.join(output_folder,'val','object.data'),
    os.path.join(output_folder,'train','object.data')
]

for fn in class_list_files:
    with open(fn,'w') as f:
        for class_name in class_list:
            f.write(class_name + '\n')


#%% Prepare yolo annotation files

from data_management.yolo_output_to_md_output import read_classes_from_yolo_dataset_file
from md_utils.ct_utils import invert_dictionary
from tqdm import tqdm

# Read input and output dataset files

input_yolo_category_id_to_name = read_classes_from_yolo_dataset_file(input_yolo_dataset_file)
assert input_yolo_category_id_to_name[14] == 'tegu'
assert input_yolo_category_id_to_name[6] == 'human'

output_yolo_category_id_to_name = read_classes_from_yolo_dataset_file(output_yolo_dataset_file)
assert output_yolo_category_id_to_name[0] == 'tegu'
assert output_yolo_category_id_to_name[1] == 'human'

# Map input categories to output categories
input_category_name_to_id = invert_dictionary(input_yolo_category_id_to_name)
output_category_name_to_id = invert_dictionary(output_yolo_category_id_to_name)

# For every image

from collections import defaultdict
category_name_to_files = defaultdict(set)
missing_output_files = []

# fn_image_relative = usgs_images[0]
for fn_image_relative in tqdm(all_images_relative):
    
    fn_text_relative = os.path.splitext(fn_image_relative)[0] + '.txt'
    fn_input_image_abs = os.path.join(input_folder,fn_image_relative)
    fn_output_image_abs = os.path.join(output_folder,fn_image_relative)
    
    assert os.path.isfile(fn_input_image_abs)
    if not os.path.isfile(fn_output_image_abs):
        assert fn_image_relative not in usgs_images
        missing_output_files.append(fn_image_relative)
        continue
    
    fn_input_text_abs = os.path.join(input_folder,fn_text_relative)
    fn_output_text_abs = os.path.join(output_folder,fn_text_relative)
    
    if not os.path.isfile(fn_input_text_abs):
        assert ('blanks_and_very_small_things#' in fn_image_relative) or \
               ('lila-blanks' in fn_image_relative)
        continue
    
    # Read the txt file
    with open(fn_input_text_abs,'r') as f:
        input_lines = f.readlines()
    input_lines = [s.strip() for s in input_lines]
    
    # For each detection
    output_annotation_lines = []
    
    # annotation_line = input_lines[0]
    for annotation_line in input_lines:
        
        tokens = annotation_line.split()
        assert len(tokens) == 5
        input_category_id = int(tokens[0])
        input_category_name = input_yolo_category_id_to_name[input_category_id]
        
        if input_category_name not in output_category_name_to_id:
            continue
        
        category_name_to_files[input_category_name].add(fn_image_relative)
        output_category_id = output_category_name_to_id[input_category_name]
        output_annotation_line = str(output_category_id) + ' ' + ' '.join(tokens[1:])
        output_annotation_lines.append(output_annotation_line)
        
    if len(output_annotation_lines) == 0:
        continue
            
    # Write the output text file if necessary
    with open(fn_output_text_abs,'w') as f:
        for s in output_annotation_lines:
            f.write(s + '\n')
    
# ...for every image

# Only 10 in practice    
print('{} missing output images'.format(len(missing_output_files)))

for category_name in category_name_to_files.keys():
    print('{}: {} images'.format(category_name,len(category_name_to_files[category_name])))


#%% Train

training_image_size = str(448)
epochs = 300
batch_size = 128
dataset_file = output_yolo_dataset_file
device_string = '0,1'
patience = 50
cache = False
base_weights = 'yolov5s.pt'
project = os.path.expanduser('~/tmp/usgs-tegus/yolov5-mini-training')
training_run_name = 'usgs-tegus-tegu_human_w_goanna_lilablanks-im{}-e{}-b{}-{}'.format(
    training_image_size,epochs,batch_size,base_weights.split('.')[0])

use_ddp = True
if use_ddp:
    base_train_command = 'python -m torch.distributed.run --nproc_per_node 2 train.py'
else:
    base_train_command = 'python train.py'    

train_cmd = \
    '{} --img {} --epochs {} --data "{}" --batch {} --weights {} --patience {}'.format(
        base_train_command,training_image_size,epochs,dataset_file,batch_size,base_weights,patience)
    
train_cmd += ' --project "{}" --name "{}"'.format(project,training_run_name)

if cache:
    train_cmd += ' --cache'
    
print(train_cmd)
# import clipboard; clipboard.copy(train_cmd)