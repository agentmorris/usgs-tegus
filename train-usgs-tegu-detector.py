########
#
# train-usgs-tegu-detector.py
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

#%% Constants and imports

import os
import shutil
import datetime
import clipboard

training_base_folder = os.path.expanduser('~/tmp/usgs-tegus/yolov5-training')
os.makedirs(training_base_folder,exist_ok=True)

yolo_dataset_file = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-yolo/dataset.yaml')
assert os.path.isfile(yolo_dataset_file)


#%% Import yolov5 tools for stripping optimizer state if we stop training early

import sys

# Remove other YOLOv5 folders
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
    
utils_imported = False
if not utils_imported:
    try:
        from yolov5.utils.general import strip_optimizer
        utils_imported = True
    except Exception:
        pass
if not utils_imported:
    try:
        from ultralytics.utils.general import strip_optimizer # noqa
        utils_imported = True
    except Exception:
        pass        
if not utils_imported:
    try:
        from utils.general import strip_optimizer # noqa
        utils_imported = True
    except Exception:
        pass
assert utils_imported


#%% Environment prep

"""
mamba create --name yolov5 python=3.11 pip -y
mamba activate yolov5
cd ~/git
git clone https://github.com/agentmorris/yolov5-training
cd yolov5-training
pip install -r requirements.txt

mamba install -c conda-forge spyder
pip install clipboard

# Windows typically doesn't get PyTorch-GPU from the default YOLOv5 install, force a re-install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""


#%% Train (YOLOv5)

"""
cd ~/git/yolov5-training

mamba activate yolov5

# I usually have an older commit of yolov5 on my PYTHONPATH, remove it.

# Linux
export PYTHONPATH=
LD_LIBRARY_PATH=

# Windows
set PYTHONPATH=
"""

batch_size = 8
image_size = 1280
epochs = 300
device_string = '0,1'
use_ddp = True
patience = 50 # defaults to 100

dt = datetime.datetime.now()
dt_string = '{}{}{}{}{}{}'.format(dt.year,str(dt.month).zfill(2),str(dt.day).zfill(2),
  str(dt.hour).zfill(2),str(dt.minute).zfill(2),str(dt.second).zfill(2))

run_index_string = ''

# Core data + LILA blanks (Linux DDP)
# dt_string = '20240205101724'; exp_name = 'lilablanks'; run_index_string = '6'

# Core data + LILA blanks + goannas (WSL DDP)
dt_string = '20240205105940'; exp_name = 'lilablanks_goannas'; run_index_string = '3'

assert len(dt_string) == 14

training_run_name = 'usgs-tegus-yolov5-{}-{}-b{}-img{}-e{}{}'.format(
    exp_name,dt_string,batch_size,image_size,epochs,run_index_string)

base_model = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
assert os.path.isfile(base_model)

project_dir = training_base_folder

run_dir = os.path.join(project_dir,training_run_name)
if os.path.exists(run_dir):
    print('\n*** Warning: folder {} exists. ***\n\nIf you are resuming, this is fine.\n'.format(
        run_dir))

# See https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training/
if use_ddp:
    base_train_command = 'python -m torch.distributed.run --nproc_per_node 2 train.py'
else:
    base_train_command = 'python train.py'    


train_cmd = f'{base_train_command} --img {image_size} --batch {batch_size} --epochs {epochs} --weights "{base_model}" --device {device_string} --project "{project_dir}" --name "{training_run_name}" --data "{yolo_dataset_file}" --patience "{patience}"'
print('Training command:\n')
print(train_cmd)
# clipboard.copy(train_cmd)

# Resume command
resume_checkpoint = os.path.join(project_dir,training_run_name,'weights/last.pt')

# This file doesn't exist when we start training the first time
# os.path.isfile(resume_checkpoint)

resume_cmd = f'{base_train_command} --resume {resume_checkpoint}'

print('\nResume command:\n')
print(resume_cmd)
# clipboard.copy(resume_cmd)


#%% Train (YOLOv8)

"""
cd ~/git

# I usually have an older commit of yolov5 on my PYTHONPATH, remove it.
export PYTHONPATH=
LD_LIBRARY_PATH=

# Not a typo; the ultralytics package is installed in the YOLOv5 environment
mamba activate yolov5
"""

batch_size = -1
image_size = 640
epochs = 200

import datetime
dt = datetime.datetime.now()

dt_string = '{}{}{}{}{}{}'.format(dt.year,str(dt.month).zfill(2),str(dt.day).zfill(2),
  str(dt.hour).zfill(2),str(dt.minute).zfill(2),str(dt.second).zfill(2))
# dt_string = '20231205053411'
assert len(dt_string) == 14

training_run_name = 'usgs-tegus-yolov8-{}-b{}-img{}-e{}'.format(
    dt_string,batch_size,image_size,epochs)

base_model = 'yolov8x.pt'

if base_model != 'yolov8x.pt':
    assert os.path.isfile(base_model)

project_dir = training_base_folder

cmd = f'mamba activate yolov5 && yolo detect train data="{yolo_dataset_file}" model="{base_model}" epochs={epochs} imgsz={image_size} project="{project_dir}" name="{training_run_name}" device=0,1'

print('Training command:\n\n{}'.format(cmd))
# clipboard.copy(cmd)

# This file doesn't exist yet
resume_checkpoint = os.path.join(project_dir,training_run_name,'weights/last.pt')

resume_command = f'mamba activate yolov5 && yolo detect train resume data="{yolo_dataset_file}" model="{resume_checkpoint}" epochs={epochs} imgsz={image_size} project="{project_dir}" name="{training_run_name}" device=0,1'

print('\nResume command:\n\n{}'.format(resume_command))
# print('BACK UP WEIGHTS FIRST'); assert os.path.isfile(resume_checkpoint); clipboard.copy(resume_command)


#%% Back up models after (or during) training, removing optimizer state if appropriate

# Input folder(s)
training_output_dir = os.path.join(project_dir,training_run_name)
training_weights_dir = os.path.join(training_output_dir,'weights')
assert os.path.isdir(training_weights_dir)

# Output folder
model_folder = os.path.expanduser('~/models/usgs-tegus/{}'.format(training_run_name))
checkpoint_tag = 'unknown'
assert checkpoint_tag != 'unknown' # 20240000
model_folder = os.path.join(model_folder,'checkpoint-' + checkpoint_tag)
os.makedirs(model_folder,exist_ok=True)

# weight_name = 'last'
for weight_name in ('last','best'):
    source_file = os.path.join(training_weights_dir,weight_name + '.pt')
    assert os.path.isfile(source_file)
    target_file = os.path.join(model_folder,'{}-{}-cp-{}.pt'.format(
        training_run_name,weight_name,checkpoint_tag))
    
    shutil.copyfile(source_file,target_file)
    target_file_optimizer_stripped = target_file.replace('.pt','-stripped.pt')
    strip_optimizer(target_file,target_file_optimizer_stripped)
    
# Copy dataset.yaml    
target_dataset_file = os.path.join(model_folder,os.path.basename(yolo_dataset_file))
shutil.copyfile(yolo_dataset_file,target_dataset_file)

other_files = os.listdir(training_output_dir)
other_files = [os.path.join(training_output_dir,fn) for fn in other_files]
other_files = [fn for fn in other_files if os.path.isfile(fn)]

# source_file_abs = other_files[0]
for source_file_abs in other_files:
    assert not source_file_abs.endswith('.pt')
    fn_relative = os.path.basename(source_file_abs)
    target_file_abs = os.path.join(model_folder,fn_relative)
    shutil.copyfile(source_file_abs,target_file_abs)


#%% Make plots during training

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure

from md_utils.path_utils import open_file

assert 'yolov5' in training_run_name or 'yolov8' in training_run_name
if 'yolov5' in training_run_name:
    model_type = 'yolov5'
else:
    model_type = 'yolov8'

results_page_folder = os.path.join(training_base_folder,'training-progress-report')
os.makedirs(results_page_folder,exist_ok=True)
fig_00_fn_abs = os.path.join(results_page_folder,'figure_00.png')
fig_01_fn_abs = os.path.join(results_page_folder,'figure_01.png')
fig_02_fn_abs = os.path.join(results_page_folder,'figure_02.png')
    
results_file = os.path.join(project_dir,training_run_name,'results.csv')
assert os.path.isfile(results_file)
clipboard.copy(results_file)

df = pd.read_csv(results_file)
df = df.rename(columns=lambda x: x.strip())

plt.ioff()

fig_w = 12
fig_h = 8

fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), tight_layout=True)
ax = fig.subplots(1, 1)

if model_type == 'yolov5':
    df.plot(x = 'epoch', y = 'val/box_loss', ax = ax) 
    df.plot(x = 'epoch', y = 'val/obj_loss', ax = ax, secondary_y = True) 
    df.plot(x = 'epoch', y = 'train/box_loss', ax = ax) 
    df.plot(x = 'epoch', y = 'train/obj_loss', ax = ax, secondary_y = True) 
else:
    df.plot(x = 'epoch', y = 'val/box_loss', ax = ax) 
    df.plot(x = 'epoch', y = 'val/dfl_loss', ax = ax, secondary_y = True) 
    df.plot(x = 'epoch', y = 'train/box_loss', ax = ax) 
    df.plot(x = 'epoch', y = 'train/dfl_loss', ax = ax, secondary_y = True) 

fig.savefig(fig_00_fn_abs,dpi=100)
plt.close(fig)

fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), tight_layout=True)
ax = fig.subplots(1, 1)

df.plot(x = 'epoch', y = 'val/cls_loss', ax = ax) 
df.plot(x = 'epoch', y = 'train/cls_loss', ax = ax) 

fig.savefig(fig_01_fn_abs,dpi=100)
plt.close(fig)

fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), tight_layout=True)
ax = fig.subplots(1, 1)

if model_type == 'yolov5':
    df.plot(x = 'epoch', y = 'metrics/precision', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/recall', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/mAP_0.5', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/mAP_0.5:0.95', ax = ax)
else:
    df.plot(x = 'epoch', y = 'metrics/precision(B)', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/recall(B)', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/mAP50(B)', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/mAP50-95(B)', ax = ax)

fig.savefig(fig_02_fn_abs,dpi=100)
plt.close(fig)

results_page_html_file = os.path.join(results_page_folder,'index.html')
with open(results_page_html_file,'w') as f:
    f.write('<html><body>\n')
    f.write('<img src="figure_00.png"><br/>\n')
    f.write('<img src="figure_01.png"><br/>\n')
    f.write('<img src="figure_02.png"><br/>\n')    
    f.write('</body></html>\n')

open_file(results_page_html_file,browser_name='chrome')
