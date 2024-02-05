########
#
# add-lila-blanks.py
#
# Add blank images downloaded from LILA (via create_lila_blank_set.py) to the usgs-tegus YOLO
# folder, including splitting blank locations into train/val.
#
# Blanks will be put in full folder trees within train/lila-blanks and val/lila-blanks.
#
########

#%% Imports and constants

import os
import json
import random
import shutil

from collections import defaultdict
from tqdm import tqdm

from md_utils.path_utils import find_images

output_folder_base = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-yolo')


#%% Load the list of blank images downloaded from LILA

# Enumerate blank images
lila_blank_base = os.path.expanduser('~/lila/lila_blanks')
lila_blank_image_folder = os.path.join(lila_blank_base,'confirmed_blanks')
blank_images = find_images(lila_blank_image_folder,recursive=True,return_relative_paths=True)

print('Found {} blank images in {}'.format(len(blank_images),lila_blank_base))

# Load the mapping from filenames to locations, and invert to get a mapping from locations to filenames
fn_relative_to_location_file = os.path.join(lila_blank_base,'confirmed_fn_relative_to_location.json')

with open(fn_relative_to_location_file,'r') as f:
    fn_relative_to_location = json.load(f)

assert len(blank_images) == len(fn_relative_to_location)
    
location_to_relative_image_filenames = defaultdict(list)

for fn_relative in tqdm(fn_relative_to_location.keys()):
    location = fn_relative_to_location[fn_relative]
    location_to_relative_image_filenames[location].append(fn_relative)
    

#%% Split blank images locations into train/val

random.seed(0)
all_locations = list(location_to_relative_image_filenames)
val_fraction = 0.15
n_val_locations = round(val_fraction * len(all_locations))
n_train_locations = len(all_locations) - n_val_locations
val_locations = random.sample(all_locations,n_val_locations)
train_locations = []
for location in tqdm(all_locations):
    if location not in val_locations:
        train_locations.append(location)
assert len(train_locations) == n_train_locations
print('\nSplit locations into {} train and {} val'.format(
    n_train_locations,n_val_locations))


#%% Copy blank images into the training folder

split_names = ('train','val')
split_to_locations = {'train':train_locations,'val':val_locations}

# split_name = split_names[0]
for split_name in split_names:
    
    split_base = os.path.join(output_folder_base,split_name)
    assert os.path.isdir(split_base)
    split_lila_blank_output_base = os.path.join(split_base,'lila-blanks')
    os.makedirs(split_lila_blank_output_base,exist_ok=True)
    
    split_locations = split_to_locations[split_name]
    
    n_locations_this_split = len(split_locations)
    n_images_this_split = 0
    
    # location = split_locations[0]
    for location in tqdm(split_locations):
        
        relative_image_filenames = location_to_relative_image_filenames[location]
        
        # fn_relative = relative_image_filenames[0]
        for fn_relative in relative_image_filenames:
            
            source_fn_abs = os.path.join(lila_blank_image_folder,fn_relative)
            assert os.path.isfile(source_fn_abs)
            target_fn_abs = os.path.join(split_lila_blank_output_base,fn_relative)
            os.makedirs(os.path.dirname(target_fn_abs),exist_ok=True)
            shutil.copyfile(source_fn_abs,target_fn_abs)
            n_images_this_split += 1
            
    print('\nCopied {} files from {} locations for split {}'.format(
        n_images_this_split,n_locations_this_split,split_name))


#%% Summarize folder content

images = find_images(output_folder_base,recursive=True)
print('Found {} images'.format(len(images)))

lila_blanks = [fn for fn in images if 'lila-blank' in fn]
print('Found {} LILA-blank images'.format(len(lila_blanks)))

for split_name in ('train','val'):
    split_folder = os.path.join(output_folder_base,split_name)
    assert os.path.isdir(split_folder)
    split_images = find_images(split_folder,recursive=True)
    print('Found {} images for split {}'.format(len(split_images),split_name))


#%% Resize blank images in place

# It would have been faster to do this during the copying step, but this on a single thread, 
# this is a *lot* slower than copying, and it was useful to do some consistency-checking quickly
# right after the copying step, so, this is a compromise: copy first, then resize in parallel.

from md_visualization.visualization_utils import resize_image
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool

def resize_training_image(fn_abs):
    # cmd = 'file "{}"'.format(fn_abs); clipboard.copy(cmd)    
    _ = resize_image(fn_abs, target_width=1600, target_height=-1, output_file=fn_abs, 
                     no_enlarge_width=True, verbose=True, quality=85)
    return None

pool_type = 'process'
n_workers = 16

if n_workers == 1:    
    
    # fn_abs = lila_blanks[0]
    for fn_abs in tqdm(lila_blanks):
        resize_training_image(fn_abs)

else:
    
    if pool_type == 'thread':
        pool = ThreadPool(n_workers); poolstring = 'threads'                
    else:
        assert pool_type == 'process'
        pool = Pool(n_workers); poolstring = 'processes'
    
    print('Starting resizing pool with {} {}'.format(n_workers,poolstring))
    
    _ = list(tqdm(pool.imap(resize_training_image, lila_blanks)))


#%% Scrap

if False:

    #%% Experimenting with image resizing
    
    fn_in = '/home/user/lila/lila_blanks/confirmed_blanks/idaho-camera-traps/public/loc_0003/loc_0003_im_000359.jpg'
    fn_out = os.path.expanduser('~/tmp/loc_0003_im_000359-resized.jpg')
    _ = resize_image(fn_in, target_width=1600, target_height=-1, output_file=fn_out, no_enlarge_width=True, verbose=True, quality=85)
