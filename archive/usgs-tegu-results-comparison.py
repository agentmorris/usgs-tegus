########
#
# usgs-tegu-results-comparison.py
#
# Scrap code for comparing a few experimental outputs.
#
########

#%% Load results files

import json

validation_old_output_file = '/home/user/tmp/usgs-tegus-val-analysis/md_val_results.json'
validation_new_output_file = '/home/user/tmp/usgs-tegus/model-comparison/all-classes_usgs-only_yolov5x6.json'

with open(validation_old_output_file,'r') as f:
    d_old = json.load(f)
    
with open(validation_new_output_file,'r') as f:
    d_new = json.load(f)

tegu_category_id = '14'
other_bird_category_id = '8'


#%% Make sure both files have the same images

images_old = [im['file'] for im in d_old['images']]
images_new = [im['file'] for im in d_new['images']]

assert set(images_new) == set(images_old)


#%% Compare specific files

fn_to_results_old = {im['file']:im for im in d_old['images']}
fn_to_results_new = {im['file']:im for im in d_new['images']}

test_fn = 'tegu/tegu#TuMe#111.18_C93#2016#C93_111.18#(43) 29 MAR 17 - 15 APR 17 CMG FTC#MFDC2018.JPG'

im_results_old = fn_to_results_old[test_fn]
im_results_new = fn_to_results_new[test_fn]

c_id = other_bird_category_id
detections_old = [d for d in im_results_old['detections'] if d['category'] == c_id]
detections_new = [d for d in im_results_new['detections'] if d['category'] == c_id]
