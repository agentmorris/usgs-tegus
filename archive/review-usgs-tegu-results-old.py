########
#
# review-usgs-tegu-results.py
#
# Create data review pages for USGS tegu validation data
#
########

#%% Imports and constants

import os
import json

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from md_utils.path_utils import find_images
from md_utils.path_utils import flatten_path
from md_utils.path_utils import open_file    
from md_visualization import visualization_utils as vis_utils
from md_utils.write_html_image_list import write_html_image_list

import md_visualization.plot_utils as plot_utils

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool

# YOLOv5 model
if True:    
    model_file = os.path.expanduser('~/models/usgs-tegus/usgs-tegus-yolov5x-231003-b8-img1280-e3002/weights/usgs-tegus-yolov5x-231003-b8-img1280-e3002-best-stripped.pt')
    model_type = 'yolov5'
    scratch_folder = os.path.expanduser('~/tmp/usgs-tegus-val-analysis')
    confidence_thresholds = {'default':0.5,'tegu':0.45}
    rendering_confidence_thresholds = {'default':0.3,'tegu':0.08}
    job_name = 'USGS tegu val'

# YOLOv8 model:
if False:
    model_file = os.path.expanduser('~/models/usgs-tegus/usgs-tegus-yolov8x-2023.10.26-b-1-img640-e300/weights/usgs-tegus-yolov8x-2023.10.26-b-1-img640-e300-best.pt')
    model_type = 'yolov8'
    scratch_folder = os.path.expanduser('~/tmp/usgs-tegus-val-analysis-v8-300')
    confidence_thresholds = {'default':0.5,'tegu':0.45}
    rendering_confidence_thresholds = {'default':0.3,'tegu':0.08}
    job_name = 'USGS tegu val (yolov8-300)'

assert os.path.isfile(model_file)
augment = False
if augment:
    job_name += ' (aug)'

if augment:
    scratch_folder += '-aug'
os.makedirs(scratch_folder,exist_ok=True)

if model_type == 'yolov5':
    yolo_working_folder = os.path.expanduser('~/git/yolov5-training')
else:
    yolo_working_folder = None
    
training_data_folder = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training')
training_data_folder_resized = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-resized')

training_metadata_file = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training.json')
training_metadata_file_resized = os.path.expanduser('~/data/usgs-tegus/usgs-kissel-training-resized.json')

training_metadata_file_val_only = \
    training_metadata_file.replace('.json','-val_only.json')
assert training_metadata_file_val_only != training_metadata_file    

val_image_folder = os.path.join(training_data_folder,'val')
assert os.path.isdir(val_image_folder)

yolo_dataset_file = os.path.join(os.path.dirname(model_file),'dataset.yaml')
assert os.path.isfile(yolo_dataset_file)
results_file = os.path.join(scratch_folder,'md_val_results.json')

preview_folder = os.path.join(scratch_folder,'preview')
preview_images_folder = os.path.join(preview_folder,'images')
os.makedirs(preview_images_folder,exist_ok=True)

target_image_size = (1280,-1)

parallelize_rendering = True
parallelize_rendering_n_cores = 10
parallelize_rendering_with_threads = False

force_render_images = False


#%% Run the model on the validation data

cmd = 'python run_inference_with_yolov5_val.py {} {} {} --model_type {}'.format(
    model_file,
    val_image_folder,
    results_file,
    model_type)

if model_type == 'yolov5':
    assert os.path.isdir(yolo_working_folder)
    cmd += ' --yolo_working_folder {}'.format(yolo_working_folder)
    
cmd += ' --overwrite_handling overwrite'
cmd += ' --yolo_dataset_file {}'.format(yolo_dataset_file)

if not augment:
    cmd += ' --augment_enabled 0'

print(cmd)
# import clipboard; clipboard.copy(cmd)


#%% Create a val-specific ground truth file

if os.path.isfile(training_metadata_file_val_only):
  
    print('Val ground truth file {} exists, bypassing'.format(training_metadata_file_val_only))
    
else:   
    
    with open(training_metadata_file,'r') as f:
        ground_truth_all = json.load(f)
        
    val_images = []
    
    for im in ground_truth_all['images']:
        if 'val/' in im['file_name']:
            im['file_name'] = im['file_name'].replace('val/','')
            val_images.append(im)
    
    val_image_ids = [im['id'] for im in val_images]
    val_image_ids = set(val_image_ids)
    
    val_annotations = []
    
    for ann in ground_truth_all['annotations']:
        if ann['image_id'] in val_image_ids:
            val_annotations.append(ann)
            
    print('Keeping {} of {} images and {} of {} annotations'.format(
        len(val_images),len(ground_truth_all['images']),
        len(val_annotations),len(ground_truth_all['annotations'])))
    
    ground_truth_val = ground_truth_all
    del ground_truth_all
    
    ground_truth_val['images'] = val_images
    ground_truth_val['annotations'] = val_annotations
    
    with open(training_metadata_file_val_only,'w') as f:
        json.dump(ground_truth_val,f,indent=1)
    

#%% Load val-specific ground truth

with open(training_metadata_file_val_only,'r') as f:
    ground_truth_val = json.load(f)

filename_to_ground_truth_im = {}
for im in ground_truth_val['images']:
    assert im['file_name'] not in filename_to_ground_truth_im
    filename_to_ground_truth_im[im['file_name']] = im


#%% Confirm that the ground truth file matches the val folder

val_images = find_images(val_image_folder,return_relative_paths=True,recursive=True)
assert len(val_images) == len(ground_truth_val['images'])
del val_images


#%% Map images to categories

gt_image_id_to_image = {im['id']:im for im in ground_truth_val['images']}
gt_image_id_to_annotations = defaultdict(list)

ground_truth_category_id_to_name = {}
for c in ground_truth_val['categories']:
    ground_truth_category_id_to_name[c['id']] = c['name']

ground_truth_category_names = sorted(list(ground_truth_category_id_to_name.values()))
    
for ann in ground_truth_val['annotations']:
    gt_image_id_to_annotations[ann['image_id']].append(ann)
    
gt_filename_to_category_names = defaultdict(set)

for im in ground_truth_val['images']:
    annotations_this_image = gt_image_id_to_annotations[im['id']]
    for ann in annotations_this_image:
        category_name = ground_truth_category_id_to_name[ann['category_id']]
        gt_filename_to_category_names[im['file_name']].add(category_name)
        
for filename in gt_filename_to_category_names:
    category_names_this_file = gt_filename_to_category_names[filename]
    if 'empty' in category_names_this_file:
        assert len(category_names_this_file)
    assert len(category_names_this_file) > 0

    
#%% Load results

with open(results_file,'r') as f:
    md_results = json.load(f)

results_category_id_to_name = md_results['detection_categories']


#%% Render images with detections    

def image_to_output_file(im):
    
    if isinstance(im,str):
        filename_relative = im
    else:
        filename_relative = im['file']
        
    fn_clean = flatten_path(filename_relative).replace(' ','_')
    return os.path.join(preview_images_folder,fn_clean)


def render_image(im):
    
    assert im['file'] in filename_to_ground_truth_im
    
    input_file = os.path.join(val_image_folder,im['file'])
    assert os.path.isfile(input_file)
                          
    output_file = image_to_output_file(im)
    if os.path.isfile(output_file) and not force_render_images:
        return output_file
    
    detections_to_render = []
    
    for det in im['detections']:
        category_name = results_category_id_to_name[det['category']]
        detection_threshold = rendering_confidence_thresholds['default']
        if category_name in rendering_confidence_thresholds:
            detection_threshold = rendering_confidence_thresholds[category_name]
        if det['conf'] > detection_threshold:
            detections_to_render.append(det)
        
    vis_utils.draw_bounding_boxes_on_file(input_file, output_file, detections_to_render,
                                          detector_label_map=results_category_id_to_name,
                                          label_font_size=20,target_size=target_image_size)
    
    return output_file


if parallelize_rendering:
    
    if parallelize_rendering_n_cores is None:                
        if parallelize_rendering_with_threads:
            pool = ThreadPool()
        else:
            pool = Pool()
    else:
        if parallelize_rendering_with_threads:
            pool = ThreadPool(parallelize_rendering_n_cores)
            worker_string = 'threads'
        else:
            pool = Pool(parallelize_rendering_n_cores)
            worker_string = 'processes'
        print('Rendering images with {} {}'.format(parallelize_rendering_n_cores,
                                                   worker_string))
        
    rendering_results = list(tqdm(pool.imap(render_image,md_results['images']),
                                  total=len(md_results['images'])))        

else:
    
    # im = md_results['images'][0]
    for im in tqdm(md_results['images']):    
        render_image(im)


#%% Map images to predicted categories, and vice-versa

filename_to_predicted_categories = defaultdict(set)
predicted_category_name_to_filenames = defaultdict(set)

# im = md_results['images'][0]
for im in tqdm(md_results['images']):
    
    assert im['file'] in filename_to_ground_truth_im
    
    # det = im['detections'][0]
    for det in im['detections']:
        category_name = results_category_id_to_name[det['category']]
        detection_threshold = confidence_thresholds['default']
        if category_name in confidence_thresholds:
            detection_threshold = confidence_thresholds[category_name]
        if det['conf'] > detection_threshold:
            filename_to_predicted_categories[im['file']].add(category_name)
            predicted_category_name_to_filenames[category_name].add(im['file'])
            
    # ...for each detection

# ...for each image


##%% Create TP/TN/FP/FN lists

category_name_to_image_lists = {}

# These may not be identical; currently the ground truth contains an "unknown" category
results_category_names = sorted(list(results_category_id_to_name.values()))

sub_page_tokens = ['fn','tn','fp','tp']

for category_name in ground_truth_category_names:
    
    category_name_to_image_lists[category_name] = {}
    for sub_page_token in sub_page_tokens:
        category_name_to_image_lists[category_name][sub_page_token] = []
    
# filename = next(iter(gt_filename_to_category_names))
for filename in gt_filename_to_category_names.keys():
    
    ground_truth_categories_this_image = gt_filename_to_category_names[filename]
    predicted_categories_this_image = filename_to_predicted_categories[filename]
    
    for category_name in ground_truth_category_names:
        
        assignment = None
        
        if category_name == 'empty':
            # If this is an empty image
            if category_name in ground_truth_categories_this_image:
                assert len(ground_truth_categories_this_image) == 1
                if len(predicted_categories_this_image) == 0:
                    assignment = 'tp'
                else:
                    assignment = 'fn'
            # This not an empty image
            else:
                if len(predicted_categories_this_image) == 0:
                    assignment = 'fp'
                else:
                    assignment = 'tn'
                
        else:
            if category_name in ground_truth_categories_this_image:
                if category_name in predicted_categories_this_image:
                    assignment = 'tp'
                else:
                    assignment = 'fn'
            else:
                if category_name in predicted_categories_this_image:
                    assignment = 'fp'
                else:
                    assignment = 'tn'        
                        
        category_name_to_image_lists[category_name][assignment].append(filename)
        
# ...for each filename


#%% Create confusion matrix

n_categories = len(ground_truth_category_names)
gt_category_name_to_category_index = {}

for i_category,category_name in enumerate(ground_truth_category_names):
    gt_category_name_to_category_index[category_name] = i_category    

# indexed as [true,predicted]
confusion_matrix = np.zeros(shape=(n_categories,n_categories),dtype=int)

filename_to_results_im = {im['file']:im for im in md_results['images']}

true_predicted_to_file_list = defaultdict(list)

# filename = next(iter(gt_filename_to_category_names.keys()))
for filename in gt_filename_to_category_names.keys():
    
    ground_truth_categories_this_image = gt_filename_to_category_names[filename]
    assert len(ground_truth_categories_this_image) == 1
    ground_truth_category_name = next(iter(ground_truth_categories_this_image))
    
    results_im = filename_to_results_im[filename]
    
    if len(results_im['detections']) == 0:
        predicted_category_name = 'empty'
    else:
        # Find all above-threshold detections
        results_category_name_to_confidence = defaultdict(int)
        for det in results_im['detections']:
            category_name = results_category_id_to_name[det['category']]
            detection_threshold = confidence_thresholds['default']
            if category_name in confidence_thresholds:
                detection_threshold = confidence_thresholds[category_name]
            if det['conf'] > detection_threshold:
                results_category_name_to_confidence[category_name] = max(
                    results_category_name_to_confidence[category_name],det['conf'])
            # If there were no detections above threshold
            if len(results_category_name_to_confidence) == 0:
                predicted_category_name = 'empty'
            else:
                predicted_category_name = max(results_category_name_to_confidence,
                    key=results_category_name_to_confidence.get)
    
    ground_truth_category_index = gt_category_name_to_category_index[ground_truth_category_name]
    predicted_category_index = gt_category_name_to_category_index[predicted_category_name]
    
    true_predicted_token = ground_truth_category_name + '_' + predicted_category_name
    true_predicted_to_file_list[true_predicted_token].append(filename)
    
    confusion_matrix[ground_truth_category_index,predicted_category_index] += 1

plt.ioff()    

fig_h = 3 + 0.3 * n_categories
fig_w = fig_h
fig = plt.figure(figsize=(fig_w, fig_h),tight_layout=True)
    
plot_utils.plot_confusion_matrix(
    matrix=confusion_matrix,
    classes=ground_truth_category_names,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues,
    vmax=1.0,
    use_colorbar=False,
    y_label=True,
    fig=fig)

cm_figure_fn_relative = 'confusion_matrix.png'
cm_figure_fn_abs = os.path.join(preview_folder, cm_figure_fn_relative)
# fig.show()
fig.savefig(cm_figure_fn_abs,dpi=100)
plt.close(fig)

# open_file(cm_figure_fn_abs)


#%% Create HTML confusion matrix

html_confusion_matrix = '<table class="result-table">\n'
html_confusion_matrix += '<tr>\n'
html_confusion_matrix += '<td>{}</td>\n'.format('True category')
for category_name in ground_truth_category_names:
    html_confusion_matrix += '<td>{}</td>\n'.format('&nbsp;')
html_confusion_matrix += '</tr>\n'

for true_category in ground_truth_category_names:
    
    html_confusion_matrix += '<tr>\n'
    html_confusion_matrix += '<td>{}</td>\n'.format(true_category)
    
    for predicted_category in ground_truth_category_names:
        
        true_predicted_token = true_category + '_' + predicted_category
        image_list = true_predicted_to_file_list[true_predicted_token]
        if len(image_list) == 0:
            td_content = '0'
        else:
            html_image_list_options = {}
            title_string = 'true: {}, predicted {}'.format(
                true_category,predicted_category)
            html_image_list_options['headerHtml'] = '<h1>{}</h1>'.format(title_string)
            
            html_image_info_list = []
            
            for image_filename_relative in image_list:
                html_image_info = {}
                detections = filename_to_results_im[image_filename_relative]['detections']
                if len(detections) == 0:
                    max_conf = 0
                else:
                    max_conf = max([d['conf'] for d in detections])
                
                title = '<b>Image</b>: {}, <b>Max conf</b>: {:0.3f}'.format(
                    image_filename_relative, max_conf)
                image_link = 'images/' + os.path.basename(image_to_output_file(image_filename_relative))
                html_image_info = {
                    'filename': image_link,
                    'title': title,
                    'textStyle':\
                     'font-family:verdana,arial,calibri;font-size:80%;' + \
                         'text-align:left;margin-top:20;margin-bottom:5'
                }                
                
                html_image_info_list.append(html_image_info)
            
            target_html_file_relative = true_predicted_token + '.html'
            target_html_file_abs = os.path.join(preview_folder,target_html_file_relative)
            write_html_image_list(
                filename=target_html_file_abs,
                images=html_image_info_list,
                options=html_image_list_options)
            
            td_content = '<a href="{}">{}</a>'.format(target_html_file_relative,
                                                      len(image_list))
        
        html_confusion_matrix += '<td>{}</td>\n'.format(td_content)
    
    # ...for each predicted category
    
    html_confusion_matrix += '</tr>\n'
    
# ...for each true category    

html_confusion_matrix += '<tr>\n'
html_confusion_matrix += '<td>&nbsp;</td>\n'

for category_name in ground_truth_category_names:
    html_confusion_matrix += '<td class="rotate"><p style="margin-left:20px;">{}</p></td>\n'.format(
        category_name)
html_confusion_matrix += '</tr>\n'

html_confusion_matrix += '</table>'

    
##%% Create HTML sub-pages and HTML table

html_table = '<table class="result-table">\n'

html_table += '<tr>\n'
html_table += '<td>{}</td>\n'.format('True category')
for sub_page_token in sub_page_tokens:
    html_table += '<td>{}</td>'.format(sub_page_token)
html_table += '</tr>\n'
    
filename_to_results_im = {im['file']:im for im in md_results['images']}

sub_page_token_to_page_name = {
    'fp':'false positives',
    'tp':'true positives',
    'fn':'false negatives',
    'tn':'true negatives'
}

# category_name = ground_truth_category_names[0]
for category_name in ground_truth_category_names:
    
    html_table += '<tr>\n'
    
    html_table += '<td>{}</td>\n'.format(category_name)
    
    # sub_page_token = sub_page_tokens[0]
    for sub_page_token in sub_page_tokens:
        
        html_table += '<td>\n'
        
        image_list = category_name_to_image_lists[category_name][sub_page_token]
        
        if len(image_list) == 0:
            
            html_table += '0\n'
            
        else:
            
            html_image_list_options = {}
            title_string = '{}: {}'.format(category_name,sub_page_token_to_page_name[sub_page_token])
            html_image_list_options['headerHtml'] = '<h1>{}</h1>'.format(title_string)
            
            target_html_file_relative = '{}_{}.html'.format(category_name,sub_page_token)
            target_html_file_abs = os.path.join(preview_folder,target_html_file_relative)
            
            html_image_info_list = []
            
            # image_filename_relative = image_list[0]
            for image_filename_relative in image_list:
                
                source_file = os.path.join(val_image_folder,image_filename_relative)
                assert os.path.isfile(source_file)
                
                html_image_info = {}
                detections = filename_to_results_im[image_filename_relative]['detections']
                if len(detections) == 0:
                    max_conf = 0
                else:
                    max_conf = max([d['conf'] for d in detections])
                
                title = '<b>Image</b>: {}, <b>Max conf</b>: {:0.3f}'.format(
                    image_filename_relative, max_conf)
                image_link = 'images/' + os.path.basename(image_to_output_file(image_filename_relative))
                html_image_info = {
                    'filename': image_link,
                    'title': title,
                    'linkTarget': source_file,
                    'textStyle':\
                     'font-family:verdana,arial,calibri;font-size:80%;' + \
                         'text-align:left;margin-top:20;margin-bottom:5'
                }                
                
                html_image_info_list.append(html_image_info)
                
            # ...for each image
                
            write_html_image_list(
                filename=target_html_file_abs,
                images=html_image_info_list,
                options=html_image_list_options)

            html_table += '<a href="{}">{}</a>\n'.format(target_html_file_relative,len(image_list))
        
        html_table += '</td>\n'

    # ...for each sub-page
        
    html_table += '</tr>\n'

# ...for each category
    
html_table += '</table>'        

html = '<html>\n'

style_header = """<head>
    <style type="text/css">
    a { text-decoration: none; }
    body { font-family: segoe ui, calibri, "trebuchet ms", verdana, arial, sans-serif; }
    div.contentdiv { margin-left: 20px; }
    table.result-table { border:1px solid black; border-collapse: collapse; margin-left:50px;}
    td,th { padding:10px; }
    .rotate {    
      padding:0px;
      writing-mode:vertical-lr;
      -webkit-transform: rotate(-180deg);        
      -moz-transform: rotate(-180deg);            
      -ms-transform: rotate(-180deg);         
      -o-transform: rotate(-180deg);         
      transform: rotate(-180deg);
    }
    </style>
    </head>"""
    
html += style_header + '\n'

html += '<body>\n'

html += '<h1>Results summary for {}</h1>\n'.format(job_name)

html += '<p><b>Model file</b>: {}</p>'.format(os.path.basename(model_file))

html += '<p><b>Augmentation</b>: {}</p>'.format('enabled' if augment else 'disabled')

html += '<p><b>Confidence thresholds</b></p>'

for c in confidence_thresholds.keys():
    html += '<p style="margin-left:15px;">{}: {}</p>'.format(c,confidence_thresholds[c])

html += '<h2>Confusion matrix</h2>\n'

html += '<p>...assuming a single category per image.</p>\n'

html += '<img src="{}"/>\n'.format(cm_figure_fn_relative)

html += '<h2>Confusion matrix (with links)</h2>\n'

html += '<p>...assuming a single category per image.</p>\n'

html += html_confusion_matrix

html += '<h2>Per-class statistics</h2>\n'

html += html_table

html += '</body>\n'
html += '<html>\n'

target_html_file = os.path.join(preview_folder,'index.html')

with open(target_html_file,'w') as f:
    f.write(html)
    
open_file(target_html_file,browser_name='chrome')
# import clipboard; clipboard.copy(target_html_file)