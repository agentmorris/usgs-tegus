# USGS tegu detector

## Overview

The code in this repo trains, runs, and evaluates models to detect wildlife in camera trap images, particularly invasive [tegus](https://en.wikipedia.org/wiki/Tegu) in Florida.  This project is trained on data provided by the [USGS Fort Collins Science Center](https://www.usgs.gov/centers/fort-collins-science-center).

<img src="tegu-image.jpg" style="width:600px;">

## Important steps/files

These are listed in roughly the order in which you would use them.


### Map annotations to images, and split images into train/val sets

This happens in [usgs-tegus-training-data-prep.py](usgs-tegus-training-data-prep.py).  This is the only script referred to in this README that does not live in this repo; it has all sorts of local paths in it and is kept in a separate repo.

This script does the following:

* Match labels in the USGS .csv file with images on disk
* Map species-level labels to category-level labels (e.g. collapsing rodent species into one category)
* Split locations into train/val
* Copy images out to train/val folers to prepare for further annotation and training

The input to this step is:

* The original image folder
* The original .csv file

The output at the end of this step will be a series of folders that look like this:

```console
- train
  - blank
  - tegu
  - crow
  ...
 
- val
  - blank
  - tegu
  - crow
  ...
```

Within those folders, each filename contains the species label, the remapped category label, the location ID, and the original full path, with '/' replaced by '#'.  E.g. this image:

`train/green_iguana/green_iguana#IgIg#11E.11_C102#2017-2019#C102_11E.11#(19) 26APR18 - 10MAY18 AMH ARB#MFDC6689.JPG`

...was at this location on the original disk:

`2017-2019/C102_11E.11/(19) 26APR18 - 10MAY18 AMH ARB/MFDC6689.JPG`


### Turn species-level labels into bounding boxes

* Run [MegaDetector](https://github.com/agentmorris/MegaDetector/) on the data, using test-time augmentation and aggressive repeat detection elimination.  Details of that process are outside the scope of this README.
* Convert MegaDetector results to the format used by [labelme](https://github.com/wkentaro/labelme/), using a slightly different threshold for each class (e.g. 0.4 for raccoons, where we know MD works quite well, but something like 0.1 for tegus).  This happens in [prepare-labelme-files-from-md-results.py](prepare-labelme-files-from-md-results.py). 
* Review every image in labelme (I'm specifically using [this fork of labelme](https://github.com/agentmorris/labelme)), fixing up any boxes that are broken/missing/etc.  Along the way, delete images where it's not possible to assign boxes (typically images where the animal isn't visible), and images with multiple species present (very rare).
* After manual cleanup of the data, review boxes to make sure they look sensible.

The input to this step is:

* Just the image folders written by the previous step, metadata is in folder names.

The output at the end of this step is:

* labelme-formatted .json files associated with each image, e.g., for the sample image referred to above:

`train/green_iguana/green_iguana#IgIg#11E.11_C102#2017-2019#C102_11E.11#(19) 26APR18 - 10MAY18 AMH ARB#MFDC6689.json`


### Convert to COCO format, and preview the COCO dataset to make sure everything still looks sensible

This happened via [labelme_to_coco.py](https://github.com/agentmorris/MegaDetector/blob/main/data_management/labelme_to_coco.py); the driver code is in [prepare-labelme-files-from-md-results.py](prepare-labelme-files-from-md-results.py).  Along the way, I also resized most images to 1600px on the long side in-place, with bounding boxes resized accordingly; this happened via [resize_coco_dataset.py](https://github.com/agentmorris/MegaDetector/blob/main/data_management/resize_coco_dataset.py).  At the end of this step, we do a complete resize operation on the whole database (images and .json).

The input to this step is:

* The train/val image folders (with copies of the images at their original sizes)
* The labelme .json files

The output from this step is:

* A copy of the train/val image folders in which everything has been resized to 1600px wide
* A COCO-formatted .json file containing all the train/val images and labels


### Convert to YOLO format, discarding a few classes, adding out-of-domain blanks

This happens in [prepare-yolo-training-set.py](prepare-yolo-training-set.py).  Specifically, this script does the following:

* Split the COCO-formatted data into train/val sets. This is not strictly necessary, it's just handy to have COCO files later for the train and val data separately.
* Previews the train/val sets to make sure everything looks sensible
* Converts to YOLO training format (YOLO-formatted annotations and a YOLOv5 dataset.yaml file).  As we do this, we also:
  * Sample blanks randomly
  * Exclude everything in the "blanks and very small things" folder that isn't blank: insects, tiny reptiles, etc.  I just decided to punt on these.
  * Exclude the "other" and "unknown" categories
* Optionally, we also add a sample of out-of-domain blank images at the end of this script.  Blanks are originally fetched from LILA and organized via [create_lila_blank_set.py](https://github.com/agentmorris/MegaDetector/blob/main/data_management/lila/create_lila_blank_set.py), but the splitting into train/val and the resizing to 1600px for faster training happens here.


### Train

Training (using MDv5 as a starting point) happens at the CLI, but [train-usgs-tegu-detector.py](train-usgs-tegu-detector.py) tracks all the commands used to train, and also contains cells that:

* Run the YOLOv5 validation scripts
* Convert YOLOv5 val results to MD .json format
* Use the MD visualization pipeline to visualize results
* Use the MD inference pipeline to run the trained model


### Postprocess and review results

[review-usgs-tegu-results.py](review-usgs-tegu-results.py) is a notebook that:

* Runs a trained model on the validation data
* Renders detections onto thumbnail images
* Generates confusion matrices and HTML preview pages to present the results

## TODOs 

* Since we're mostly interested in tegus, experiment with resampling the training data to reduce the number of raccoons and crows
* Combine the YOLOv5 and YOLOv8 outputs to maximize tegu F1
* Include other large-reptile datasets, likely lumping other reptiles (e.g. goannas) into the "tegu" class
* Hyperparameter optimization, in particular try freezing some layers during YOLOv8 training
* This is mostly a curiosity, but... it's unknown how much training benefits from using MD as a starting point; compare to training YOLOv5x6 from a COCO-trained starting point
* Maybe this whole approach is overkill; try training an image classifier rather than a detector

