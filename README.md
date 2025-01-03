# How to Train and Deploy YOLO Models with Ultralytics (YOLO11, YOLOv8, and YOLOv5)
Tutorials and examples showing how to train and deploy Ultralytics YOLO models.

**This repository is a work in progress!**

Click below to acces the initial Colab notebook for training YOLO models. I am still working on finalizing the notebook, but it currently includes all the steps needed for training.

<a href="https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Script Usage Instructions
The `yolo_detect.py` script provides a basic example that shows how to load a model, run inference on an image source, parse the inference results, and display boxes around each detected class in the image. This script shows how to work with YOLO models in Python, and it can be used as a starting point for more advanced applications. 

To download `yolo_detect.py` from this repository, issue: 

```
curl --output yolo_detect.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/yolo_detect.py
```

To run inference with a yolov8s model on a USB camera at 1280x720 resolution, issue:

```
python yolo_detect.py --model yolov8s.pt --source usb0 --resolution 1280x720
```

Here are all the arguments for yolo_detect.py:

- `--model`: Path to a model file (e.g. `my_model.pt`). If the model isn't found, it will default to using `yolov8s.pt`.
- `--source`: Source to run inference on. The options are:
    - Image file (example: `test.jpg`)
    - Folder of images (example: `my_images/test`)
    - Video file (example: `testvid.mp4`)
    - Index of a connected USB camera (example: `usb0`)
- `--thresh` (optional): Minimum confidence threshold for displaying detected objects. Default value is 0.5 (example: `0.4`)
- `--resolution` (optional): Resolution in WxH to display inference results at. If not specified, the program will match use the source resolution. (example: `1280x720`)
- `--record` (optional): Record a video of the results and save it as `demo1.avi`. (If using this option, the `--resolution` argument must also be specified.)
