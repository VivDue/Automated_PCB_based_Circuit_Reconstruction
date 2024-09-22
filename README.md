# PCB Reconstruct

Component Detection 

## Introduction


![Mask Detection](assets/trace_detect_masking_process.png)

![Net Detection](assets/trace_detect_netlist_process.png)

![Model Overview](assets/nn_training_prediction.png)

## Installation

## Related Work

## Trace Detection

### Usage

The `TraceDetection` class provides functionality to process PCB layer images and generate masks, create net lists, and visualize the results. The key methods are:

- `create_masks(layer_images)`: Generates masks for the background, silkscreen, traces, pads, vias, and inpainted images from the input layer images.
- `create_net_list(masks)`: Creates a net list based on the masks, which groups nets by traces and pads.
- `show(layer_images, masks=None, net_list=None, save=False, path=None)`: Visualizes the original layer images, masks, and net list in a grid. Optionally, the results can be saved to a specified path.

### Example

```python
from trace_detection import TraceDetection
import cv2

# Load images for each layer
top_layer = cv2.imread('images/pcbs/0001_bin_alarm/base_top.png')
bottom_layer = cv2.imread('images/pcbs/0001_bin_alarm/base_bottom.png')
layer_images = [top_layer, bottom_layer]

# Create Masks and show results
td = TraceDetection()
masks = td.create_masks(layer_images)
td.show(layer_images, masks)

# Create net list and show results
net_list = td.create_net_list(masks, mirror=[False, True])
td.show(layer_images, None, net_list)
```

### Component Detection

To predict the bounding boxes of components in an image using YOLO, you can use the `predict_file()` method. This method works as follows:

1. **Patch Generation**: The input image (typically a PCB image) is split into smaller patches, each of size 768x768 pixels. This ensures that the YOLO model can process the image without running into memory or size limitations.
2. **YOLO Prediction**: Each patch is fed into the YOLO model to detect the components. The detected components are saved as new images, with bounding boxes drawn around the detected objects.
3. **Recombination**: Once all patches have been processed, the method recombines them into the original image format. This can be saved in two forms:
   - The **Cut** image: Cropped to remove unnecessary borders.
   - The **All** image: The full recombined image without any cropping.

### Example

```python
from yolo_predict import YoloPredictor

# files and directories
input_file = "test\DSC_0250.jpg"
output_directory = "Predictions"
yolomodel = "models/yolov8m_updated_labels.pt"
cut_ends = False

# create an instance of the DesignatorCopy
predictor = YoloPredictor(yolomodel, input_file, output_directory, cut_ends)
predictor.predict_file()
```

## Showroom



