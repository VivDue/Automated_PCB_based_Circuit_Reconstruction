# PCB Reconstruct

## Introduction

## Usage

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

### Component Detection


## Showroom

