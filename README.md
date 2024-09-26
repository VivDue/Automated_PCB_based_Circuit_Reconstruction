# Automated PCB based Circuit Reconstruction

## Introduction
Reconstruction and analysis of PCB's by hand is very complex and time consuming. This work aims to make some aspects of this easier by introducing two tool for PCB-Image analysis.

### Trace Detection
The trace detection is used to create a netlist of the PCB and connect the components on it. An image of the bare PCB is need for the trace detection to work. The image is then processed in multiple steps to remove the silkscreen, create masks of all pads and traces and connect the top and bottom layers.
![Mask Detection](assets/trace_detect_masking_process.png)

In it's current form the tool returns the resulting netlist in 2 images. One shows all traces sorted by top and bottom layer in red and blue while the second image shows the connected netlist, giving each net a unique color.
![Net Detection](assets/trace_detect_netlist_process.png)

### Component detection
For the component detection multiple neural networks have been trained using the YOLOv8 framework and FPIC-Dataset (see related works), the best working one been the 3 net, with split images and updated labels. The network can detect the common SMD and THT components reliably and returns bounding boxes and segmentation masks for detected components.
![Model Overview](assets/nn_training_prediction.png)

## Related Work
„EE 368 : Reverse Engineering of Printed Circuit Boards“ by Ben Johnson at Stanford University in 2013
In his work Ben Johnson introduced a program he wrote in C++ to detect traces and pads on a PCB and detect components by comparing footprints from a library with the processed images.
For our work we took inspiration from his approach to trace detection and expanded on it.


„FPIC: A Novel Semantic Dataset for Optical PCB Assurance“ by Nathan Jessurun, Olivia P. Dizon-Paradis and others publicized in the „ACM Journal on Emerging Technologies in Computing Systems“ Volume 19; DOI: doi.org/10.1145/3588032
In this work the authors introduce a publicly available labeled PCB-dataset for training of neural networks and describe the steps involved in it's creation. 
We used this dataset to train our neural networks and later edited parts of it to improve the trainingresults.


„PCBSegClassNet“ by Dhruv Makwana, Sai Chandra Teja R. and Sparsh Mittal publicized in „Expert Systems with Applications“ Volume 225; DOI: doi.org/10.1016/j.eswa.2023.120029	
The first published project to use the FPIC-dataset to train a neural network for component detection and classification from images. The authors introduced multiple interesting ways to improve the learning behavior of their neural network which we used a inspiration for our preprocessing of the dataset.


### Addition Works
„Automated PCB Reverse Engineering“ by Stephan Kleber, Henrik Ferdinand Nölscher and Frank Kargl publicized on „ WOOT'17: Proceedings of the 11th USENIX Conference on Offensive Technologies“ in August 2017

„FICS-PCB: A Multi-Modal Image Dataset for Automated Printed Circuit Board Visual Inspection” by Hangwei Lu, Olivia Paradis and others in July 2020




## Installation

To install Automated_PCB_based_Circuit_Reconstruction and its dependencies, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/VivDue/Automated_PCB_based_Circuit_Reconstruction.git
    ```

2. Navigate to the cloned directory and replace your path with your save path:

    ```bash
    cd your_path/Automated_PCB_based_Circuit_Reconstruction
    ```

3. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv .venv
    ```

4. Activate the virtual environment:

    - **Windows:**

    ```bash
    .venv\Scripts\activate.bat
    ```

    - **Linux/macOS:**

    ```bash
    source .venv/bin/activate
    ```

5. Install the required dependencies and replace the your_path with your save path:

    - `Using` requirements.txt:
    ```bash
    python -m pip install --upgrade pip
    python -m pip install -r your_path/requirements.txt
    ```

    - `Alternatively`, install packages individually:
    If the requirements.txt installation fails, you can install the necessary packages manually:
    ```bash
    python -m pip install opencv-python
    python -m pip install matplotlib
    python -m pip install ultralytics
    python -m pip install simple-lama-inpainting
    python -m pip install tqdm
    python -m pip install ipykernel
    ```

6. You are now ready to use the Automated_PCB_based_Circuit_Reconstruction in your Python projects.

## Usage

### Trace Detection

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

### Dataset Preprocessing
To convert the FPIC-Datasets annotations into YOLOv8-Format our `S3A_yolo_converter` is used.
The converter can be found in it's own [repository](https://github.com/VivDue/s3a_yolo_converter). It does annotation conversion, Image-to-Patch splitting and HSI-CLAHE preprocessing.
The updated annotations which we used to train the 3rd and 4th neural net can be found in `training/annotations_updated`.


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
input_file = "Testimages\\b1.webp"
output_directory = "Predictions"
yolomodel = "../models/yolov8m_updated_labels.pt"

# Whether the prediction should be used on the images as a whole
#  or split the image in patches beforehand.
#   (should be False for Net 1 and True for Net 2, 3 and 4)
patches = True

#Set line width for bounding boxes and font size
line_width_in = 2
font_size_in = 2

# create an instance of the Predictor
predictor = YoloPredictor(yolomodel, input_file, output_directory)
predictor.predict_file(patches, line_width_in, font_size_in)
```

The code can be found in `Predict\Prediction_skripts.ipynb` with another example to process whole folders in one go.

## Showroom
### Trace Detection
![Trace Mask Top](assets/showroom_trace_detect_1.png)

![Trace Mask Botom](assets/showroom_trace_detect_2.png)

![Networks](assets/showroom_trace_detect_3.png)

### Component Detection
<img src="assets/showroom_predict_2.png" alt="drawing" width="100%"/>
