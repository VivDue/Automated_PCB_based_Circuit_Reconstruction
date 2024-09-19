from ultralytics import YOLO
import glob
import torch
import os 
import cv2
import numpy as np

from importlib.metadata import version

class YoloPredictor:
    def __init__(self, yoloModel, input_file, output_dir, cut_ends):
        self.input_file = input_file
        self.output_directory = output_dir
        self.final_out = output_dir
        self.yolomodel = yoloModel
        self.cut_ends = cut_ends
        image_name = self.input_file.split('\\')[-1]
        image_name = image_name.split(".")[0]
        self.image_name = image_name

    def predict_file(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"Using device: {device}")

        self.output_directory =  f'{self.output_directory}/{self.image_name}'

        self._create_directories(self.output_directory)
        self._patch_it(self.input_file, 768)
        files = glob.glob(self.output_directory + "/patches/*")

        model = YOLO(self.yolomodel)
        results = model.predict(files, save=False, verbose=False)
        
        for i, result in enumerate(results):
            tmp = str(i)
            if(len(tmp) == 1):
                tmp = '00' + tmp
            if(len(tmp) == 2):
                tmp = '0' + tmp
            result.save(filename=f'{self.output_directory}/patches_predict/{self.image_name}_{tmp}.png')
            #if i == 1:
            #    result.save_txt(txt_file=f'{self.output_directory}/patches_predict/{self.image_name}_{str(i)}.txt')
        self._recombine(self.input_file, 768)

    def _recombine(self, input_file:str, patch_size:int)->None:
        original = cv2.imread(input_file)

        #get faktor of how many images to concatenate
        height, width, _ = original.shape       
        height_faktor = height / patch_size
        width_faktor = width / patch_size

        #concat images
        i = 0
        vert_while = 0 #horizontal recombine
        main_cut = None
        while vert_while < height_faktor:
            if(vert_while == 0):
                i, main, main_cut = self._hor_recombine(self.image_name, i, width_faktor, width)
                curr_height = 768
            else:
                i, imgpart, imgpart_cut = self._hor_recombine(self.image_name, i, width_faktor, width)
                curr_height = curr_height + 768
                if(curr_height > height):
                    snip = curr_height - height
                    main_cut = np.concatenate((main_cut, imgpart_cut[snip:, :]), axis=0)
                    main = np.concatenate((main, imgpart), axis=0)
                else:
                    main_cut = np.concatenate((main_cut, imgpart_cut), axis=0)
                    main = np.concatenate((main, imgpart), axis=0)
            vert_while = vert_while + 1
        if main_cut is None:
            main_cut = main

        cv2.imwrite(f'{self.final_out}/Cut/{self.image_name}_Cut.png', main_cut)
        cv2.imwrite(f'{self.final_out}/All/{self.image_name}_all.png', main)

        cv2.imwrite(f'{self.output_directory}/{self.image_name}_Cut.png', main_cut)
        cv2.imwrite(f'{self.output_directory}/{self.image_name}_all.png', main)



    def _hor_recombine(self, image_name, i, width_faktor, width):
        hor_while = 0 #horizontal recombine
        main_cut = None
        while hor_while < width_faktor:
            tmp = str(i)
            if(len(tmp) == 1):
                tmp = '00' + tmp
            if(len(tmp) == 2):
                tmp = '0' + tmp
            if(hor_while == 0):
                main = cv2.imread(f'{self.output_directory}/patches_predict/{image_name}_{tmp}.png')
                curr_width = 768
            else:
                imgpart = cv2.imread(f'{self.output_directory}/patches_predict/{image_name}_{tmp}.png')
                curr_width = curr_width + 768
                if(curr_width > width):
                    snip = curr_width - width
                    main_cut = np.concatenate((main, imgpart[:, snip:]), axis=1)
                    main = np.concatenate((main, imgpart), axis=1)
                else:
                    main = np.concatenate((main, imgpart), axis=1)
            i = i + 1
            hor_while = hor_while + 1
        if main_cut is None:
            main_cut = main
        return i, main, main_cut

        

    def _patch_it(self, input_file:str, patch_size:int)->None:
        image = cv2.imread(input_file)
        height, width, _ = image.shape
        patches = []
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                temp = image[i:i+patch_size, j:j+patch_size]
                # check if the extracted image is smaller than patch size
                # and move the starting x and y point to match the given patch size  
                if(height > patch_size):
                    if temp.shape[0] < patch_size:
                        i = height - patch_size
                if(width > patch_size):
                    if temp.shape[1] < patch_size:
                        j = width - patch_size
                patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        for i, patch in enumerate(patches):
            tmp = str(i)
            if(len(tmp) == 1):
                tmp = '00' + tmp
            if(len(tmp) == 2):
                tmp = '0' + tmp
            cv2.imwrite(f'{self.output_directory}/patches/{self.image_name}_{tmp}.png', patch)

    def _create_directories(self, output_dir:str)->None:
        """
        Create the output directories if they don't already exist.

        :param output_dir: Path to the output directory.
        """
        # create output directory if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create ann and img subdirectories
        if not os.path.exists(output_dir + "/patches"):
            os.makedirs(output_dir + "/patches")

        if not os.path.exists(output_dir + "/patches_predict"):
            os.makedirs(output_dir + "/patches_predict")

        if not os.path.exists(self.final_out + "/Cut"):
            os.makedirs(self.final_out + "/Cut")
        if not os.path.exists(self.final_out + "/All"):
            os.makedirs(self.final_out + "/All")
    