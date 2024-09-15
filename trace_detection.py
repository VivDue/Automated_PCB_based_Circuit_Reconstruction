import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from simple_lama_inpainting import SimpleLama

class TraceDetection:
    def __init__(self):
        self.colors = [
                (0, 255, 0),    # green
                (0, 0, 255),    # red
                (255, 0, 0),    # blue
                (255, 255, 0),  # yellow
                (0, 255, 255),  #cyan
                (255, 0, 255),  # magenta
                (255, 165, 0),  # orange 
                (191, 255, 0),  # lime
                (255, 105, 180),# pink
                (148, 0, 211),  # violett
                (255, 215, 0),  # gold
                (64, 224, 208), # turqouise
                (255, 127, 80), # coral
                (0, 128, 0),    # dark green
                (128, 0, 0),    # dark red
                (0, 0, 128),    # dark blue
                (128, 128, 0),  # olive
                (0, 128, 128),  # teal
                (128, 0, 128),  # purple
                ]

    def create_masks(self, layer_images):
        """
        """
        masks = {
            "silkscreen": [],
            "traces": [],
            "pads": [],
            "vias": [],
            "inpainted": []
        }

        for layer_image in tqdm(layer_images):

            # mask_silkscreen
            silkscreen_mask = self._mask_silkscreen(layer_image)

            # inpaint
            inpainted_image = self._inpaint(layer_image, silkscreen_mask)

            # mask_pads
            pad_mask = self._mask_pads(inpainted_image)
            
            # mask_traces
            trace_mask = self._mask_traces(inpainted_image, pad_mask)
            
            # save masks
            masks["silkscreen"].append(silkscreen_mask)
            masks["traces"].append(trace_mask)
            masks["pads"].append(pad_mask)
            masks["inpainted"].append(inpainted_image)

        return masks
    
    def create_net_list(self, masks, mirror = [False, True]):

        # Mirror layers if necessary
        mirrored_masks = self._mirror_layers(masks, mirror)

        # Align layers
        aligned_masks = self._align_layers(mirrored_masks)

        # Initialize list to store nets
        net_list = []

        # Loop over each layer
        for i in range(len(masks["pads"])):
            # Get the contours and hierarchy of the traces and pads
            contours_traces, hierarchy_traces = cv2.findContours(aligned_masks["traces"][i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_pads, hierarchy_pads = cv2.findContours(aligned_masks["pads"][i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Add traces and vias to net list
            net_list = self._add_traces_and_vias_to_net_list(contours_traces, hierarchy_traces, net_list, i)

            # Add pads to net list
            net_list = self._add_pads_to_net_list(contours_pads, hierarchy_pads, net_list, i)

        # Group the net list
        image = masks["inpainted"][0]
        grouped_net_list = self._group_net_list(image,net_list)

        return grouped_net_list

    def show(self, layer_images, masks = None, net_list = None):
        """
        """

        if masks is not None:
            for i in range(len(layer_images)):
                fig, axs = plt.subplots(2, 3, figsize=(15, 8))

                # show original image
                if layer_images[i] is not None:
                    axs[0, 0].imshow(layer_images[i])
                    axs[0, 0].set_title('Original Image')
                    axs[0, 0].axis('off')

                # show silkscreen mask
                if masks["silkscreen"][i] is not None:
                    axs[0, 1].imshow(masks["silkscreen"][i], cmap='gray')
                    axs[0, 1].set_title('Silkscreen Mask')
                    axs[0, 1].axis('off')

                # show inpainted image
                if masks["inpainted"][i] is not None:
                    axs[0, 2].imshow(masks["inpainted"][i])
                    axs[0, 2].set_title('Inpainted Image')
                    axs[0, 2].axis('off')

                # show pad mask
                if masks["pads"][i] is not None:
                    axs[1, 0].imshow(masks["pads"][i], cmap='gray')
                    axs[1, 0].set_title('Pad Mask')
                    axs[1, 0].axis('off')

                # show trace mask
                if masks["traces"][i] is not None:
                    axs[1, 1].imshow(masks["traces"][i], cmap='gray')
                    axs[1, 1].set_title('Trace Mask')
                    axs[1, 1].axis('off')

                # show traces
                if masks["traces"][i] is not None:
                    # find contours
                    contours, hierarchy = cv2.findContours(masks["traces"][i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    traces = self._draw_contours_by_level(masks["traces"][i], contours, hierarchy, self.colors)
                    axs[1, 2].imshow(traces)
                    axs[1, 2].set_title('Traces')
                    axs[1, 2].axis('off')

                plt.show()

        if net_list is not None:
            fig, axs = plt.subplots(2, 3, figsize=(15, 8))

            # Draw Traces, Pads and Vias for top, bottom, combined layer wise
            top, bottom, combined = self._draw_contours_by_layer(net_list, layer_images[0])
            axs[0, 0].imshow(top)
            axs[0, 0].set_title('Top Layer')
            axs[0, 0].axis('off')

            axs[0, 1].imshow(bottom)
            axs[0, 1].set_title('Bottom Layer')
            axs[0, 1].axis('off')

            axs[0, 2].imshow(combined)
            axs[0, 2].set_title('Combined')
            axs[0, 2].axis('off')

            # Draw grouped nets
            top, bottom, combined = self._draw_countours_by_group(net_list, layer_images[0])
            axs[1, 0].imshow(top)
            axs[1, 0].set_title('Top Layer')
            axs[1, 0].axis('off')

            axs[1, 1].imshow(bottom)
            axs[1, 1].set_title('Bottom Layer')
            axs[1, 1].axis('off')

            axs[1, 2].imshow(combined)
            axs[1, 2].set_title('Combined')
            axs[1, 2].axis('off')


####################################################################################################
# per layer functions
####################################################################################################
    
    def _mask_silkscreen(self, layer_image):
        
        # thresholding
        gray_image = cv2.cvtColor(layer_image, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image,200,255,cv2.THRESH_BINARY)

        # morphological operations to remove noise
        kernel = kernel = np.ones((3,3),np.uint8)
        erode_image = cv2.erode(thresh_image, kernel, iterations=1)
        dilate_image = cv2.dilate(erode_image, kernel, iterations=6)
        silkscreen_mask = cv2.medianBlur(dilate_image, 3)

        return silkscreen_mask

    def _mask_pads(self, inpainted_image):
        
        # convert to hsv and get saturation channel
        saturation_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2HSV)[:,:,1]

        # otsu thresholding
        _, thresh_image = cv2.threshold(saturation_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # morphological operations to remove noise
        kernel = np.ones((3,3),np.uint8)
        blured_image = cv2.medianBlur(thresh_image, 15)
        pad_mask = cv2.erode(blured_image, kernel, iterations=1)

        return pad_mask
    
    def _mask_traces(self, inpainted_image, pad_mask):

        # detect most common color 
        #color = np.bincount(inpainted_image.flatten()).argmax()

        # remove pads from inpainted image
        inpainted_image[pad_mask == 0] = 255

        # reshape image to 2D array of pixels and 3 color values (RGB)
        Z = inpainted_image.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        _,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # convert back to 8 bit values and create the clustered image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((inpainted_image.shape))

        # thresholding
        gray_image = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image,68,255,cv2.THRESH_BINARY)

        # morphological operations to remove noise
        kernel = np.ones((3,3),np.uint8)
        erode_image = cv2.erode(thresh_image, kernel, iterations=1)
        trace_mask = cv2.medianBlur(erode_image, 5)

        return trace_mask
    
    def _inpaint(self, layer_image, mask): 
        # lama inpainting model
        simple_lama = SimpleLama()

        # inpaint
        inpainted_image = simple_lama(layer_image, mask)
        inpainted_image = np.array(inpainted_image)

        return inpainted_image
    
    def _draw_contours_by_level(self, image, contours, hierarchy, colors):
        contour_counter = 0
        background = image.copy()
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        for i, _ in enumerate(contours):
            # determine the level of the contour
            level = 0
            h = hierarchy[0][i]
            while h[3] != -1:
                level += 1
                h = hierarchy[0][h[3]]
            
            # choose the color based on the level
            color = colors[min(level, len(colors) - 1)]  
            
            # draw the contour
            if level == 2:
                contour_color = self.colors[contour_counter % len(self.colors)]
                cv2.drawContours(background, contours, i, contour_color, -1) 
                contour_counter = contour_counter + 1
            else:
                pass
                # cv2.drawContours(background, contours, i, color, -1)

        return background


####################################################################################################
# per board functions
####################################################################################################
    
    def _mirror_layers(self, masks, mirror):
        # Initialize list to store mirrored masks
        mirrored_masks = {
            "silkscreen": [],
            "traces": [],
            "pads": [],
            "inpainted": []
        }

        silkscreen = masks["silkscreen"]
        traces = masks["traces"]
        pads = masks["pads"]
        inpainted = masks["inpainted"]

        # Loop over each layer
        for i in range(len(pads)):
            # Check if the layer should be mirrored
            if mirror[i]:
                # Mirror the masks
                mirrored_silkscreen = cv2.flip(silkscreen[i], 1)
                mirrored_traces = cv2.flip(traces[i], 1)
                mirrored_pads = cv2.flip(pads[i], 1)
                mirrored_inpainted = cv2.flip(inpainted[i], 1)

                # Append mirrored masks to list
                mirrored_masks["silkscreen"].append(mirrored_silkscreen)
                mirrored_masks["traces"].append(mirrored_traces)
                mirrored_masks["pads"].append(mirrored_pads)
                mirrored_masks["inpainted"].append(mirrored_inpainted)

            else:
                # Append original masks to list
                mirrored_masks["silkscreen"].append(silkscreen[i])
                mirrored_masks["traces"].append(traces[i])
                mirrored_masks["pads"].append(pads[i])
                mirrored_masks["inpainted"].append(inpainted[i])

        return mirrored_masks

    def _align_layers(self, masks):
        # Initialize list to store aligned masks
        # The first mask in each list is the reference mask (no offset)
        aligned_masks = {
            "silkscreen": [masks["silkscreen"][0]],
            "traces": [masks["traces"][0]],
            "pads": [masks["pads"][0]],
            "inpainted": [masks["inpainted"][0]]
        }

        silkscreen = masks["silkscreen"]
        traces = masks["traces"]
        pads = masks["pads"]
        inpainted = masks["inpainted"]

        # Invert and convert pads to float32
        
        pads_f32 = [np.float32(cv2.bitwise_not(pad)) for pad in pads]

        # Loop over each layer
        for i in range(len(pads)-1):
            
            # Detect offset between first and current layer
            offset = cv2.phaseCorrelate(pads_f32[0], pads_f32[i+1])[0]

            # Round offset to nearest integer
            offset = -1*np.round(offset, 0).astype(int)

            # Apply offset to all masks
            aligned_silkscreen = self._offset_mask(silkscreen[i+1], offset)
            aligned_traces = self._offset_mask(traces[i+1], offset)
            aligned_pads = self._offset_mask(pads[i+1], offset)
            aligned_inpainted = self._offset_mask(inpainted[i+1], offset)

            # Append aligned masks to list
            aligned_masks["silkscreen"].append(aligned_silkscreen)
            aligned_masks["traces"].append(aligned_traces)
            aligned_masks["pads"].append(aligned_pads)
            aligned_masks["inpainted"].append(aligned_inpainted)

        return aligned_masks

    def _offset_mask(self, mask, offset):
        # shift the bottom pads to the right
        offset_mask = np.roll(mask, offset[0], axis=1)

        # shift the bottom pads to the bottom
        offset_mask = np.roll(offset_mask, offset[1], axis=0)

        return offset_mask

    def _add_traces_and_vias_to_net_list(self, contours, hierarchy, net_list, layer):
        # Loop over each contour and its corresponding hierarchy
        for i, contour in enumerate(contours):
            # Start by determining the level of the current contour (nesting depth)
            level = 0
            h = hierarchy[0][i]
            parent_idx = h[3]  # h[3] is the index of the parent contour
            
            # Traverse the hierarchy upwards to determine the contour's level
            while parent_idx != -1:
                level += 1
                parent_idx = hierarchy[0][parent_idx][3]
            
            # Process only if the contour is at level 1 (inside another contour)
            if level == 2:

                # Collect all via contours (siblings) by traversing the hierarchy
                via_contours = []
                child_idx = h[2]  # h[2] is the index of the first child contour

                # Traverse all sibling contours (vias) at the same level
                while child_idx != -1:
                    via_contour = contours[child_idx]
                    enlarged_via_contour = self._enlarge_contours(via_contour)
                    via_contours.append(enlarged_via_contour)
                    child_idx = hierarchy[0][child_idx][0]  # Move to the next sibling contour

                trace_contour = contour
                # Append the current contour and its corresponding via contours to the net_list
                net_list.append([trace_contour, None, via_contours if via_contours else None, layer])
        
        return net_list
    
    def _enlarge_contours(self, contour, enlargement_factor=10):
        """
        Enlarges a contour by applying dilation to a mask created from the contour.
        """
        # Create a mask from the contour
        mask = np.zeros((1000, 1000), dtype=np.uint8)

        # Draw the contour on the mask
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Apply dilation to the mask
        kernel = np.ones((enlargement_factor, enlargement_factor), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Find the new contour from the dilated mask
        new_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return new_contours[0] if new_contours else contour

    def _add_pads_to_net_list(self, contours, hierarchy, net_list, layer):
        # Loop over each contour and its corresponding hierarchy
        for i, contour in enumerate(contours):

            # Start by determining the level of the current contour (nesting depth)
            level = 0
            h = hierarchy[0][i]
            parent_idx = h[3]  # h[3] is the index of the parent contour
            
            # Traverse the hierarchy upwards to determine the contour's level
            while parent_idx != -1:
                level += 1
                parent_idx = hierarchy[0][parent_idx][3]
            
            # Process only if the contour is at level 0
            if level == 1:

                # Each of theis contours is a pad
                pad_contour = contour
                net_list.append([None, pad_contour, None, layer])
        
        return net_list

    def _group_net_list(self, image, net_list):
        """
        Group nets based on the overlap between pads, vias, and traces,
        but excluding trace-trace overlap.
        """
        grouped_nets = []
        
        # Iterate over each net in the list
        for net in net_list:
            trace_contour = net[0]
            pad_contours = net[1] if net[1] is not None else []
            via_contours = net[2] if net[2] is not None else []

            # List to store indices of groups that overlap with the current net
            overlapping_groups = []

            # Iterate over existing groups to find overlap with pads, vias, or traces (ignoring trace-trace overlap)
            for group_idx, group in enumerate(grouped_nets):
                group_has_overlap = False

                for existing_net in group:
                    existing_trace = existing_net[0]
                    existing_pads = existing_net[1] if existing_net[1] is not None else []
                    existing_vias = existing_net[2] if existing_net[2] is not None else []

                    # We don't check trace-trace overlap

                    # Check if the current pads overlap with existing pads or traces in the group
                    for pad in pad_contours:
                        for existing_pad in existing_pads:
                            if self._contours_overlap(image, pad, existing_pad):
                                group_has_overlap = True
                        if existing_trace is not None and self._contours_overlap(image, pad, existing_trace):
                            group_has_overlap = True
                    
                    # Check if vias overlap with existing vias or traces
                    for via in via_contours:
                        for existing_via in existing_vias:
                            if self._contours_overlap(image, via, existing_via):
                                group_has_overlap = True
                        if existing_trace is not None and self._contours_overlap(image, via, existing_trace):
                            group_has_overlap = True

                    # Check if traces overlap with vias
                    if trace_contour is not None:
                        for existing_via in existing_vias:
                            if self._contours_overlap(image, trace_contour, existing_via):
                                group_has_overlap = True

                if group_has_overlap:
                    overlapping_groups.append(group_idx)

            # Merge overlapping groups
            if overlapping_groups:
                # Merge all overlapping groups and add the current net
                merged_group = []
                for group_idx in sorted(overlapping_groups, reverse=True):
                    merged_group.extend(grouped_nets.pop(group_idx))
                merged_group.append(net)
                grouped_nets.append(merged_group)
            else:
                # If no overlap found, create a new group
                grouped_nets.append([net])

        return grouped_nets
    
    def _contours_overlap(self, image, contour1, contour2):
            """ Check if two contours overlap using bounding rectangles and intersection tests. """
            rect1 = cv2.boundingRect(contour1)
            rect2 = cv2.boundingRect(contour2)
            
            # Check if rectangles intersect
            if (rect1[0] < rect2[0] + rect2[2] and
                rect1[0] + rect1[2] > rect2[0] and
                rect1[1] < rect2[1] + rect2[3] and
                rect1[1] + rect1[3] > rect2[1]):
                # Perform more accurate check with mask intersection
                mask1 = np.zeros(image.shape[:2], dtype=np.uint8)
                mask2 = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask1, [contour1], -1, 255, -1)
                cv2.drawContours(mask2, [contour2], -1, 255, -1)
                intersection = np.logical_and(mask1, mask2)
                return np.any(intersection)
            return False

    def _draw_contours_by_layer(self, grouped_nets, image):

        # Create three white backgrounds: Layer 1 (Top), Layer 2 (Bottom), and a combined view
        white_back_top = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
        white_back_bottom = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
        white_back_combined = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255

        # Traces Top (red), Bottom (blue)
        # Pads Top (green), Bottom (yellow)
        # Vias Top (cyan), Bottom (magenta)
        colors = {
            "trace": [(0, 0, 255), (255, 0, 0)],
            "pad": [(0, 255, 0), (255, 255, 0)],
            "via": [(255, 0, 255), (0, 255, 255)]
        }

        # create net_list from grouped_nets
        net_list = []
        for group in grouped_nets:
            for net in group:
                net_list.append(net)

        # Draw nets by layer
        for net in net_list:
            trace_contour = net[0]
            pad_contours = net[1]
            via_contours = net[2]
            layer = net[3]


            # Depending on the layer, draw on the corresponding background with specified color
            if layer == 1:
                if trace_contour is not None:
                    cv2.drawContours(white_back_top, [trace_contour], -1, colors["trace"][0], -1)  # Draw on combined
                if pad_contours is not None:
                    for pad in pad_contours:
                        if pad is not None:
                            cv2.drawContours(white_back_top, [pad], -1, colors["pad"][0], -1)
                if via_contours is not None:
                    for via in via_contours:
                        if via is not None:
                            cv2.drawContours(white_back_top, [via], -1, colors["via"][0], -1)

            else:
                if trace_contour is not None:
                    cv2.drawContours(white_back_bottom, [trace_contour], -1, colors["trace"][1], -1)  # Draw on combined
                if pad_contours is not None:
                    for pad in pad_contours:
                        if pad is not None:
                            cv2.drawContours(white_back_bottom, [pad], -1, colors["pad"][1], -1)
                if via_contours is not None:
                    for via in via_contours:
                        if via is not None:
                            cv2.drawContours(white_back_bottom, [via], -1, colors["via"][1], -1)

            # Draw on combined view
            if trace_contour is not None:
                cv2.drawContours(white_back_combined, [trace_contour], -1, colors["trace"][layer-1], -1)  # Draw on combined
            if pad_contours is not None:
                for pad in pad_contours:
                    if pad is not None:
                        cv2.drawContours(white_back_combined, [pad], -1, colors["pad"][layer-1], -1)
            if via_contours is not None:
                for via in via_contours:
                    if via is not None:
                        cv2.drawContours(white_back_combined, [via], -1, colors["via"][layer-1], -1)

        return white_back_top, white_back_bottom, white_back_combined

    def _draw_countours_by_group(self, grouped_nets, image):
        """ Draw grouped nets with separate plots for top and bottom layers, and a combined plot. """
        # Create three white backgrounds: Layer 1 (Top), Layer 2 (Bottom), and a combined view
        white_back_top = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
        white_back_bottom = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
        white_back_combined = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255

        # Iterate over grouped nets and assign colors
        for group in grouped_nets:
            # Get a color for the group
            color = self.colors[len(group) % len(self.colors)]
            
            # Iterate over nets in the group
            for net in group:
                trace_contour = net[0]
                pad_contours = net[1]
                layer = net[3]

                # Depending on the layer, draw on the corresponding background with modified color
                if layer == 1:
                    background = white_back_top
                else:
                    background = white_back_bottom

                # Draw the trace in group color
                if trace_contour is not None:
                    cv2.drawContours(background, [trace_contour], -1, color, -1)  # Filled contours
                    cv2.drawContours(white_back_combined, [trace_contour], -1, color, -1)  # Draw on combined

                # Draw the pads in the same group color
                if pad_contours is not None:
                    for pad in pad_contours:
                        cv2.drawContours(background, [pad], -1, color, -1)
                        cv2.drawContours(white_back_combined, [pad], -1, color, -1)

        return white_back_top, white_back_bottom, white_back_combined