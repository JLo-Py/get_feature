# coding: utf-8 -*-
# Author: Alberto Sainz Dalda <asainz.solarphysics@gmail.com>
# Date: 20230327
# License: 

# Modification history: 
# Juraj Lorincik, December 2023: handling colors of frames and contours, optional suppressing of box numbers
# Juraj Lorincik, August 2024: adding the option to save output graphics as pdf/eps/svg files, though these are still rasterized because of the background image

""" Routines to make contours and recover masks taking into account the 
    intentsity threshold, the aspect ratio, and the area. """

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import os
from iris_lmsalpy import tvg


# Example of running: 
# sel_masks, aux_ima, values_rect_incli = get_feature.do(image, 5e2, [1.5, 100], [min_area, max_area], reverse_gray = False, quiet = True, show_threshold_ima=False, threshold_in_selected_box = -5e2, savefig = savefig, dpi = 3e2)

def do(input_data, threshold_ima, aspect_cond, area_cond, 
       threshold_method = cv2.THRESH_BINARY, quiet=False, reverse_gray = False,
       show_threshold_ima = False, color_contour='k', color_boxincl='fuchsia',
       cmap='afmhot', sel_mask=[0,0,1], savefig='', manual=False, show_info=0, 
       interact = False, threshold_in_selected_box = None, **kwargs):

    """ It returns multi-valued mask for each contour found. 
    
    INPUTS: 
        input_data: either a string for a filename or a 2D Numpy array.
            If the input is a 2D Numpy array a temporary file named
            'aux_make_box_3217849.png' is created to get the BGR image.
            Then, that file is removed.
        threshold_ima: value to find the contours. Values enclosed in the
            contours are >= threshold_ima. Values go from 0 to 255.
        aspect_cond: [min_aspect_ratio, max_aspect_ration],
        area_cond: [min_area, max_area] in px x px.

    KEYWORDS:
        threshold_method: method used to find the contours.
        reverse_gray: considers the contours in the reversed grayscale image.
        show_threshold_ima: shows the threshold image.
        color_contour: color for the contours following the Matplolib 
            nomenclature, e.g.: 'C1', 'magenta'.
        color_boxincl: color for the inclined box following the Matplolib.
        cmap: colormap for the 2D Numpy array.
        sel_mask: [flag_to_store_rectangular_mask, 
                   flag_to_store_inclinedbox_mask,
                   flag_to_store_contour_mask]. 
                   If flag_to_store_rectangular_mask equals 1, mask=1
                   If flag_to_store_inclinedbox_mask equals 1, mask=3
                   If flag_to_store_contour_mask equals 1, mask=5
                   The values are then add, e.g. if sel_mask=[1,0,1],
                   then the values defining the rectangular area are 
                   1 and 6. The values defining the contour is 6.
        savefig: str with the name of the file to save the image with
            the contours.
        manual: select a threshold_ima value by clicking 'w' on the 
            displayed grayscale image
        threshold_in_selected_box: it selects a contour if the mean value 
            inside the contour if the mean is  < abs(threshold_in_selected_box) when 
            threshold_in_selected_box < 0, and if the 
            mean is > threshold_in_selected_box  when threshold_in_selected_box > 0
        **kwargs: keywords passed to plt.savefig, e.g. vmin and vmax.            

    OUTPUT:
        3D Numpy array containing the masks from the contours found.

    """                   

    color_cc =  [c*255 for c in colors.to_rgb(color_contour)][::-1]
    color_bincl =  [c*255 for c in colors.to_rgb(color_boxincl)][::-1]
    if isinstance(input_data, str) and os.path.isfile(input_data): img = cv2.imread(input_data)
    if isinstance(input_data, np.ndarray): 
        print('Creating auxiliar file aux_make_box_3217849.png')
        plt.imsave('aux_make_box_3217849.png', input_data, cmap=cmap, **kwargs)
        img = cv2.imread('aux_make_box_3217849.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print('Removing auxiliar file aux_make_box_3217849.png')
        print('Considering threshold = {}'.format(threshold_ima))
    #
    
    revgray = 1
    if reverse_gray:
        img_gray = cv2.cvtColor(-img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    if manual:
        sel = tvg.show(img_gray, origin='lower')
        if len(sel['ori_coord_values']) > 0:
            threshold_ima = sel['ori_coord_values'][0][-1]
            vmin = sel['lim_ima2show'][0]
            vmax = sel['lim_ima2show'][1]
            aux = img_gray.copy()
            aux[np.where((img_gray < vmin) | (img_gray > vmax))] = 0
            img_gray = aux.copy()
            print('Now considering threshold for img_gray: {}, within [{},{}]'.format(threshold_ima, vmin, vmax))


    ret , threshold = cv2.threshold(img_gray, threshold_ima, 255, threshold_method)
    
    # Finding the contours 
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ans = 'y'
    if interact == True:
        aux_ima = img_rgb.copy()
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 700, 600)
        if show_threshold_ima:
            cv2.imshow('output', img_gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imshow('output threshold', threshold)
            cv2.drawContours(threshold, contours[:10], -1, (0, 255, 0), 3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        ans = input('{} contours found. Proceed? [Y/n]'.format(len(contours)))


    if len(contours) > 1 and ans != 'n':
        i, k = 0, 0
        draw_contour = True 
        draw_rect = False
        draw_fit_rect = True 
        aux_ima = img_rgb.copy()
        allmasks = []
        values_rect_incli = []
        while i < len(contours): # and i < 100:
            if i % 500 ==0: print('Checking contour #{} of {}'.format(i, len(contours)))
            mask1 = img_gray*0
            mask2 = img_gray*0
            mask3= img_gray*0
            mask2total = img_gray*0
            cc = contours[i]
            contourArea = cv2.contourArea(cc)
            x,y,w,h = cv2.boundingRect(cc)
            if quiet == False:
                print('Contour area:', contourArea)
            if draw_contour:
                cv2.drawContours(aux_ima, contours, i, color_cc, 2)
            if 1:
                text = '{}'.format(i)
                if show_info > 0:
                    cv2.rectangle(aux_ima, (x, y), (x + w, y + h), (255,100,0), 2)
                    cv2.putText(aux_ima, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
            fit_rect = cv2.minAreaRect(cc)
            center = (int(fit_rect[0][0]),int(fit_rect[0][1]))
            width = int(fit_rect[1][0])
            height = int(fit_rect[1][1])
            angle = fit_rect[2]
            if quiet == False:
                print('Min. Area Rect:', fit_rect)
            if width < height:
                angle = 90 - angle
            else:
                angle = -angle
            box = cv2.boxPoints(fit_rect)
            if quiet == False:
                print('Width, height = ', width,height)
            aspect = np.max([width,height])/np.min([width,height]) #np.abs(1 - d1/d2)
            area = width*height
            cv2.drawContours(mask2total, contours, i, 5, -1)
            w = np.where(mask2total == 5)
            meanBox =  np.nanmean(img_gray[w])
            if quiet == False:
                print('Area cond: {} < {} < {}'.format(area_cond[0], area, area_cond[1]))
                print('Aspect cond: {} < {} < {}'.format(aspect_cond[0], aspect, aspect_cond[1]))
            cond = aspect > aspect_cond[0] and aspect < aspect_cond[1]  and area > area_cond[0] and area < area_cond[1]
            if threshold_in_selected_box != None:
                if quiet == False:
                    print('Considering mean value in selected feature', meanBox)
                if threshold_in_selected_box < 0: cond = cond and meanBox < abs(threshold_in_selected_box)
                if threshold_in_selected_box > 0: cond = cond and meanBox > threshold_in_selected_box 
            if draw_fit_rect and cond:
                p1, p2, p3, p4 = box
                if quiet == False:
                    print('***** Contour #{} *****'.format(k))
                    print(box, width, height, contourArea, aspect, area)
                    text = '{0}: {1:3.1f}'.format(k, angle)
                    print(text)
                    print('***********************')
                #fit_rect = cv2.minAreaRect(cc)
                # Retrieve the key parameters of the rotated bounding box
                box = cv2.boxPoints(fit_rect)
                box = np.int0(box)
                cv2.drawContours(aux_ima,[box],0,color_bincl,3)
                if sel_mask[0] != 0: cv2.rectangle(mask1, (x, y), (x + w, y + h), 1, 5)
                if sel_mask[1] != 0: cv2.drawContours(mask2,[box],0,3,-1)
                if sel_mask[2] != 0: cv2.drawContours(mask3, contours, i, 5, -1)
                if quiet == False:
                    print('Mean in box #{}: {}'.format(k, meanBox))
                text = '{0}: {1:3.1f} {2:5.1f}'.format(k, angle, meanBox)
                if show_info > 0: 
                    show_info = np.min([show_info,2])
                    cv2.putText(aux_ima, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, show_info, (0,0,255), 2)

                allmasks.append(mask1+mask2+mask3)
                values_rect_incli.append([center, width, height, angle])
                k+=1
            i+=1
        sel_masks = np.zeros([*mask1.shape[:2], len(allmasks)])
        for j, m  in enumerate(allmasks): sel_masks[:,:,j] = m       
    
        if savefig !='':
            
            if savefig[-3:] == 'eps' or savefig[-3:] == 'pdf' or savefig[-3:] == 'svg':
                plt.figure(figsize=kwargs['figsize'], dpi = kwargs.get('dpi', 150))
                ax=plt.axes([0, 0, 1, 1])
                mpli = ax.imshow(aux_ima)
                ax.set_axis_off() 
                plt.show(block = False)
                plt.savefig(savefig, dpi = kwargs.get('dpi', 150), bbox_inches='tight', pad_inches=0)
                
            else:
                plt.close('all')
                cv2.imwrite(savefig, aux_ima)
                  
        cv2.imshow('output', aux_ima)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    else:
        print('None contour was found.')
        sel_masks, aux_ima, values_rect_incli = -1, -1, -1

    return sel_masks, aux_ima, values_rect_incli



