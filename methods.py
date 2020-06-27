from skimage import filters, exposure
from skimage.color import label2rgb
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as nd
from skimage import img_as_ubyte, img_as_float
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, canny
import imageio as io
import cv2
import os
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import czifile


def get_file_extension(file_path):
    file_ext = os.path.splitext(file_path)
    
    return file_ext[1]  # because splitext returns a tuple and the extension is the second element

def find_image_channel_name(file_name):
    str_idx = file_name.find('Conf ')  # this is specific to our microscopes file name format
    channel_name = file_name[str_idx + 5 : str_idx + 8]

    return channel_name
    
def max_project(img):
    projection = np.max(img, axis=0)
    
    return projection

def clear_axis_ticks(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    
def make_color_image(img, c):
    output = np.zeros(shape=(img.shape[0], img.shape[0], 3))

    if c == 'green':
        output[..., 0] = 0.0  # R
        output[..., 1] = img  # G
        output[..., 2] = 0.0  # B
    elif c == 'magenta':
        output[..., 0] = img  # R
        output[..., 1] = 0.0  # G
        output[..., 2] = img  # B
    elif c == 'cyan':
        output[..., 0] = 0.0  # R
        output[..., 1] = img  # G
        output[..., 2] = img  # B
        
    else:
        print('ERROR: Could not identify color to pseudocolor image in grapher.make_color_image')
        sys.exit(0)

    return output


def find_region_area(r):
    a = ((r[0].stop - r[0].start)) * ((r[1].stop - r[1].start))
    
    return a
    
def find_region_volume(r):
    a = (r[0].stop - r[0].start) * (r[1].stop - r[1].start) * (r[2].stop - r[2].start)
    
    return a


# load images for replicate
def load_images(data, input_params):
    
    with czifile.CziFile(data.img_path) as czi:
        img = czi.asarray()
        img = np.squeeze(img)  # this gets rid of the extra dimensions if they are 1 (e.g. if it isn't a timecourse)
        # metadata = czi.metadata  # if you want metadata. But I don't currently parse this.
    
    # NUCLEUS IMAGE
    if len(img.shape) < 4:
        return None
    
    else:
        data.nuc_img =  img[input_params.nuc_idx, :, :, :]# image is [z, x, y] array
    
        '''
        if data.nuc_img.dtype == 'float32':
            data.nuc_img = data.nuc_img.astype(np.uint16)  # my pipeline to align image stacks keeps values but in 32-bit format. This conflicts with later processing steps so we have to trick it.
        '''
    
        # PROTEIN IMAGES
        data.pro_imgs = [img[input_params.pro_idx, :, : ,:]]
        data.pro_ch_names = ['ch488']
    
        '''
        for idx, p in enumerate(protein_image_files):
            data.protein_image_paths.append(os.path.join(input_params.parent_path, data.folder, p))
            data.protein_channel_names.append(find_image_channel_name(p))
            data.pro_imgs.append(io.volread(data.protein_image_paths[idx]))
        
            if data.pro_imgs[idx].dtype == 'float32':
                data.pro_imgs[idx] = data.pro_imgs[idx].astype(np.uint16) # my pipeline to align image stacks keeps values but in 32-bit format. This conflicts with later processing steps so we have to trick it.
        '''     
    
        return data

    
def find_nucleus_2D(data, input_params):
    
    img = max_project(data.nuc_img)
#     med_img = img_as_ubyte(img)
#     med_img = cv2.medianBlur(med_img, ksize=5)
#     med_img = img_as_float(med_img)
    
    med_img = filters.gaussian(img, sigma=2)
    # threshold = np.mean(med_img) + 0.05*np.std(med_img)
    threshold = 1000/65536
    
    # if input_params.threshold is None:
    #     threshold = filters.threshold_otsu(med_img)
#     else:
#         threshold = input_params.threshold
    
    
    # nuc_mask = canny(med_img, sigma=1)
    
    nuc_mask = med_img >= threshold
    
    nuc_mask = nd.morphology.binary_opening(nuc_mask)
    # nuc_mask = nd.morphology.binary_dilation(nuc_mask)
    nuc_mask = nd.morphology.binary_fill_holes(nuc_mask)
    #nuclear_label, num_features = nd.label(nuclear_mask)
    #nuclear_label = find_watershed_3D(nuclear_mask)
    
    ## WATERSHEDDING ##
    if False:
        nuc_label = find_watershed_2D(nuc_mask)
        nuc_mask = nuc_label >= 1
    
    nuc_label, _ = nd.label(nuc_mask)
    
    data.nuc_mask = nuc_mask
    data.nuc_label = nuc_label
    
    data.nuc_regions = nd.find_objects(nuc_label)
    
    return data


def find_watershed_2D(z_slice):
    labels, _ = nd.label(z_slice)
    distance = nd.distance_transform_edt(z_slice)
    local_maxi = peak_local_max(distance, min_distance=50, indices=False, labels=labels, exclude_border=True)
    markers = nd.label(local_maxi)[0]
    
    output = watershed(-distance, markers, mask=z_slice, watershed_line=True)
    
    return output


def make_nucleus_montage_2D(data, input_params):
    
    fig, ax = plt.subplots(1, 3)
    
    disp_nuc_img = max_project(data.nuc_img)
    disp_nuc_img = exposure.equalize_adapthist(disp_nuc_img)
    disp_nuc_img = make_color_image(disp_nuc_img, 'cyan')
    
    ax[0].imshow(disp_nuc_img)
    ax[1].imshow(data.nuc_mask, cmap='gray')
    
    labeled_image = label2rgb(data.nuc_label, image=disp_nuc_img,
                             alpha=0.5, bg_label=0, bg_color=[0, 0, 0])
   
    ax[2].imshow(labeled_image)
    
    for a in ax:
        clear_axis_ticks(a)
        
    plt.tight_layout()
    plt.show()


def find_dense_objects_3D(img, mask, input_params, data):
    # bg_mean is uint16
    #img[np.invert(mask)] = bg_mean
    
    img = img_as_float(img)
    
    ''''
    #subtract background using median filter
    med_img = img_as_ubyte(img) 
    for i in range(med_img.shape[0]):
        med_img[i, :, :] = cv2.medianBlur(med_img[i,:, :], ksize=13)

    med_img = img_as_float(med_img)
    img = img - med_img
    img[np.where(img < 0)] = 0
    '''
    
    # just do simple gaussian filter and thresholding (z plotting)


    
    img = filters.gaussian(img, sigma=3)
    threshold = np.mean(img) + 2.35*np.std(img)
    
    object_mask = np.logical_and(mask, img >= threshold)
    # object_mask = nd.morphology.binary_opening(object_mask)
    object_mask = nd.morphology.binary_opening(object_mask, iterations=3) # we are trying to see if you can open until nothing changes
    bg_nuclear_mask = np.logical_and(mask, img < threshold)
    
    #parameters
    fg_dist_threshold = 0.2
    struct_size = 3
    erosion_iterations = 1
    distance = nd.distance_transform_edt(object_mask)
    z_size = img.shape[0]


    if False:
        ### WATERSHED TEST
            labels, _ = nd.label(object_mask)
    
            distance = nd.distance_transform_edt(object_mask)
    
            ##custom faster solution for getting markers
            sure_fg = distance
            sure_fg[sure_fg <= fg_dist_threshold*distance.max()] = 0.0
            sure_fg = sure_fg > 0.0
        
            ci_struct = make_struct_element(struct_size, shape='circle', dim=2).astype(img.dtype)
            ew_struct = make_struct_element(struct_size, shape='ellipse_wide', dim=2).astype(img.dtype)
            et_struct = make_struct_element(struct_size, shape='ellipse_tall', dim=2).astype(img.dtype)
        
            for z in range(z_size):
                sure_fg[z, :, :] = nd.morphology.binary_erosion(sure_fg[z, :, :], structure=ci_struct, iterations=erosion_iterations)
                # sure_fg[z, :, :] = nd.morphology.binary_erosion(sure_fg[z, :, :], structure=ew_struct, iterations=erosion_iterations)
                # sure_fg[z, :, :] = nd.morphology.binary_erosion(sure_fg[z, :, :], structure=et_struct, iterations=erosion_iterations)
    
            markers, num_regions = nd.label(sure_fg)
            row, col = optimum_subplots(object_mask.shape[0])
            fig, ax = plt.subplots(row, col)
    
            ax = ax.flatten()
    
            for idx, a in enumerate(ax):
                if idx < z_size:
                    labeled_image = label2rgb(markers[idx, :, :], image=exposure.equalize_adapthist(img[idx, :, :]),
                                     alpha=0.3, bg_label=0, bg_color=[0, 0, 0])
                    a.imshow(labeled_image)
                
                clear_axis_ticks(a)
        
            plt.savefig(os.path.join(input_params.output_path, data.sample_name + '_watershed_test.png'), dpi=150)
            plt.close()
        
    ### WATERSHED
    
   
    if True:
        ##custom faster solution for getting markers
        sure_fg = distance
        sure_fg[sure_fg <= fg_dist_threshold*distance.max()] = 0
        sure_fg = sure_fg > 0
    
        ci_struct = make_struct_element(struct_size, shape='circle', dim=2).astype(img.dtype)
    
        for z in range(z_size):
            sure_fg[z, :, :] = nd.morphology.binary_erosion(sure_fg[z, :, :], structure=ci_struct, iterations=erosion_iterations)

        markers, num_regions = nd.label(sure_fg)
        dense_object_label = watershed(-distance, markers, mask=object_mask, watershed_line=True)
        object_mask = dense_object_label > 0
    
    
        # dense_object_label, _ = nd.label(object_mask)
        dense_objects = nd.find_objects(dense_object_label)
    
        # new_object_mask = np.full(shape=img.shape, fill_value=False, dtype=bool)
        # filtered_objects = []
    
    vol_threshold = 500
    for idx, object in enumerate(dense_objects):
        if find_region_volume(object) < vol_threshold:
            object_mask[object] = 0
            dense_objects[idx] = []
            
            
    dense_objects = [o for o in dense_objects if o]
    
    if False:
        under_img = exposure.equalize_adapthist(img[z, :, :])
        labeled_img = label2rgb(object_mask[z,:,:], image=under_img,
                           alpha=0.5, bg_label=0, bg_color=[0,0,0])
        
        labeled_bg_img = label2rgb(bg_nuclear_mask[z,:,:], image=under_img,
                           alpha=0.5, bg_label=0, bg_color=[0,0,0])
    
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(img[z, :, :], cmap='magma')
        ax[1].imshow(labeled_img)
        ax[2].imshow(labeled_bg_img)
    
    '''
    # just do simple gaussian filter and thresholding (max_z plotting)
    if False: 
        img = filters.gaussian(img, sigma=2)
        
        
        threshold = np.mean(img) + 2*np.std(img)
        
        object_mask = np.logical_and(mask, img >= threshold)
        object_mask = nd.morphology.binary_opening(object_mask)
        
        under_img = exposure.equalize_adapthist(max_project(img))
        labeled_img = label2rgb(max_project(object_mask), image=under_img,
                               alpha=0.5, bg_label=0, bg_color=[0,0,0])
        
        
        
        bg_nuclear_mask = np.logical_and(mask, img < threshold)
        labeled_bg_img = label2rgb(max_project(bg_nuclear_mask), image=under_img,
                               alpha=0.5, bg_label=0, bg_color=[0,0,0])
        if False:
            fig, ax = plt.subplots(1, 3)

            ax[0].imshow(max_project(img), cmap='magma')
            ax[1].imshow(labeled_img)
            ax[2].imshow(labeled_bg_img)
            
            for a in ax:
                clear_axis_ticks(a)
            
            plt.show()
    '''
    
    '''

    threshold = filters.threshold_otsu(img)
    object_mask = img >= threshold
    object_mask = nd.morphology.binary_opening(object_mask)

    labeled_image = label2rgb(max_project(object_mask), image=max_project(img),
                         alpha=0.5, bg_label=0, bg_color=[0, 0, 0])
    '''
    return object_mask, bg_nuclear_mask, dense_objects


def make_output_graphs(nuc_label, obj_mask, data, output_path):
    num_of_proteins = len(data.pro_imgs)
    fig_h = 3.3  # empirical
    fig_w = 4.23 * num_of_proteins # empirical
    
    fig, ax = plt.subplots(1, 2+num_of_proteins)
    
    # 1st image is nuclear mask
    nuc_under_img = exposure.equalize_adapthist(max_project(data.nuc_img))
    nuc_labeled_img = label2rgb(nuc_label, image=nuc_under_img,
                                alpha=0.5, bg_label=0, bg_color=[0, 0 ,0])
    ax[0].imshow(nuc_labeled_img)
    ax[0].set_title('nuclear mask', fontsize=10)

    for r_idx, region in enumerate(data.nuc_regions):
            region_area = find_region_area(region)
            if region_area >= 30000:
                region_center_r = int((region[0].stop + region[0].start)/2)
                region_center_c = int((region[1].stop + region[1].start)/2)
                nuc_id = data.nuc_label[region_center_r, region_center_c]
                ax[0].text(region_center_c, region_center_r, str(nuc_id),
                           fontsize='6', color='w', horizontalalignment='center', verticalalignment='center')
    
    # 2nd image is total object mask
    object_labeled_img = label2rgb(max_project(obj_mask), image=nuc_under_img,
                                  alpha=0.5, bg_label=0, bg_color=[0, 0, 0])
    ax[1].imshow(object_labeled_img)
    ax[1].set_title('object mask', fontsize=10)
    
    # the rest are the protein images with boxes around
    
    for p_idx, img in enumerate(data.pro_imgs):
        img = img_as_float(img)
        temp_under_img = exposure.equalize_adapthist(max_project(img))
        temp_labeled_img = label2rgb(max_project(obj_mask), image=temp_under_img,
                               alpha=0.25, bg_label=0, bg_color=[0,0,0])
        ax[2+p_idx].imshow(temp_labeled_img)
        ax[2+p_idx].set_title(str(data.pro_ch_names[p_idx]))    
    
    for a in ax:
        clear_axis_ticks(a)
        
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.90, wspace=0.1, hspace=0.1)

    plt.savefig(output_path,dpi=300)
    plt.close()


def make_struct_element(edge_size, shape='circle', dim=3):
    
    if edge_size % 2 == 0:
        print('Error: Structuring element must have an odd edge size')
        sys.exit(0)
    else:
        if shape != 'circle':
            edge_size = edge_size + 2
            
        X, Y = np.ogrid[0:edge_size, 0:edge_size]
        center = math.floor(edge_size/2)
        scale_factor = 1./math.sqrt(edge_size)
        
        if shape == 'circle':
            element = ((X-center)**2 + (Y-center)**2 < edge_size).astype(np.uint8)
        elif shape == 'ellipse_tall':
            element = (scale_factor * (X-center)**2 + (Y-center)**2 < edge_size).astype(np.uint8)
        elif shape == 'ellipse_wide':
            element = ((X-center)**2 + scale_factor * (Y-center)**2 < edge_size).astype(np.uint8)
        else:
            print('Error: Could not recognize shape input for making structuring element')
            sys.exit(0)
            
        if dim == 3:
            # for now, we will just make the z 1 unit thick
            element = element[np.newaxis,:, :]
    
    '''
    ## STRUCT ELEMENT TEST
    print(f'Shape is {shape}')
    print(element)           
    print()
    '''
    
    return element
    
    
def optimum_subplots(n):
    row = math.floor(math.sqrt(n))
    col = math.ceil(n/row)
    
    return row, col