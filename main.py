#!/lab/solexa_young/scratch/jon_henninger/tools/venv/bin/python

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = 'sans-serif'

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import methods


def main(data_path, nuc_idx=0, pro_idx=1, threshold=None):
    # user input
    input_params = SimpleNamespace()

    input_params.parent_path = data_path
    input_params.nuc_idx = nuc_idx
    input_params.pro_idx = pro_idx
    
    if threshold is not None:
        input_params.threshold = threshold/65536
    else:
        input_params.threshold = threshold
    
    folder_list = os.listdir(input_params.parent_path)
    folder_list.sort(reverse=False)
    file_ext = '.czi'
    
    # make output directories    
    #input_params.output_path = input_params.parent_path
    input_params.output_path = '/lab/solexa_young/scratch/MECP2_Imaging/20191112_neuron_imaging/Volumes_extra/gaussian-sigma3_mean2.35_struct3_dist0.2'

    
    if not os.path.isdir(input_params.output_path):
        os.mkdir(input_params.output_path)
            
    for folder in folder_list:
        if not folder.startswith('.') and not folder.endswith('output') and os.path.isdir(os.path.join(input_params.parent_path, folder)): #SAMPLES/EXPERIMENTS        
            print()
            print('Started: ', folder, ' at ', datetime.now())
            print()
            
            temp_output = os.path.join(input_params.output_path, folder + '_output')            
            
            if not os.path.isdir(temp_output):
                os.mkdir(temp_output)
            
            file_list = os.listdir(os.path.join(input_params.parent_path, folder))
            base_name_files = [f for f in file_list if file_ext in f and os.path.isfile(os.path.join(input_params.parent_path, folder, f))]
            base_name_files.sort(reverse=False)
        
            excel_output = pd.DataFrame(columns=['sample', 'replicate_id', 'nuc_id', 'total_nuc_voxels', 'channel', 'mean_in', 'mean_out', 'norm_mean',
                                                'total_in', 'total_out', 'norm_total'])
                                                
            objects_output = pd.DataFrame(columns=['sample', 'replicate_id', 'nuc_id', 'object_id', 'voxels', 'channel', 'mean_in', 'mean_out', 'norm_mean'])
            
            replicate_count = 1
            for idx, file in enumerate(base_name_files):  #REPLICATES
                print()
                print(file)
                print()
                data = SimpleNamespace()
                data.sample_name = file.replace(file_ext,'')
                data.folder = folder
                data.img_path = os.path.join(input_params.parent_path, folder, file)
                
                data = methods.load_images(data, input_params)
                if data is not None:
                    data = methods.find_nucleus_2D(data, input_params)
                    data.z_count = data.nuc_img.shape[0]
            
                    if idx == 0:
                        # z = int(data.nucleus_image.shape[0]/2)
                        z = 10
                        # make_nucleus_montage_2D(data, input_params)
            
                    total_dense_object_mask = np.full(shape=data.nuc_img.shape, fill_value=False, dtype=bool)
                    for r_idx, region in enumerate(data.nuc_regions):
                        region_area = methods.find_region_area(region)
                        if region_area >= 30000:
                    
                            nuc_id = data.nuc_label[int((region[0].stop + region[0].start)/2), int((region[1].stop + region[1].start)/2) ]
                            nuc_box = data.nuc_img[:, region[0], region[1]]
                            nuc_mask_box = data.nuc_label[region[0], region[1]]
                            single_nuc_mask = nuc_mask_box == nuc_id
                            single_nuc_mask = np.repeat(single_nuc_mask[np.newaxis, :, :], data.z_count, axis=0)  # because our nuclear mask is 2D so we project it to 3D
                    
                            dense_obj_mask, bg_nuc_mask, dense_objects = methods.find_dense_objects_3D(nuc_box, single_nuc_mask, input_params, data)
                            total_dense_object_mask[:, region[0], region[1]][dense_obj_mask] = True                            
                    
                            for p_idx, image in enumerate(data.pro_imgs):
                                channel_name = data.pro_ch_names[p_idx]
                                protein_box = image[:, region[0], region[1]]
                            
                                mean_in = np.mean(protein_box[dense_obj_mask])
                                total_in = np.sum(protein_box[dense_obj_mask])
                        
                                mean_out = np.mean(protein_box[bg_nuc_mask])
                                total_out = total_in + np.sum(protein_box[bg_nuc_mask])
                        
                                norm_mean = mean_in/mean_out
                                norm_total = total_in/total_out
                            
                                nuc_voxels = np.sum(dense_obj_mask)
                        
                                excel_output = excel_output.append({'sample': folder,
                                                                   'replicate_id': replicate_count,
                                                                   'nuc_id': nuc_id,
                                                                   'total_voxels': nuc_voxels,
                                                                   'channel': str(channel_name),
                                                                   'mean_in': mean_in,
                                                                   'mean_out': mean_out,
                                                                   'norm_mean': norm_mean,
                                                                   'total_in': total_in,
                                                                   'total_out': total_out,
                                                                   'norm_total': norm_total},
                                                                   ignore_index=True)
                            
                                for o_idx, object in enumerate(dense_objects):
                                    voxels = np.sum(dense_obj_mask[object])
                                    mean_in = np.mean(protein_box[object])  # not perfect because this is just a 3D bounding box, which will include pixels not in the region, but good enough for now!
                                
                                    objects_output = objects_output.append({'sample': folder,
                                                                            'replicate_id': replicate_count,
                                                                            'nuc_id': nuc_id,
                                                                            'object_id': o_idx + 1,
                                                                            'voxels': voxels,
                                                                            'channel': str(channel_name),
                                                                            'mean_in': mean_in,
                                                                            'mean_out': mean_out,
                                                                            'norm_mean': mean_in/mean_out},
                                                                            ignore_index=True)
                                
                    graph_output_path = os.path.join(temp_output, folder + '_rep' + str(replicate_count) + '.png')
                    methods.make_output_graphs(data.nuc_label, total_dense_object_mask, data, graph_output_path)
                    replicate_count += 1
                else:
                    replicate_count += 1
                    
            excel_output.to_excel(os.path.join(temp_output, folder + '_enrichment.xlsx'), index=False)
            objects_output.to_excel(os.path.join(temp_output, folder + '_objects.xlsx'), index=False)
            
                        
if __name__ == "__main__":
    
    data_path = '/lab/solexa_young/scratch/MECP2_Imaging/20191112_neuron_imaging/Volumes_extra'

    main(data_path, nuc_idx=1, pro_idx=0)  # for czi files, have to hard code in which channel is which
        
    print('--------------------------------------')
    print('Completed at: ', datetime.now())
