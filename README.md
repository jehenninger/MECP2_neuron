# MECP2_neuron

## Code used for Li, Coffey et al. 2020

### Method

This script was used to quantify Hoechst-dense 3D regions in the nuclei of neurons. It uses functions from the scikit-image Python library. Nuclei were segmented on a max-projected image of nuclear signal. For segmentation, max-projected images were gaussian blurred (sigma = 2) and manually thresholded. Binary images were then subjected to a morphological opening and filling of holes, and then nuclear regions were labeled. To identify dense nuclear objects, the script looped through each nucleus individually. The 3D nucleus was subjected to a gaussian blur (sigma = 3), and was empirically thresholded by taking signal above the mean + 2.35*std. The binary image was then subjected to morphological opening and a watershedding algorithm to distinguish individual dense objects that were touching. The segmented regions were then labeled and the number of voxels per dense object was measured.
 
