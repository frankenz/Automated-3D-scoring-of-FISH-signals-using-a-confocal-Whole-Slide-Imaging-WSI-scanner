# Automated 3D scoring of FISH signals using a confocal Whole Slide Imaging (WSI) scanner

I developed an automated 3D scoring of FISH signals using a confocal Whole Slide Imaging (WSI) scanner. The algorithm employs 3D calculations for objects segmentation, signals detection and distribution of break-apart probes and copy number probes.

PseudoCode shown

Publication: https://pubmed.ncbi.nlm.nih.gov/33835321/


Abstract


Fluorescence in situ hybridization (FISH) is a technique to visualize specific DNA/RNA sequences within the cell nuclei and provide the presence, location and structural integrity of genes on chromosomes. A confocal Whole Slide Imaging (WSI) scanner technology has superior depth resolution compared to wide-field fluorescence imaging. Confocal WSI has the ability to perform serial optical sections with specimen imaging, which is critical for 3D tissue reconstruction for volumetric spatial analysis. The standard clinical manual scoring for FISH is labor-intensive, time-consuming and subjective. Application of multi-gene FISH analysis alongside 3D imaging, significantly increase the level of complexity required for an accurate 3D analysis. Therefore, the purpose of this study is to establish automated 3D FISH scoring for z-stack images from confocal WSI scanner. The algorithm and the application we developed, SHIMARIS PAFQ, successfully employs 3D calculations for clear individual cell nuclei segmentation, gene signals detection and distribution of break-apart probes signal patterns, including standard break-apart, and variant patterns due to truncation, and deletion, etc. The analysis was accurate and precise when compared with ground truth clinical manual counting and scoring reported in ten lymphoma and solid tumors cases. The algorithm and the application we developed, SHIMARIS PAFQ, is objective and more efficient than the conventional procedure. It enables the automated counting of more nuclei, precisely detecting additional abnormal signal variations in nuclei patterns and analyzes gigabyte multi-layer stacking imaging data of tissue samples from patients. Currently, we are developing a deep learning algorithm for automated tumor area detection to be integrated with SHIMARIS PAFQ.




![image](https://github.com/user-attachments/assets/db1e14ce-363a-4005-bde0-0f470591b6a3)





