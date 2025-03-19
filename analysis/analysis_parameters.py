
RUN_PATH = "/mnt/ssd/SSD_floricslimani/Fish_seq/Davide/2024-08-12 - SeqFISH - HeLa - Puro - R2TP1-2_Run7"
# RUN_PATH = "/mnt/ssd/SSD_floricslimani/Fish_seq/Davide/2024-10-09 - SeqFISH_Run10 - HeLa-Puro_POLR2"
FILTER_RNA = ['POLR2A_20']


#Distributions
distribution_measures = [
    'rna_number', 
    'cluster_number', 
    'proportion_rna_in_foci', 
    'nb_rna_in_nuc', 
    'index_mean_distance_nuc', 
    'index_mean_distance_cell'
    ]

# Density analysis
min_diversity = 3
min_spots_number = 3
cluster_radius = 400 #nm