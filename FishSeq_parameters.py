################

# 0. Global

################

RUN_PATH = "/media/floricslimani/SSD4To/SSD_floricslimani/Fish_seq/Davide/2024-08-12 - SeqFISH - HeLa - Puro - R2TP1-2_Run7/" #fullpath to main folder given by experimentalist
VOXEL_SIZE = (200,97,97) # size of a pixel in nanometer (z,y,x)
WASHOUT_KEY_WORD = 'Washout' #key for washout in gene map excel

################

# 1. Input

################
MAP_FILENAME = "HeLa-R2TP_Run7.xlsx"  #filename of required map file giving cycles names
CYCLE_KEY = "Cycle n."
GENES_NAMES_KEY = ["Gene1 (Ch1 - Cy5)", "Gene2 (Ch2 - Cy3)"] # Ordered by channel


cycle_regex = "img(\d+)_000_000000_0000000000.ome.tif" #regex to catch cycle number from tif filename.
FOLDER_KEYS = { # folder names where nucleus and fish images can be found (required keys : 'nucleus', 'fish')
    'nucleus' : "DAPI_Z-stacks",
    'fish' : "FISH_Z-stacks",
}



################

# 2. Segmentation

################
MODEL_DICT = {#cellpose model names, required keys : 'nucleus' and 'cytoplasm'
    'nucleus' : 'nuclei',
    'cytoplasm' : 'cyto3'
}

OBJECT_SIZE_DICT = {#size in px given to cellpose for prediction
    'nucleus' : 140,
    'cytoplasm' : 200
}

PLOT_VISUALS = True




################

# 3. Detection

##############
detection_MAX_WORKERS = 4 #Number of threads that can work simultaneously for detection; intensive process so recommended to not excess CPU cores number (currently : 4)

SPOT_SIZE = (200,100,100) #expected size of single molecules in nanometers (not less than voxel size)

#Big fish parameter for spot deconvolution;
ALPHA = 0.5  # aims at building reference spot as the 'alpha-percentile' of spot distribution. ex alpha= 0.5 --> reference spot = distribution median
BETA = 1 # aims at choosing regions to perform deconvolution on. Algorithm looks for regions with bright pixel connected. beta defines what bright pixel are with test : pixel_intensity >= MEDIAN_spot_intensity * beta (independantly of alpha)
GAMMA = 3 # size of kernel for gaussian blur performed before deconvolution.

CLUSTER_SIZE = 400 #size of cluster in nanometer (radius for DBScan algorithm)
MIN_SPOT_PER_CLUSTER = 4 #

ARTIFACT_RADIUS = 1400 # Radius of spots artifact to remove in nanometers.




################

# 4. Drift

##############
SAVE_PATH = RUN_PATH + '/visuals/'
BEAD_SIZE = (200, 200, 200) #size of fluorescent beads in nanometers used for fov aligment
FISH_THRESHOLD = None #Helpers for beads detection threshold
DAPI_THRESHOLD = None
DAPI_PENALTY = 2
FISH_PENALTY = 1
SLICE_TO_REMOVE = 10 # Number of slice to remove to avoid detecting noise





################

# 4. Quantification

##############
COLOC_DISTANCE = 300 #distance to consider for colocalization events in nanometers
quantif_MAX_WORKERS = 10 #Number of threads to use while computing cells features (small_process, currently good performance with 10)
COLOC_POPULATION = ('all', 'free', 'clustered') # population to consider when computing colocalisation
