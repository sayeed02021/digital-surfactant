import math

# Data
DATA_DEFAULT = {
    'max_sequence_length': 128,
    'padding_value': 0
}

# Properties


#PROPERTIES = [] # uncomment for unconditional generation 

#PROPERTIES = ['pCMC'] # uncomment for single property 

PROPERTIES = ['pCMC', 'Area_min', 'AW_ST_CMC'] # uncomment for multi property



PROPERTY_THRESHOLD = {
    'Area_min': 0.6,
    'AW_ST_CMC':31 
}
