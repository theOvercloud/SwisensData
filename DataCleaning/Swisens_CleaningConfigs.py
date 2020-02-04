"""
    Swisens Data Cleaning Configs module
    ------------
    This module is for setting configurations for the cleaning process with the Swisens_DataCleaning.py module.

    ---------
    Changes:
    0.0.
    
        
    ########################################
    Author: Elias Graf
    (c) Swisens AG
    Email Address: elias.graf@hslu.ch
    Nov 2019; Last revision: 11-Nov-2019
    ########################################

"""


# CONFIGS



# Camera
cam_offset = -2.2e-3

##################################
# Data Cleaning
##################################

# Json
write_imgProperties = True
# Load only zipped JSON files
ending = '.json.gz'


# Default json file
json_default = {
    "valid": True,
    "trig_tdiff": 0.0,
    "sipm_data":[],
    "timestamp":0.0,
    "holo0": {
        "xy": [0.0,0.0],
        "zr": 0.0,
        "zf": 0.0
    },
    "holo1": {
        "xy": [0.0,0.0],
        "zr": 0.0,
        "zf": 0.0
    },
    "velocity": 0.0,
    "adcdump": {
        "0A": [],
        "0B": [],
        "1A": [],
        "1B": [],
        "2A": [],
        "2B": []
    }
}

# image validation
area_range=(625,10e3)
sol_range=(0.9,1)
minorAxis_range=(0,float("inf"))
majorAxis_range=(0,float("inf"))
perimeter_range=(0,float("inf"))
maxIntensity_range=(0,float("inf"))
minIntensity_range=(0,float("inf"))
meanIntensity_range=(0,float("inf"))
eccentricity_range=(0,float("inf"))

# Clean by Trigger
cleanTrigger = False
display=False
## find peaks
peak_width=5
peak_prominence=2e3
peak_distance=50
peak_minDeviation=-10e3
peak_number=2
## Low level (no peak)
lowLevel_indOffset=100
lowLevelOffset_start=3e3  
lowLevelOffset_end=7e3
## First and second trigger peak must be in index range
range_first=(600,830)
range_second=(950,1100)