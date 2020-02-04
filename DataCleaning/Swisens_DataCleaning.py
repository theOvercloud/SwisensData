"""
    Swisens Data Cleaning module
    
    ########################################
    
    This module is applicable to raw measurement data of the Swisens Poleno.
    Raw datasets are meant to be in the form of zipped directories containing zipped Json event files and png Holoimage files.
    The cleaned datasets are saved as filesystems with subfolders "clean" and "invalid". Additional statistics and informations about the datasets are saved into the dataset directory.

    Cleaning configurations are meant to be passed by the cleaningConfig.py module.

    ---------
    Changes:
    0.0.1
        optimize runtime of calculateParticleProperties
        optimize drive usage, Load gzip files directly to memory
        fix image validation
    0.0.2
        optimize image validation
        
    ########################################
    Author: Elias Graf
    (c) Swisens AG
    Email Address: elias.graf@hslu.ch
    Nov 2019; Last revision: 11-Nov-2019
    ########################################
"""

__version__="0.0.2"




# Imports
import os, json, traceback, gzip, shutil, time
from zipfile import ZipFile, BadZipFile


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output

from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.io import imread

from scipy.signal import find_peaks

from io import BytesIO

import PIL
from PIL import Image

# Configs
import Swisens_CleaningConfigs as cnf
from Swisens_DataVisualisation import plotHist1, plotPositionScatter, displayStats


# JSON Validation
## collect all Keys in Json
def getKeys(json):
    keys = []
    for k in json.keys():
        keys.append(k)
        if type(json[k])==dict:
            k = getKeys(json[k])
            keys.extend(k)
    return keys

## check if all default keys in sample keys
def checkKeys(sample, default):
    sample = getKeys(sample) 
    default = getKeys(default)
    valid = True
    for k in default:
        if not k in sample: valid = False
    return valid
    
## validate json
def jsonValidation(json, json_default):
    """
    Json file validation.
    ----------------
    json : Json
        Target Json file.
    
    json_default : Json
        Default Json file.
    
        ----

    return : bool
        Validation result.
    """
    valid = checkKeys(json, json_default)
    return valid

# Trigger validation
def triggerValidation(sig):
    """
    Trigger signal validation (detector channel 0B).
    ----------------
    sig : list of int
        Trigger signal as list of int.
    
        ----
    return : bool
        Validation result.
    """
    peaks, properties = find_peaks(sig, width=cnf.peak_width,prominence=cnf.peak_prominence,distance=cnf.peak_distance)
    valid = False

    # drop small sidepeaks before second peak (occurs if peaks saturated), if in range of first/second trigger and below mean(peak_heigth)-10'000
    if len(peaks)>cnf.peak_number:
        heigth = np.array(sig)[peaks]
        condition = (heigth - heigth.mean()) > cnf.peak_minDeviation
        peaks = peaks[condition]

    # valid if peaks inside outer range |------[-^------^---]------|
    if (peaks>cnf.range_first[0]).all() and (peaks<cnf.range_second[1]).all() and len(peaks)==cnf.peak_number:
        valid = True
        
    # valid if peaks outside inner range   |------[-^--]---[--^--]----|
    if peaks[peaks>cnf.range_first[1]].any():
        p_inside = peaks[peaks>cnf.range_first[1]]
        if p_inside[p_inside<cnf.range_second[0]].any():
            valid = False    
            
    # invalid if left/right outer range+-offset is below lowLevel
    ind_startTrig = cnf.range_first[0] -cnf.lowLevel_indOffset
    ind_endTrig   = cnf.range_second[1]+cnf.lowLevel_indOffset
    
    lowLevel_start = np.mean(sig[0:ind_startTrig]) + cnf.lowLevelOffset_start
    lowLevel_end   = np.mean(sig[ind_endTrig:-1])  + cnf.lowLevelOffset_end
        
    if max(sig[0:ind_startTrig])>lowLevel_start or max(sig[ind_endTrig:-1])>lowLevel_end:
        valid = False
        #print('----^-----')
    
    if cnf.display:
        fig=plt.figure(figsize=(12,5))
        plt.plot(sig,'--k')
        c='or'
        if valid: c='og'
        plt.plot(peaks,sig[peaks],c)
        leg = 'invalid'
        if valid: leg = 'valid'
        plt.legend([leg])
        plt.ylim([0,67e3])

    return valid

# is in range
def inRange(val, range):
    """
    Check if val is in range.
    ----------------
    val: int,float,...
        Numeric value.
    range: Tuple
        Numeric range.

        ----
    
    return : bool
        Range result.
    """
    return ((val > range[0]).all() and (val <= range[1]).all())

# Holo image validation
def imgValidation(json):
    """
    Image properties validation.
    ----------------
    json : Json
        Target Json file.
        
        ----

    return : bool
        Validation result.
    """
    sol = np.array([json['holo0']['img_properties']['sol'], json['holo1']['img_properties']['sol']])
    area = np.array([json['holo0']['img_properties']['area'], json['holo1']['img_properties']['area']])
    minorAxis = np.array([json['holo0']['img_properties']['minorAxis'], json['holo1']['img_properties']['minorAxis']])
    majorAxis = np.array([json['holo0']['img_properties']['majorAxis'], json['holo1']['img_properties']['majorAxis']])
    perimeter = np.array([json['holo0']['img_properties']['perimeter'], json['holo1']['img_properties']['perimeter']])
    maxIntensity = np.array([json['holo0']['img_properties']['maxIntensity'], json['holo1']['img_properties']['maxIntensity']])
    minIntensity = np.array([json['holo0']['img_properties']['minIntensity'], json['holo1']['img_properties']['minIntensity']])
    meanIntensity = np.array([json['holo0']['img_properties']['meanIntensity'], json['holo1']['img_properties']['meanIntensity']])
    eccentricity = np.array([json['holo0']['img_properties']['eccentricity'], json['holo1']['img_properties']['eccentricity']])


    if not inRange(sol, cnf.sol_range):
        return False
        
    if not inRange(area, cnf.area_range):
        return False
        
    if not inRange(minorAxis, cnf.minorAxis_range):
        return False

    if not inRange(majorAxis, cnf.majorAxis_range):
        return False

    if not inRange(perimeter, cnf.perimeter_range):
        return False

    if not inRange(maxIntensity, cnf.maxIntensity_range):
        return False

    if not inRange(minIntensity, cnf.minIntensity_range):
        return False

    if not inRange(meanIntensity, cnf.meanIntensity_range):
        return False

    if not inRange(eccentricity, cnf.eccentricity_range):
        return False

    return True

# Image Properties
def calculateParticleProperties(img='none', zip_obj='none', name='none', holo='none'):
    """
    Image properties calculation.
    ----------------
    img : pil.Image, ='none'
        Image as pillow Image.
    
    zip_dir : str, ='none'
        Source zip directory as string.
    
    name : str, ='none'
        Json event filename.
    
    holo : int, ='none'
        Number of image (holo0, holo1).
    
        ----

    return :
        dict
            Image properties as dictionary.

        array
            Image as numpy array.

        str
            Image name as string.
    """

    if img=='none':
        if name.endswith('_event.json'):
            name=name[:-11]+'_rec'+str(holo)+'.png'
        try:
            # extract png
            if not name in zip_obj.namelist():
                return {}, np.array([]), name
            img = zip_obj.read(name)
            dataEnc = BytesIO(img)
            dataEnc.seek(0)

            with dataEnc as file:
                img = np.array(Image.open(file))
        except:
            print('error in extracting file. '+ name)
            traceback.print_exc()
            return
    else:
        try:
            with open(path,'r') as i:
                img = imread(i)
        except:
            print('error in extracting file: '+ name)
            traceback.print_exc()
            return

    # Img Properties           
    binImg = img < threshold_otsu(img)
    labImg = label(binImg)
    bigestCluster = np.argsort(np.bincount(labImg.flat))[-2] - 1
    regProp = regionprops(labImg, intensity_image=img)#, coordinates='rc')

    return {
         'sol': float(regProp[bigestCluster].solidity), 
         'area': int(regProp[bigestCluster].area),
         'minorAxis': float(regProp[bigestCluster].minor_axis_length),
         'majorAxis': float(regProp[bigestCluster].major_axis_length),
         'perimeter': float(regProp[bigestCluster].perimeter),
         'maxIntensity': int(regProp[bigestCluster].max_intensity),
         'minIntensity': int(regProp[bigestCluster].min_intensity),
         'meanIntensity': int(regProp[bigestCluster].mean_intensity),
         'eccentricity': float(regProp[bigestCluster].eccentricity),
     }, img, name.split("/")[-1]

# save file (json, img0, img1)
def saveTo(dst_path, f, eventJson, img0, img1, img_name0, img_name1):
    """
    Save sample files to directory.
    
    Parameters
    ----------
    dst_path : str
        Destination path.
        
    f : str
        File name.
        
    eventJson : dict
        Json file.
    
    img0 : array
        Holo0 image as array.
        
    img1 : array
        Holo1 image as array.
        
    img_name0 : str
        Holo0 image name.
        
    img_name1 : str
        Holo1 image name.
    
    """
    
    # save json
    with gzip.GzipFile(dst_path+'\\'+f.split("/")[-1]+'.gz','w') as fout:
        fout.write(json.dumps(eventJson).encode('utf-8'))
        
    ## save images
    img0=Image.fromarray(img0)
    img0.save(dst_path+'\\'+img_name0)
    img1=Image.fromarray(img1)
    img1.save(dst_path+'\\'+img_name1)


# Save particle statistics
def saveParticleStats(df_particleStats, dst_path,display=False):
    """
    Save Particle statistics plots if not displayed.
    
    Parameters
    ----------
    df_particleStats : DataFrame
        Particle statistics as DataFrame, one row per particle.
        
    dst_path : str, ='none'
        Path the plots is saved to.
    
    display : bool, =False
        If display is True the plots are only displayed and not saved.
    """
    ## plot particle stats
    plotPositionScatter(df_particleStats)
    fig = plt.gcf()
    fig.tight_layout()
    if not display:
        fig.savefig(dst_path+'\\holoPosition.png',dpi=300)
        plt.close(fig)
    ### histogram velocity
    plotHist1(df_particleStats,'cleaned data')
    fig = plt.gcf()
    fig.tight_layout()
    if not display:
        fig.savefig(dst_path+'\\velocityHist.png',dpi=300)
        plt.close(fig)
    ### histogram image properties
    pd.concat(list(df_particleStats['holo0|img_properties'])).drop(columns=[]).hist(sharey=True, histtype='bar',color='r', grid=False)
    fig = plt.gcf()
    fig.tight_layout()
    if not display:    
        fig.savefig(dst_path+'\\holo0ParticleStats.png',dpi=300)
        plt.close(fig)
    pd.concat(list(df_particleStats['holo1|img_properties'])).drop(columns=[]).hist(sharey=True, histtype='bar',color='b', grid=False)
    fig = plt.gcf()
    fig.tight_layout()
    if not display:
        fig.savefig(dst_path+'\\holo1ParticleStats.png',dpi=300)
        plt.close(fig)


# dataset cleaning
def cleanDataset(src_dir, dst_dir, n='all'):
    """"
    Extracting and cleaning of raw dataset directory.
    
    Parameters
    ----------
    src_dir : str, list
        Source directory.
        If passed as list, will iterate all directories.
        
    dst_dir : str
        Destination directory.
        
    n : int / str, ='all'
        Define the amount of files moved.

        ----

    return:
        DataFrame
            Cleaning statistics as DataFrame.

        DatFrame
            Image properties as DataFrame.

    """
    
    # cleaning stats
    df_stats = pd.DataFrame(columns=['total_cnt','valid_cnt','json_failCnt','holo_failCnt','imgProp_failCnt','trig_failCnt','others_failCnt'], index=[0])
    df_stats.iloc[0] = np.zeros(len(df_stats.columns))
    
    # particle stats
    df_particleStats = pd.DataFrame(columns=['timestamp','velocity','holo0|zf','holo1|zf','holo0|img_properties','holo1|img_properties'])
    
    # iterate source dirs if list
    if type(src_dir)==list:
        if len(src_dir)>1:
            for src_ in src_dir:
                temp1,temp2=cleanDataset(src_,dst_dir,n=n)
                df_stats+=temp1
                df_particleStats=df_particleStats.append(temp2)
            clear_output()
            print('Overview:')
            display(df_stats)
            displayStats(df_stats)
            return df_stats, df_particleStats
        elif len(src_dir)==1:
            src_dir = src_dir[0]
    
    # check if src_dir is zip directory
    if not src_dir.endswith('.zip'):
        print('Please pass .zip directory as src_dir.')
        return
    
    # check if dst_dir exists
    name = src_dir.split('\\')[-1].replace('.zip','')
    dst_path = dst_dir + '\\' + name
    # Experiment ID
    ID = ''
    if '_id' in name: ID = '_'+name.split('_')[2][2:len(name.split('_')[2])]
    if not os.path.exists(dst_path): os.makedirs(dst_path)
    if os.path.exists(dst_path+'\\'+'clean'+ID): shutil.rmtree(dst_path+'\\'+'clean'+ID)
    time.sleep(0.1)
    os.makedirs(dst_path+'\\'+'clean'+ID)
    if os.path.exists(dst_path+'\\'+'invalid'+ID): shutil.rmtree(dst_path+'\\'+'invalid'+ID)
    time.sleep(0.1)
    os.makedirs(dst_path+'\\'+'invalid'+ID)
    
    # open files
    try:
        with ZipFile(src_dir, 'r') as zipObj:
            # list json files
            json_files = [[f.filename,f.file_size] for f in zipObj.filelist if f.filename.endswith(cnf.ending)]
            
            # max files
            if n=='all': n = len(json_files)
            
            # iterate json filenames
            for f_i in tqdm(range(n)):
                f=''
                # stats
                df_stats['total_cnt']+=1
                valid = True

                # continue if empty
                if not json_files[f_i][1] > 0:
                    df_stats['others_failCnt']+=1
                    continue

                with zipObj.open(json_files[f_i][0]) as fzip:
                    with gzip.open(fzip,'rb') as jin:
                        eventJson = json.loads(jin.read().decode('utf-8'))
                        if json_files[f_i][0].endswith('.gz'): f=json_files[f_i][0][:-3]
            
    # cleaning validation
                ## json validation
                valid = valid and jsonValidation(eventJson, cnf.json_default)
                if not valid:
                    df_stats['json_failCnt']+=1
                    continue
                    
                ## get images and properties
                imgProp_0, img0, img_name0 = calculateParticleProperties(zip_obj=zipObj,name=f,holo=0)
                imgProp_1, img1, img_name1 = calculateParticleProperties(zip_obj=zipObj,name=f,holo=1)
                
                ## write img Properties and clean
                if cnf.write_imgProperties:
                    ### write
                    eventJson['holo0']['img_properties'] = imgProp_0
                    eventJson['holo1']['img_properties'] = imgProp_1
                    ### clean
                    valid = valid and imgValidation(eventJson)
                    if not valid:
                        df_stats['imgProp_failCnt']+=1
                        saveTo(dst_path+'\\'+'invalid'+ID, f.split("/")[-1], eventJson, img0, img1, img_name0, img_name1) 
                        continue
                
                ## trigger validation (truned off by default)
                if cnf.cleanTrigger:
                    valid = valid and triggerValidation(eventJson['adcdump']['0B'],lowLevel_indOffset=100)
                    if not valid:
                        df_stats['trig_failCnt']+=1
                        saveTo(dst_path+'\\'+'invalid'+ID, f.split("/")[-1], eventJson, img0, img1, img_name0, img_name1) 
                        continue
                
                # save file
                if valid:
                    saveTo(dst_path+'\\'+'clean'+ID, f, eventJson, img0, img1, img_name0, img_name1)                    
                    # cleaning stats
                    df_stats['valid_cnt']+=1
                    # particle stats ('timestamp','velocity','holo0|zf','holo1|zf','holo0|img_properties','holo1|img_properties')
                    temp = [eventJson['timestamp'], eventJson['velocity'],eventJson['holo0']['zf'],eventJson['holo1']['zf'],
                            pd.DataFrame.from_dict(imgProp_0,'index').T,pd.DataFrame.from_dict(imgProp_1,'index').T]
                    
                    df_particleStats = df_particleStats.append(pd.Series(temp,index=df_particleStats.columns), ignore_index=True)
    
    except BadZipFile as e:
        print(str(e)+' | in path: '+ dst_path)
    except:
        print('error in opening zip dir: '+dst_path)
        traceback.print_exc()
        return
    
    # Stats
    display(src_dir.replace('.zip','').split('\\')[-1])
    display(df_stats)
    with open(dst_path+'\\cleaning_stats.json', 'w') as f:
        json.dump(json.loads(df_stats.T.to_json()),f)
    
    ## plot piechart
    displayStats(df_stats,dst_path)
    ## save particle stats
    saveParticleStats(df_particleStats,dst_path)
    
    return df_stats, df_particleStats
