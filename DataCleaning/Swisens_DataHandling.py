"""
    Swisens Data Handling module
    
    ########################################
    
    This module is applicable to measurement data of the Swisens Poleno.
    - Samples are JSON event files with the corresponding holo images (rec0, rec1).
    - Classes are directories containing samples.
    - Datasets are a set of classes.

    ####### UNDER CONSTRUCTION ########
    # This module is in an early development stage.
    # Therefore, incomplete or faulty methods/classes may be present.
    ########################################

    ---------
    Changes:
    0.0.0
        
        
    ########################################
    Author: Elias Graf
    (c) Swisens AG
    Email Address: elias.graf@hslu.ch
    Nov 2019; Last revision: 11-Nov-2019
    ########################################
"""

__version__="0.0.0"



# Imports
import os, gzip, json, traceback
import ipywidgets as widgets
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output

from Swisens_DataCleaning import triggerValidation
from Swisens_DataVisualisation import plotHist

###############################################################################
## Temporary variable declaration
###############################################################################
# metadata to read
metadata_names = ['timestamp','valid','trig_tdiff','trigger_thresh_adc','velocity']
label_names = ['date','label','loc','device','iteration']

# polarisation
pol_names = ['pol_0_adc','pol_1_adc']

# additional data
pos_names = [['holo0','holo1'],['xy','zr','zf'],
             ['sol','area','minorAxis','majorAxis','perimeter','maxIntensity','minIntensity','meanIntensity','eccentricity']]

# FL wavelength windows
windows = [
    "0A",
    "0B",
    "1A",
    "1B",
    "2A",
    "2B"  
]

# samplerate
b = 10
fs = 250e6/2**b
ts = 1/fs

# setuptime
t_setup = 100e-6
setup_samples = int(t_setup/ts)

# time offset measurebegin
t_offset = 220e-6
###############################################################################


# checkboxes
def chooseDirs(method, kwargs, src_path, ending='.zip', add_id=False, ret=[]):
    """
    Choose directories to load with checkboxes.
    ------------
    method : method
        Method to call on button click.
    
    kwargs : dict
        Keyword arguments of method.
        Choosen directory paths will be set as first keyword argument. Therefore the first kwarg must be the source path/directory argument for the method.
        
    src_path : str
        Directories will be listed from src_path.
    
    ending : str, ='.zip'
        Ending of directories to list.

    add_id : bool, =False
        ClusterSorting searches for classes with name "clean" and ID-number as suffix.

    ret : list, =[]
        Return value of method will be pased through ret argument.
    
    """
    def listUpdate(b):
        val = b['owner'].description
        if b['new']==True:
            dir_choice.append(val)
        elif b['new']==False:
            dir_choice.remove(val)

    print('Choose directories to load:')    
    dir_list = [p for p in os.listdir(src_path) if p.endswith(ending)]
    dir_choice=[]
    box_list = []
    for d_i in range(len(dir_list)):
        box_list.append(
            widgets.Checkbox(
                value=False,
                description=dir_list[d_i],
                disabled=False,
                layout={'width': '500px'}))
        box_list[d_i].observe(listUpdate)
        display(box_list[d_i])

    # button
    button = widgets.Button(
        description='OK',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='',
        icon=''
    )
    display(button)
    
    def clicked(change):
        if len(dir_choice)<1:
            return
        elif len(dir_choice)>=1:
            src_dir = [src_path+"\\"+d for d in dir_choice]
        clear_output()
        kwargs[list(kwargs.keys())[0]] = src_dir
        if add_id: kwargs['specified_folders'] = ['clean'+'_'+name.split('_')[2][2:len(name.split('_')[2])] for name in dir_choice]
        ret.append(method(**kwargs))
        
    button.on_click(clicked)



# save class samples in object
class samplesObj:
    def __init__(self):
        self.metadata = pd.DataFrame()
        self.holo0 = pd.DataFrame()
        self.holo1 = pd.DataFrame()
        self.adcdump = []
        self.adcdumpStats = []
        self.sipm = []

    def name(self):
        return self.metadata['label'][0]
    
    def location(self):
        return self.metadata['loc'][0]
    
    def device(self):
        return self.metadata['device'][0]
    
    def iteration(self):
        return self.metadata['iteration'][0]

    def ident_str(self):
        return self.name() + ' | ' + self.location() + '-' + self.device() + '-' + str(self.iteration())
    
## ADCDUMP methodes
    
    # recall time series of all samples of a specific channel
    def getTimeSeries(self,win:str):
        df_TimeSeries = pd.DataFrame()
        for df in self.adcdump:
            #print(df[win])
            df_TimeSeries = df_TimeSeries.append(pd.Series(df[win]), ignore_index=True)
        return df_TimeSeries.T

    # recall time series of random sample of a specific channel
    def randTimeSeries(self,win):
        df_TimeSeries = pd.DataFrame()
        df_TimeSeries = df_TimeSeries.append(pd.Series(df[win]), ignore_index=True)
        return df_TimeSeries    
    
    # plot statistics of one channel (all samples) as Time Series
    def plotStatsTimeSeries(self,win:str,stats:str):
        timeSeries = self.getStatsTimeSeries(win)[stats]
        plt.plot(timeSeries)
        plt.title(str(self.metadata['label'][0] + ' | ' + win + ' | ' + stats))
        plt.xlabel('time')
        plt.ylabel('fl. int.')
        plt.grid()
        
    # get statistics of one channel (all samples) as Time Series
    def getStatsTimeSeries(self,win:str):
        # statistics calculations are done at first execution
        
        if not self.adcdumpStats:
            for i in tqdm(range(len(self.adcdump[0].columns))):
                self.adcdumpStats.insert(i,self.getTimeSeries(self.adcdump[0].columns[i]).T.describe())
            clear_output()
        index = [i for i,x in enumerate(self.adcdump[0].columns==win) if x][0]
        #print(index)
        return self.adcdumpStats[index].T
    
## SIPM methodes
    # get sipm values of specific sample by value name
    def getSipmVal(self, val, sample=0,flat=False,exclude_config=[],exclude_win=[],name='none',plot=False):
        sipm = self.sipm[sample]
        df_sipm = pd.melt(sipm.reset_index(),value_vars=['0A','1A','1B','2A','2B'],id_vars='index')
        phase = df_sipm[df_sipm['variable_1']==val].drop(columns='variable_1')#.reset_index().drop(columns='index')
        # exclude configuration section
        for num in exclude_config:
            phase = phase[phase['index']!=num]
        phase = phase.pivot(index='index',
                            columns='variable_0',
                            values='value')
        # exclude receiver windonw
        phase = phase.drop(columns=exclude_win)
        
        if flat:
            phase = phase.melt()['value'].values
        return phase
    
    # get grouped sipm list
    def getSipmGrouped(self,showMean=False):
        ret = pd.concat(self.sipm).groupby(level=0)
        if showMean:
            ret = ret.mean()
        return ret


# Search for specific samples
def searchForElement(data, meta, element, by='label'):
    meta = copy.deepcopy(meta)
    ind = meta.index[meta[by]==element].tolist()
    meta = meta.loc[ind].reset_index().drop(columns=['index'])
    return [data[i] for i in ind], meta, ind

def getInd(data, meta, element, by='label'):
    d, m, ind = searchForElement(data,meta,element,by=by)
    return ind

def getMeta(data, meta, element, by='label'):
    d, m, ind = searchForElement(data,meta,element,by=by)
    return m

def getData(data, meta, element, by='label'):
    d, m, ind = searchForElement(data,meta,element,by=by)
    return d



# Read samples of one class
def readClass(path, max_samples, random=True, trigValid=True, background=False, specified_folders=[]):
        exc_configs = []
        if type(path)==list: path=path[0]
        if type(specified_folders)==list: specified_folders=specified_folders[0]
        # Dir name as Dataset Info
        meta_info = os.path.basename(path).split('_')
        if not specified_folders=='': path+="\\"+specified_folders
            
        # just use json file list
        files = sorted(os.listdir(path))
        json_files = []
        for f in files:
            if not (f.endswith("event.json") or f.endswith("event.json.gz")): continue
            json_files.append(f)
            
        if max_samples<0: max_samples=len(json_files)
            
        # randomize file list
        np.random.shuffle(json_files)

        # metadata
        meta_temp = []
        meta_temp = label_names
        #meta_temp.extend(metadata_names)
        meta_temp = ['date', 'label', 'loc', 'device', 'iteration', 'timestamp', 'valid', 'trig_tdiff', 'trigger_thresh_adc', 'velocity','file']
        metadata_names = ['timestamp','valid','trig_tdiff','trigger_thresh_adc','velocity']
        if background:
            meta_temp = ['date', 'label', 'loc', 'device', 'iteration', 'timestamp', 'valid', 'trig_tdiff', 'velocity','file']
            metadata_names = ['timestamp','valid','trig_tdiff','velocity']
            
        metadata = pd.DataFrame(columns=meta_temp)
        metadata.index.name = 'sample'
        
        # polarisation
        pol_adc = []
        
        # position data (holo position)
        cols = ['x','y','zr','zf']
        cols.extend(pos_names[2])
        holo0 = pd.DataFrame(columns=cols)
        holo1 = pd.DataFrame(columns=cols)
        
        # sipm DataFrame
        col1 = []
        for w in windows:
            col1.extend(4*[w])
        col2 = ['of_hits','avg','corr_mag','corr_pha']
        col2 += 5*col2
        sipm_cols = [np.array(['excitation','excitation','excitation','excitation']+col1),
                     np.array(['f_exc','corr_interval','sources','wavelength']+col2)]
        # simp list
        sipm = []

        # samples list
        adcdump = [] 
        
        # file stats
        filestats = pd.DataFrame(columns=['total_cnt','valid_cnt','trig_failCnt'],index=[0])
        filestats.loc[0] = np.zeros(len(filestats.columns))
        
        # read files
        for f in tqdm(json_files,total=max_samples):
            
        # check if break criteria reached
            if filestats['valid_cnt'].values>=max_samples: break
            
            try:
                jin = []
                if f.endswith('.gz'):
                    jin = gzip.open(path+'\\'+f,'rb').read().decode('utf-8')
                else:
                    jin = open(path+'\\'+f, 'r').read()
                eventJson = json.loads(jin)
                
            except:
                print("error in file: " + path + f)
                traceback.print_exc()
                break
                
            filestats['total_cnt'] += 1
                
        # trigger validation
            if trigValid:
                if not triggerValidation(eventJson['adcdump']['0B']):
                    filestats['trig_failCnt'] += 1
                    continue
             
            filestats['valid_cnt'] += 1
            
        # read metadata
            meta_temp = []
            meta_temp.extend(meta_info)
            for m in metadata_names:
                m_val = eventJson[m]
                meta_temp.append(m_val)
            meta_temp.append(f)
            metadata = metadata.append(pd.Series(meta_temp,index=metadata.columns), ignore_index=True)
            
        # read polarisation data
            df_pol = pd.DataFrame(columns=pol_names)
            #df_pol = pd.DataFrame.from_dict(eventJson[pol_names[0]], orient='index')         --> for Time Series
            #df_pol = pd.DataFrame.from_dict(eventJson[pol_names[1]], orient='index')         --> for Time Series
            df_pol.loc[0] = [eventJson['pol_0_adc'], eventJson['pol_1_adc']]
            pol_adc.append(df_pol)
            
        # read position data (holo)
            pos_temp0 = []
            pos_temp1 = []
            if not background:
                for a in pos_names[1]:
                    # append x and y individually
                    if a == 'xy':
                        pos_temp0.append(eventJson['holo0'][a][0])
                        pos_temp0.append(eventJson['holo0'][a][1])
                        pos_temp1.append(eventJson['holo1'][a][0])
                        pos_temp1.append(eventJson['holo1'][a][1])
                    else:   
                        pos_temp0.append(eventJson['holo0'][a])
                        pos_temp1.append(eventJson['holo1'][a])

                for prop in pos_names[2]:
                    pos_temp0.append(eventJson['holo0']['img_properties'][prop])
                    pos_temp1.append(eventJson['holo1']['img_properties'][prop])
            else:
                pos_temp0 = np.zeros(13)
                pos_temp1 = pos_temp0
                    
            holo0 = holo0.append(pd.Series(pos_temp0, index=holo0.columns),ignore_index=True)
            holo1 = holo1.append(pd.Series(pos_temp1, index=holo1.columns),ignore_index=True)
        
        # read sipm data
            df_sipm = pd.DataFrame(columns=sipm_cols)
            for corr_interval in eventJson['sipm_data']:
                interval_list = []
                interval_list.append(corr_interval['f_exc'])
                interval_list.append(corr_interval['corr_interval'])
                src, wl = corr_interval['sources'][0].split(' ')
                wl, _ = wl.split('nm')
                interval_list.append(src)
                interval_list.append(wl)
                
                for corr_channel in corr_interval['corr_channels']:
                    interval_list.append(corr_channel['of_hits'])
                    interval_list.append(corr_channel['avg'])
                    interval_list.append(corr_channel['corr_mag'])
                    interval_list.append(corr_channel['corr_pha'])

                df_sipm = df_sipm.append(pd.Series(interval_list,index=sipm_cols), ignore_index=True)
                
            # add current sipm df to list
            sipm.append(df_sipm)
            
            lastIndex = -1 # load all
            
        # read adcdump: intensity values
            if len(exc_configs)<1: exc_configs = df_sipm['excitation']['corr_interval'].values
            ind_start = 0
            ind_end = int(np.cumsum(exc_configs+t_setup)[-1]/ts)

            df_temp = pd.DataFrame.from_dict(eventJson['adcdump'], orient='index')
            df_adcdump = df_temp.T[ind_start:ind_end]
            df_adcdump = df_adcdump.reset_index()
            df_adcdump = df_adcdump.drop(columns=['index'])
            df_adcdump.name = eventJson['timestamp']           
                
            # add time axis
            n = df_adcdump.index.stop-1
            t_axis = np.arange(0,n*ts+1e-10,ts)
            df_adcdump.index = pd.Series(t_axis)
            df_adcdump.index.name = 'time'
            
            # add current sample to list
            adcdump.append(df_adcdump)            
        
        if metadata.empty:
            # read metadata
            meta_temp = []
            meta_temp.extend(meta_info)
            for m in metadata_names:            
                meta_temp.append('none') 
            meta_temp.append('none')

            metadata = metadata.append(pd.Series(meta_temp,index=metadata.columns), ignore_index=True)
            
        # Create object
        return_obj = samplesObj()
        return_obj.metadata  = metadata
        return_obj.filestats = filestats
        return_obj.holo0     = holo0
        return_obj.holo1     = holo1
        return_obj.pol_adc   = pol_adc
        return_obj.sipm      = sipm
        return_obj.adcdump   = adcdump
        display(filestats)
        #print("json files total:    " + str(len(json_files)))        
        #print("json files complete: " + str(complete_files_count))
        
        return return_obj, metadata.iloc[0,0:5]


# read Dataset
def readDataset(paths, max_samples=-1, random=True, display_stats=True,trigValid=True, background=False, specified_folders=[]):
    dataset = []

    metadata = pd.DataFrame(columns=label_names)
    filestats = pd.DataFrame(columns=['total_cnt','valid_cnt','trig_failCnt'])
    count = 0
    print('Reading Dataset...')
    for path in tqdm(paths):
        # append to dataset/metadata/filestats list
        class_obj, meta = readClass(path, max_samples,random=random,
                                    trigValid=trigValid, background=background,
                                    specified_folders=specified_folders[paths.index(path)])
        dataset.append(class_obj)
        metadata = metadata.append(meta, ignore_index=True)
        filestats = filestats.append(class_obj.filestats, ignore_index=True)
        count += 1
        
    print(str('Dataset contains ' + str(count) + ' classes'))

    # check for differing configurations between dataset
    configs = checkConfigs(dataset,metadata)
    metadata = metadata.join(filestats)
    
    if display_stats:
        clear_output()
        plotHist(metadata)
        
    return dataset, metadata, configs

# check for differing configurations
def checkConfigs(data,meta):
    configs = []
    meta['config'] = 0
    configs.append(data[0].sipm[0]['excitation'])
    
    for i_d in range(len(data)-1):
        # get conifgs uf current class
        c2 = data[i_d+1].sipm[0]['excitation']
        
        new_config = True
        for i_c in range(len(configs)):
            # check if config different/new
            if (configs[i_c]==c2).all().all(): new_config=False
                
        if new_config: configs.append(c2)
        for i_c in range(len(configs)):        
            if (configs[i_c]==c2).all().all():
                meta.loc[i_d,'config'] = i_c
    return configs  
