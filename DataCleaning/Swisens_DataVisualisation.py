"""
    Swisens Data Visualisation module
    
    ########################################
    
    This module is applicable to measurement data of the Swisens Poleno.
    Raw datasets are meant to be in the form of zipped directories containing zipped Json event files and png Holoimage files.
    The cleaned datasets are saved as filesystems with subfolders "clean" and "invalid". Additional statistics and informations about the datasets are saved into the dataset directory.

    ####### UNDER CONSTRUCTION ########
    # This module is in an early development stage.
    # Therefore, incomplete or faulty methods/classes may be present.
    ########################################

    ---------
    Changes:
    0.0.
        
    ########################################
    Author: Elias Graf
    (c) Swisens AG
    Email Address: elias.graf@hslu.ch
    Nov 2019; Last revision: 12-Nov-2019
    ########################################
"""

__version__="0.0.0"


# Imports
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import Swisens_CleaningConfigs as cnf


###############################################################################
## Temporary variable declaration
###############################################################################
# FL wavelength windows
windows = [
    "0A",
    "0B",
    "1A",
    "1B",
    "2A",
    "2B"  
]

win_wl = {
        '0A':'330-380 nm',
        '0B':'trigger',
        '1A':'415-460 nm',
        '1B':'465-500 nm',
        '2A':'540-585 nm',
        '2B':'660-695 nm'}
###############################################################################



# plot Histogram
def plotHist(data,x_label='label',y1_label='total_cnt',y2_label='trig_failCnt',y3_label='valid_cnt',figsize=(10,7)):
    # changes metadata arrays from object to int
    data[y1_label]=data[y1_label].astype(int)
    data[y2_label]=data[y2_label].astype(int)
    data[y3_label]=data[y3_label].astype(int)
        
    meta = data.set_index(x_label).groupby([x_label]).sum()
    x = meta.index.values
    y1 = meta[y1_label]
    y2 = meta[y1_label].values-meta[y2_label].values
    y3 = meta[y3_label].values
    perc = y3/y1*100
    
    
    sns.set(rc={'figure.figsize':figsize},style='whitegrid')
    ax1 = sns.barplot(x = x, y = y1, color='lightgrey')
    ax2 = sns.barplot(x = x, y = y2, color='red')
    ax3 = sns.barplot(x = x, y = y3)
    ax3.set_ylabel('Samples', fontsize = 20)
    ax3.set_xlabel(x_label, fontsize = 20)
    ax3.set_title('Dataset' , fontsize = 30)
    ax3.legend([y1_label,y2_label,y3_label])
    ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)
    
    n = int(len(ax3.patches)/3)
    i = 0
    for p in ax1.patches[-n:3*n]:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height() - np.round(0.05*np.max(y1))
        value = str('%s %%\n' %np.round(perc[i],1))
        ax1.text(_x, _y, value, va="center", color='k', fontsize=10,rotation='vertical')  
        i+=1
    

# Histogram plot
def plotHist1(data,title, label='velocity',unit='[m/s]',figsize=(10,8)):
    """
    Plot particle positions as scatter with marker color according to the particle velocity.
    
    Parameters
    ----------
    data : DataFrame or samplesObj
        Data as DataFrame or in samplesObj.* as DataFrame.
    
    title : str
        Figure title.
    
    label : str, ='velocity'
        Data label.

    unit : str, ='[m/s]'
        Data unit.

    figsize : Tuble of int, =(10,8)
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if type(data)==pd.DataFrame:
        x=data[label]
    else:
        x=data.metadata[label]
    sns.distplot(x, ax=ax,  kde=False)
    ax.set_title(title+' - '+label, fontsize = 20)
    ax.set_xlabel(label+' '+unit,fontsize=15)



# Holo Position Scatter plot
def plotPositionScatter(data):
    """
    Plot particle positions as scatter with marker color according to the particle velocity.
    
    Parameters
    ----------
    data : DataFrame or samplesObj
        Particle properties as DataFrame or in samplesObj.holo DataFrame.
    
    figsize : Tuble of int, =(10,8)
        Figure size.
    """

    class_data = data
    #fig, ax = plt.subplots(figsize=fig)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_title('holo position to velocity',fontsize=15)
    if type(data)==pd.DataFrame:
        x=data['holo0|zf'].astype(float)
        y=data['holo1|zf'].astype(float)
        c=data['velocity'].astype(float)
    else:
        x=class_data.holo0['zf']
        y=class_data.holo1['zf']
        c=class_data.metadata['velocity']
    sctr = plt.scatter(  x=x+cnf.cam_offset,
                         y=y+cnf.cam_offset,
                         c=c, marker='o', cmap='coolwarm')
    cbar=plt.colorbar(sctr, ax=ax, format='%.2f m/s')
    cbar.set_label('Velocity',fontsize=15)

    lims = [-1e-3,1e-3]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('holo0 [mm]',fontsize=15)
    ax.set_ylabel('holo1 [mm]',fontsize=15)
    ax.set_xticklabels(ax.get_xticks()*1000)
    ax.set_yticklabels(ax.get_yticks()*1000)
    ax.legend(['camera focus $zf$', 'camera axis $x$'])
    

    # Display cleaning statistics
def displayStats(df_stats, dst_path='none'):
    """
    Display cleaning statistics as pie chart.
    
    Parameters
    ----------
    df_stats : DataFrame
        Cleaning statistics as DataFrame with one row.
        
    dst_path : str, ='none'
        Path the pie chart is saved to if not 'none'.
    
    """
    ## Pie chart
    labels=[]
    values=[]
    explode = [0.1]
    colors=['#01a14b','#d00002','#f20002','#ff5302','#ff7602','#ff8300']
    for c in df_stats.drop(columns=['total_cnt']):
        val = df_stats[c].values[0]
        if val>0:
            labels.append(c)
            values.append(val)
            explode.append(0.02)
        else:
            colors.pop(-1)
    explode.pop(-1)
    fig, ax = plt.subplots()
    ax.pie(values, explode=explode, colors=colors,
            labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ### Equal aspect ratio
    ax.axis('equal')
    plt.tight_layout()
    plt.show()
    
    if dst_path!='none': fig.savefig(dst_path+'\\cleaning_overview.png',dpi=300)


# Fluorescence: ADCDUMP
def adcdumpPlot(data, title, figsize=(15,10), wins=[], xnth=1,ylim=[7e3,65e3], sample_n=0):

    drop=[w for w in windows if not w in wins]
    
    # melt data
    sample=data.adcdump[sample_n]
    df_melt = sample.reset_index().drop(columns=drop).loc[::xnth].melt(id_vars=['time'])
    
    g=sns.lineplot(x='time', y='value',
                hue='variable',
                data=df_melt)
    ax=plt.gca()
    plt.grid()
    # x Axis
    plt.xticks(fontsize=15)  
    plt.xlabel('Time [s]',fontsize=20)
    # y Axis
    plt.yticks(fontsize=17)  
    ylabel = 'FL Intensity [a.u.]'
    if not '0B' in drop: ylabel='Intensity [a.u.]'
    plt.ylabel(ylabel,fontsize=20) 
    plt.ylim(ylim)
    # legend
    legend = [w+' | '+win_wl[w] for w in windows if not w in drop]
    legend.insert(0,'Detector Windows')
    legend_text = ax.legend().texts
    for t,l in zip(legend_text,legend): t.set_text(l)

    plt.title(title+' - ('+data.ident_str()+')',fontsize=22)
    plt.grid()