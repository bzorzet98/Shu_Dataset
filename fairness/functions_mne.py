import mne as mne
from mne.io import RawArray
from mne.viz import *
from mne.preprocessing import (ICA,corrmap)
from mne.viz import plot_topomap as mne_plot_topomap
import numpy as np
import matplotlib.pyplot as plt
from .functions import *

def plot_eeg_epochs(X,ch_names,sfreq,stdmontage="standard_1020",savefig=False,path_fig=""):
    n_epochs=X.shape[0]
    n_ch=len(ch_names)
    epochs=create_epochs(X,ch_names,sfreq,stdmontage="standard_1020")
    if savefig:
        fig = plot_epochs(epochs, title="EEG",show_scrollbars=False,
                show_scalebars=False,theme="light")
        fig.savefig(path_fig+"EEG_epochs.png")
    else:
        epochs.plot(n_channels=32,n_epochs=20, title="EEG",theme="light",overview_mode=None)

def plot_eeg_sources(X,ch_names,sfreq,stdmontage="",savefig=False,path_fig=""):
    n_epochs=X.shape[0]
    n_ch=len(ch_names)
    epochs=create_epochs(X,ch_names,sfreq,stdmontage="")
    if savefig:
        fig = plot_epochs(epochs, title="ICA - Sources",show_scrollbars=False,
                show_scalebars=False,theme="light")
        fig.savefig(path_fig+"ICA_sources.png")
    else:
        plot_epochs(epochs,n_channels=32,n_epochs=20, title="ICA - Sources",theme="light")
    
def plot_eeg_epochs_psd(X,ch_names,sfreq,stdmontage="standard_1020",savefig=False,path_fig=""):
    n_epochs=X.shape[0]
    n_ch=len(ch_names)
    epochs=create_epochs(X,ch_names,sfreq,stdmontage="standard_1020")
    if savefig:
        fig = epochs.plot_psd()
        fig.set_size_inches((6,6))
        fig.savefig(path_fig+"Espectro_Potencias.png")
    else:
        epochs.plot_psd()
    
def plot_topomap(data, ch_names, sfreq,axes, vmin=None, vmax=None):
    nchans=len(ch_names)
    info=create_info(ch_names, nchans, sfreq)
    mne_plot_topomap(data, info, vmin=vmin, vmax=vmax,
                     show=False, sensors=True,axes=axes)    
    
def plot_ica_components(A,ch_names,sfreq,savefig=False,path_fig=""):
    n_chans=A.shape[0]
    n_components=A.shape[1]
    p=20
    n_figs=n_components // 20
    flag=False
    if n_components % 20 !=0:
         flag=True
    figs=[]
    nums=list(np.linspace(0,n_components,n_components,dtype=int))
    names=["ICA"+str(x) for x in nums]
    for itf in range(n_figs):
        fig, axs = plt.subplots(4, 5)
        [axi.set_axis_off() for axi in axs.ravel()]
        for it1 in range(4):
            for it2 in range(5):
                it3=it1*(5)+it2
                plot_topomap(A[:,it3],ch_names, sfreq,axes=axs[it1,it2], vmin=None, vmax=None)    
                axs[it1,it2].set_title(names[it3])
        plt.show()
        figs.append(fig)
    
    fig, axs = plt.subplots(4, 5)
    [axi.set_axis_off() for axi in axs.ravel()]
    it3=n_figs*(4*5)
    flag=False
    for it1 in range(4):
        for it2 in range(5):
            it3_=it3+it1*5+it2
            if it3_ == n_components:
                flag=True
                break
            plot_topomap(A[:,it3_],ch_names, sfreq,axes=axs[it1,it2], vmin=None, vmax=None)
            axs[it1,it2].set_title(names[it3_])   
        if flag:
            break
    plt.show()
    figs.append(fig)
    if savefig:
        it=0
        for it in range(len(figs)):
            figs[it].set_size_inches((6,6))
            figs[it].savefig(path_fig+"ICA_comp_"+str(it)+".png")
    return figs
    
def create_raws(X,ch_names,sfreq,stdmontage="standard_1020"):
    n_channels=len(ch_names)
    microV=1.e-6
    info=create_info(ch_names,n_channels,sfreq,stdmontage)
    #raw=RawArray(X*microV, info)
    raw=RawArray(X, info)
    return raw
    
def create_epochs(X,ch_names,sfreq,stdmontage="standard_1020",cant_eog=0.):
    n_channels=len(ch_names)
    microV=1.e-6
    info=create_info(ch_names,n_channels,sfreq,stdmontage,cant_eog=cant_eog)
    #epochs=mne.EpochsArray(X*microV, info)
    epochs=mne.EpochsArray(X, info)
    return epochs
    
def create_info(ch_names,n_chan,sfreq=12,stdmontage="standard_1020",cant_eog=0.):
        if cant_eog>0:
            info= mne.create_info(
                    ch_names=ch_names,
                    ch_types=['eeg']*(n_chan-cant_eog)+cant_eog*['eog'],
                    sfreq=sfreq)
        else:
            info= mne.create_info(
                    ch_names=ch_names,
                    ch_types=['eeg']*(n_chan),
                    sfreq=sfreq)
        if stdmontage!="":
            montage = mne.channels.make_standard_montage(stdmontage)
            info.set_montage(montage)
        return info

def filter_ica_data(X,sfreq,l_freq,h_freq):
    kw = dict(phase='zero-double', filter_length='10s', fir_window='hann',
                  l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                  fir_design='firwin2')
    X_=mne.filter.filter_data(X, sfreq, l_freq, h_freq,**kw)
    return X_
    

def ica_mne(X,channels,sfreq,n_components,n_eog=0,random_seed=51):
    X_mne=create_epochs(X,channels,sfreq, stdmontage="standard_1020",cant_eog=n_eog)
    #Ploteamos las epocas
    #plot_epochs(X_mne,picks='all')
    ICA_mne=ICA(n_components, max_iter='auto',random_state=random_seed)
    ICA_mne.fit(X_mne,picks=['eeg'])    
    
    #Obtain components TOPOMAP
    C_topo=ICA_mne.get_components()
    #Obtain sources
    S=ICA_mne.get_sources(X_mne)
    S=S.get_data()
    #Obtain MIXING, UNMIXING, AND PCA MATRIX    
    A=ICA_mne.mixing_matrix_
    W_pca=ICA_mne.pca_components_
    U=ICA_mne.unmixing_matrix_
    return A , U , W_pca , C_topo , S, ICA_mne
    
def icamne_find_eog(ICA,X):
    labels, scores = ICA.find_bads_eog(X)
    return labels, scores 

def icamne_find_emg(ICA,X,ch_names,sfreq,threshold=0.5848):
    raw=create_raws(X,ch_names,sfreq,stdmontage="standard_1020")
    labels, scores = ICA.find_bads_muscle(raw,threshold=threshold)
    return labels, scores 

def icamne_reconstruct_data(ICA,X,components):
    #X is mne object
    X_=ICA.apply(X,exclude=components)
    return X_
    