import numpy as np
import matplotlib.pyplot as plt
from .functions import * 
from .functions_mne import * 
from .ica import FastICA as fICA
from .sobi import *
#from coroica import UwedgeICA

import mne as mne
from mne.decoding import CSP

import scipy.io as sio
from scipy import signal

class Processing:
    def __init(self):
        pass
    
    @staticmethod
    def _power_band(X,fs=1.,bands=None):
        if bands != None:
            if len(X.shape) == 3:
                n_0 = X.shape[0]
                n_1= X.shape[1]
                n_bands = len(bands)
                X_bands_power=np.zeros((n_0,n_1,n_bands))
                #Calculate spectrum power along axis -1
                freqs , psd_fft = signal.welch(X,fs=fs)
                it=0
                for band in bands:
                    # Define lower and upper limits
                    low, high = band
                    # Find intersecting values in frequency vector
                    idx_delta = np.squeeze(np.logical_and(np.round(freqs) >= low, np.round(freqs) <= high))
                    #Calculate Power Spectrum 
                    df=fs/len(freqs)
                    X_bands_power[:,:,it] = np.sum(psd_fft[:,:,idx_delta],axis=2)*df
                    it+=1
                return X_bands_power
            else:
                print('No se ha implementado los casos para diferente tamaño de X')
        else:
            print('No se implemento el caso general de calculo de la potencia total de la señal')
    
    @staticmethod 
    def _whitening_data(X, n_epochs):
        X = np.hstack(X)
        X_w , W = whitening(X)
        return np.array(np.hsplit(X_w.copy(),n_epochs)) 
        
    @staticmethod 
    def _commun_spatial_pattern(X,Y, n_comp,sfreq=None, ch_names='',method='',savefig=False,path_file=""):
        if method == 'mne':
            csp = CSP(n_components = n_comp, reg='oas')
            csp.fit(X, Y)
            
            inf = create_info(ch_names.tolist(), len(ch_names),sfreq)
            fig = csp.plot_patterns(inf, ch_type='eeg', units='Patterns (AU)', size=1.5)
            if savefig:
                fig.set_size_inches((6,6))
                fig.savefig(path_file+'_CSPfilter.png')
            
            #Transform  CSP with all epochs per participant
            return csp.transform(X)
        else: 
            print('No se ha implementado todavia')
            return None
        
    @staticmethod 
    def _pca(X,n_comp,method=''):
        if method == 'pca': 
            pca = Pca()
            eig_values, X_pca = pca.fit_transform(X,n_comp)
            return X_pca, eig_values

class Preprocessing:
    def __init__(self):
        pass
    
    @staticmethod 
    def _filter(X,freq,sfreq,filter_type,order,iir=True):
        if iir:
            b,a=signal.butter(N=order, Wn=freq, btype=filter_type,fs=sfreq)
            padl=X.shape[2]-2
            if padl> 3*max(len(a),len(b)):
                padl=3*max(len(a),len(b))
            X = signal.filtfilt(b, a, X,axis=2,padlen=padl)
        else: 
            print("No implementado")
        return X

    @staticmethod 
    def _filter_notch(X,sfreq,f=50.,quality_factor=20.):
        b_notch, a_notch = signal.iirnotch(f, quality_factor, sfreq)
        X = signal.filtfilt(b_notch, a_notch, X,axis=2)
        return X
    
    @staticmethod 
    def _bad_channels(X,threshold=4.,savefig=False,path_file="",plot=True):
        #Find standar desviation
        ch_std=np.std(X,axis=1,dtype=float)
        #Detect bad channels z-scores iterative
        idx=find_outliers(ch_std,threshold=threshold,maxiter=10,tail=0)
        if plot:
            fig = plot_desv(np.abs(zscore(ch_std)), 'Dispersión de los canales',return_fig=True)
            if savefig:
                fig.set_size_inches((6,6))
                fig.savefig(path_file+'disp_canales.png')
        return idx.tolist()
    
    @staticmethod 
    def _z_score(X):
        """ X n_channels x n_frames*n_epochs """
        return standar_score(X)
    
    @staticmethod 
    def _bad_epochs(X,threshold=4.,savefig=False,path_file="",plot=True):
        #we find bad epochs computing mdcm
        ch_mean=np.mean(X,axis=(0,2))
        epochs_mean=np.mean(X,axis=2)
        epochs_MDCM=np.mean(np.abs(epochs_mean)-ch_mean,axis=1)
        #Detect bad channels z-scores iterative
        idx=find_outliers(epochs_MDCM,threshold=threshold,maxiter=10,tail=0)
        #Plot
        if plot:
            fig = plot_desv(np.abs(zscore(epochs_MDCM)), 'Dispersión de las epocas',return_fig=True)
            if savefig:
                fig.set_size_inches((6,6))
                fig.savefig(path_file+'disp_epocas.png')
        return idx.tolist()
    
    @staticmethod 
    def _remove_bad_epochs(X,idx_bd):
        return np.delete(X,idx_bd,axis=0)
    
    @staticmethod 
    def _remove_bad_trials(X,idx_bd):
        return np.delete(X,idx_bd,axis=1)
    
    @staticmethod 
    def _ica(X,ch_names,sfreq,nchans=None,method="SOBI"):
        if method == "FastICA-mne":
            A , U , W_pca , C_topo , S, ica_mne_inst = ica_mne(X, 
                                                          ch_names,
                                                          sfreq,
                                                          nchans)
            return A , U , W_pca , C_topo , S, ica_mne_inst
        elif method=="SOBI":
            print('No implementado')
        elif method == "FastICA":
            X_w , W = whitening(X)
            U_ , A = ica_descompotition(X_w,method='FastICA',random_seed=51)
            #Real Unmixing Matrix: 
            U = U_ @ W
            #Mixing Matrix
            A=np.linalg.pinv(U)
            #Sources
            S = U @ X
            return U, A, S
            
    @staticmethod     
    def _find_eog_artifact(data={},method='',threshold=4.):
        idx_eog , zscore_eog = None, None
        if method=='mne':
            X=data['X']
            name_channels=data['name_channels']
            sfreq=data['sfreq']
            cant_eog=data['cant_eog']
            X_mne=create_epochs(X,name_channels,sfreq, stdmontage="standard_1020",cant_eog=cant_eog)
            idx_eog , zscore_eog = icamne_find_eog(data['ICA_mne'],X_mne)
            return idx_eog, zscore_eog
        else:
            #X is (n_sources+n_eog_ref) x samples
            #in the last n_eog_ref rows we find eof ref
            EOG = data['EOG']
            X = data['X']
            sfreq=data['sfreq']
            #Filtering data
            EOG=filter_ica_data(EOG,sfreq,1.,10.)
            S=filter_ica_data(S,sfreq,1.,10.)
            
            n_eog_ref=EOG.shape[0]
            X=np.concatenate((X,EOG),axis=0)
            corrcoef_eog=np.corrcoef(X)[-n_eog_ref:,:-n_eog_ref]
            idx=[]
            for it in range(n_eog_ref):
                this_idx=find_outliers(corrcoef_eog[it],threshold=3.,maxiter=10,tail=0)
                idx += [this_idx]
            idx_ = np.concatenate(idx)
            idx_eog = list(np.unique(idx_))
            return idx_eog , zscore_eog   
    
    @staticmethod     
    def _find_muscle_artifact(data={},method='',threshold=4.):
        idx_emg , zscore_emg = None , None
        if method == "mne":
            idx_emg , zscore_emg = icamne_find_emg(data['ICA_mne'],data['X'],data['ch_names'],data['sfreq'],threshold=0.5848)
        else:
            print('METODO INCOMPLETO')
            
            # X=data['X']
            # sfreq=data['sfreq']
            # n_lag=data['n_lag']
            # # X is n_components x n_frames
            # n_components=X.shape[0]
            # n_frames=X.shape[1]
            # n_corr=n_frames
            # #Calculate power of signals
            # f, _=signal.welch(X[0,:], fs=sfreq)
            # PDD_total=np.zeros((n_components,len(f)))
            # for it in range(n_components):
            #     f, PDD_total[it,:]=signal.welch(X[it,:], fs=sfreq)
            # Plot_PSD(PDD_total,f)
            # HF=50.
            # IF=25.
            # PDD=np.sum(np.abs(PDD_total[:,f<HF]),axis=1)
            # PDD_IM=np.sum(np.abs(PDD_total[:,f<IF]),axis=1)
            # PDD_Noise=PDD-PDD_IM        
            # PDD_ratio=PDD_IM/PDD
            # PDD_zscore=Standar_Score(PDD_ratio)
            # idx_muscle=np.where(np.abs(PDD_zscore)>=threshold)
            # Plot_Desv(np.abs(PDD_zscore), 'Componentes con alta componente frecuencial')
        return idx_emg , zscore_emg
    
    @staticmethod     
    def _reconstruct_data(data={},method=''):
        X_ = None
        if method=='mne':
            components = data['components']
            X = data['X']
            ica_mne = data['ICA_mne']
            ch_names = data['ch_names']
            sfreq=data['sfreq']
            X_mne = create_epochs(X,ch_names,sfreq, stdmontage="standard_1020")
            X_mne = icamne_reconstruct_data(ica_mne,X_mne,components)    
            X_=X_mne.get_data()
        else: 
            print('NO SE HA IMPLEMENTADO EL METODO')
            
        return X_
   