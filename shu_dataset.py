""" Base de datos de Shu: 
A large EEG dataset for studying cross-session 
variability in motor imagery brain-computer interface
https://figshare.com/articles/software/shu_dataset/19228725/1
12 Participantes
13 canales de EEG
"""
import mne as mne
from mne.decoding import CSP

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold , train_test_split

from fairness.eeg import Preprocessing, Processing
from fairness.functions import *
from fairness.functions_mne import *
from fairness.sobi import*



class RegistersShuDataset(Processing):
    def __init__(self):
        self.registers = {}
        self.participants = None
        self.sessions = None
        self.genders = []
        self.NM = 0
        self.NF = 0
        self.train_participants = None
        self.validation_participants = None
        self.data_train = None
        self.data_validation = None
        self.csp=None
             
    def load_registers(self, path='', participants=[], sessions=[]):
        if path == "":
            print('No se ha seleccionado una dirección')
        else:
            self.participants = participants
            self.sessions = sessions
            for participant in participants:
                flag_participant=True
                self.registers[participant] = {}
                for session in sessions: 
                    database_path= path + participant +"/"
                    data_path=participant+"_"+session+"_task_motorimagery_eeg_preprocessing.mat"
                    data=sio.loadmat(database_path + data_path)
                    self.registers[participant][session]= {}
                    self.registers[participant][session]['data'] = data['data']
                    self.registers[participant][session]['labels_trials'] = np.squeeze(data['labels_trials'])
                    if flag_participant:
                        self.registers[participant]['group_medidator'] = data['group_medidator']
                        self.registers[participant]['gender'] = data['gender']
                        self.genders.append(data['gender'])
                        if data['gender'] == 'M' : 
                            self.NM += 1
                        else:
                            self.NF += 1
                        self.registers[participant]['id_participant'] = data['id_participant']
                        self.registers[participant]['sfreq'] = data['sfreq']
                        ch_names = data['ch_names']
                        it=0
                        for ch in data['ch_names']:
                            if ' ' in ch:
                                ch_names[it]=ch[:-1]
                            it+=1
                        self.registers[participant]['ch_names']=ch_names 
                        self.registers[participant]['age'] = data['age']
                        self.registers[participant]['n_epochs'] = data['data'].shape[0]
                        self.registers[participant]['n_samples'] = data['data'].shape[2]
                        self.registers[participant]['n_chans'] = data['data'].shape[1]
                        flag_participant = False
    
    def split_database(self,percent_train, percent_validation, percent_test = None):
        if percent_test == None: 
            self.train_participants, self.validation_participants = train_test_split(np.array(self.participants),train_size= percent_train,
                                                         test_size= percent_test,
                                                         stratify= np.array(self.genders))
        
    def group_data(self):
        self.data_train = np.zeros((1,32,1000))
        self.labels_train = np.zeros((1))
        self.data_validation = np.zeros((1,32,1000))
        self.labels_validation = np.zeros((1))
        for train_participant in self.train_participants:
            self.data_train = np.concatenate((self.data_train,
                                              self.registers[train_participant]["ses-01"]['data'],
                                              self.registers[train_participant]["ses-02"]['data'],
                                              self.registers[train_participant]["ses-03"]['data'],
                                              self.registers[train_participant]["ses-04"]['data'],
                                              self.registers[train_participant]["ses-05"]['data']),axis=0)     
            self.labels_train =np.concatenate((self.labels_train,
                                               self.registers[train_participant]["ses-01"]['labels_trials'],
                                               self.registers[train_participant]["ses-02"]['labels_trials'],
                                               self.registers[train_participant]["ses-03"]['labels_trials'],
                                               self.registers[train_participant]["ses-04"]['labels_trials'],
                                               self.registers[train_participant]["ses-05"]['labels_trials']))  
        self.data_train = self.data_train[1:,:,:]
        self.labels_train = self.labels_train[1:]
        
        
    def csp_per_subject(self,nc=6,savefig=False,path=''):
        data_ = np.zeros((1,32,1000))
        labels_ = np.zeros(1)
        for participant in self.participants:
            csp = CSP(n_components = nc, reg='oas', norm_trace = True)
            data_= np.concatenate((data_,
                                    self.registers[participant]["ses-01"]['data'],
                                    self.registers[participant]["ses-02"]['data'],
                                    self.registers[participant]["ses-03"]['data'],
                                    self.registers[participant]["ses-04"]['data'],
                                    self.registers[participant]["ses-05"]['data']),axis=0)     
            labels_ =np.concatenate((labels_,
                                    self.registers[participant]["ses-01"]['labels_trials'],
                                    self.registers[participant]["ses-02"]['labels_trials'],
                                    self.registers[participant]["ses-03"]['labels_trials'],
                                    self.registers[participant]["ses-04"]['labels_trials'],
                                    self.registers[participant]["ses-05"]['labels_trials']))  
            data_ = data_[1:,:,:]
            labels_ = labels_[1:]
            csp.fit(data_, labels_)
            self.registers[participant]["ses-01"]['data_csp']=csp.transform(self.registers[participant]["ses-01"]['data'])
            self.registers[participant]["ses-02"]['data_csp']=csp.transform(self.registers[participant]["ses-02"]['data'])
            self.registers[participant]["ses-03"]['data_csp']=csp.transform(self.registers[participant]["ses-03"]['data'])
            self.registers[participant]["ses-04"]['data_csp']=csp.transform(self.registers[participant]["ses-04"]['data'])
            self.registers[participant]["ses-05"]['data_csp']=csp.transform(self.registers[participant]["ses-05"]['data'])
            
            inf = create_info(self.registers[participant]['ch_names'].tolist(),
                              len(self.registers[participant]['ch_names']),
                              self.registers[participant]['sfreq'])
            fig = csp.plot_patterns(inf, ch_type='eeg', units='Patterns (AU)', size=1.5)
            if savefig:
                fig.set_size_inches((6,6))
                fig.savefig(path+participant+'_CSPfilter.png')
            plt.close()
            data_ = np.zeros((1,32,1000))
            labels_ = np.zeros(1)
            
            
            
    def csp_fit(self,nc = 6,savefig=False, path=""):
        self.csp = CSP(n_components = nc)
        self.csp.fit(self.data_train, self.labels_train, reg='oas')
        
    def csp_transform(self):
        for val_participant in self.validation_participants:
            self.registers[val_participant]["ses-01"]['data_csp']=self.csp.transform(self.registers[val_participant]["ses-01"]['data'])
            self.registers[val_participant]["ses-02"]['data_csp']=self.csp.transform(self.registers[val_participant]["ses-02"]['data'])
            self.registers[val_participant]["ses-03"]['data_csp']=self.csp.transform(self.registers[val_participant]["ses-03"]['data'])
            self.registers[val_participant]["ses-04"]['data_csp']=self.csp.transform(self.registers[val_participant]["ses-04"]['data'])
            self.registers[val_participant]["ses-05"]['data_csp']=self.csp.transform(self.registers[val_participant]["ses-05"]['data'])
            
        for train_participant in self.train_participants:
            self.registers[train_participant]["ses-01"]['data_csp']=self.csp.transform(self.registers[train_participant]["ses-01"]['data'])
            self.registers[train_participant]["ses-02"]['data_csp']=self.csp.transform(self.registers[train_participant]["ses-02"]['data'])
            self.registers[train_participant]["ses-03"]['data_csp']=self.csp.transform(self.registers[train_participant]["ses-03"]['data'])
            self.registers[train_participant]["ses-04"]['data_csp']=self.csp.transform(self.registers[train_participant]["ses-04"]['data'])
            self.registers[train_participant]["ses-05"]['data_csp']=self.csp.transform(self.registers[train_participant]["ses-05"]['data'])
            
            
    def plot_scatter_csp_per_subject(self,savefig=False,path=''):
        marks=['o','^','s','H','D']
        for participant in self.participants: 
            fig, ax = plt.subplots()
            fig.set_size_inches((8,8))
            it=0
            colors = sns.color_palette()
            df = pd.DataFrame()
            for session in self.sessions:
                df_aux = pd.DataFrame()
                X=self.registers[participant][session]['data_csp'].T
                N = X.shape[1]
                X_mu = np.array([X[0,:],X[-1,:]]).mean(axis=1)
                X_sigma = np.cov(np.array([X[0,:],X[-1,:]]))
                df_aux['session'] = N*[session]
                df_aux['CSP Primer Componente']=X[0,:]
                df_aux['CSP Ultima Componente']=X[-1,:]
                df_aux['labels_trials'] = self.registers[participant][session]['labels_trials']
                df = pd.concat([df, df_aux], axis=0)
            
            
            # #Hallamos la media de cada trial de cada se3sion
            # df_aux_1 = df.loc[df['labels_trials'] == 1.]
            # df_aux_    
                
            scatter_plot = sns.scatterplot(data=df, x="CSP Primer Componente", y="CSP Ultima Componente",
                                           hue="session",style='labels_trials')
            scatter_plot.set_title('Dispersión de trials ' + participant)
            scatter_plot.grid(True)
            scatter_plot.set_aspect('equal', adjustable='datalim')
            fig = scatter_plot.figure
            fig.set_size_inches((8,8))
            if savefig:
                fig.savefig(path+participant+'_disp_per_session.png')
                
            plt.show()
            plt.close()

            
    def data_pca_register(self,nc=2,plot_variance=False):
        for participant in self.participants:
                self.registers[participant]['data_pca'], eig_values= self._pca(self.registers[participant]['data_csp'], n_comp=2)
                if plot_variance:
                    fig,ax = plt.subplots()
                    a=1
                    for it in range(X.shape[0]):  
                        x=np.linspace(0,len(eig_values[it]),len(eig_values[it]))
                        ax.bar(x,eig_values[it],alpha=a)
                        a-=0.005
                self.registers[participant]['data_pca_mu']=self.registers[participant]['data_pca'].mean(axis=1)
                self.registers[participant]['data_pca_sigma']=self.registers[participant]['data_pca'].std(axis=1)          
    
    def data_gender(self):
        it=0
        self.idx_m=[]
        self.idx_f=[]
        for participant in self.participants:
            if self.registers[participant]['gender'] == 'M':
                self.idx_m.append(it)
            else:
                self.idx_f.append(it)
            it=it+1
            
    def plot_csp_pattern(self, gender = None, ndims=2, method='csp', n_data=None):
        idx_f = None
        idx_m = None
        if method == 'pca':
            print('No se ha implementado')
        elif method == 'csp':
            data2plot = []
            mu2plot = []
            sigma2plot = []
            pts = []
            it=0
            if gender == None: 
                pts = self.participants
                idx_f = self.idx_f[:n_data]
                idx_m = self.idx_m[:n_data]
                title = 'Dispersión CSP M-H'
            elif gender == 'M':
                for id in self.idx_m:
                    pts.append(self.participants[id])
                idx_m = [0,1]
                title = 'Dispersión CSP Hombres'
            else: 
                for id in self.idx_f:
                    pts.append(self.participants[id])
                idx_f = [0,1]
                title = 'Dispersión CSP Mujeres'
                
            for participant in pts:
                data_ = self.registers[participant]['data_csp'].T
                data2plot.append(np.array([data_[0,:],data_[-1,:]]))
                mu2plot.append(data2plot[-1].mean(axis=1))
                sigma2plot.append(np.cov(data2plot[-1]))
            # data2plot = np.array(data2plot)
            # mu2plot = np.array(mu2plot)
            # sigma2plot = np.array(sigma2plot) 
            
            if n_data == None: 
                plot_scatter_register_bygender(data=data2plot,mu=mu2plot,sigma=sigma2plot,idx_m=idx_m,idx_f=idx_f,title=title)
            else:
                plot_scatter_register_bygender(data=data2plot,mu=mu2plot,sigma=sigma2plot,idx_f=idx_f,idx_m=idx_m,title=title)

    def save_registers(self,path=''):
        if path == '':
            print('No se ha seleccionado una ruta de guardado')
        else:
            for participant in self.participants:
                save_path=path+participant+"_task_motorimagery_eeg_preprocessing_csp.mat"
                data_save = {}
                for session in self.sessions:
                    data_save[session +'_data_csp']=self.registers[participant][session]['data_csp']
                    data_save[session +'_labels_trials']=self.registers[participant][session]['labels_trials']
                data_save['sfreq'] = self.registers[participant]['sfreq'],
                data_save['age']=self.registers[participant]['age'],
                data_save['gender']=self.registers[participant]['gender'],
                data_save['group_medidator']=self.registers[participant]['group_medidator'],
                data_save['id_participant']=self.registers[participant]['id_participant']
                sio.savemat(save_path, data_save)
        
class ShuDataset(Preprocessing):
    def __init__(self,path=None,preprocessing=False):
        self.register={}
        self.n_epochs=None
        self.n_samples=None
        self.n_chans=None
        self.ica=None
        self.preprocessing=preprocessing
        if path==None:
            print('No se ha seleccionado un archivo')
        else:
            print('Se cargó el archivo')
            self.load_path(path)

    def load_path(self,path=''):
        if ".mat" in path:
            data=sio.loadmat(path)
            self.register['data']=data['data']
            self.n_epochs=data['data'].shape[0]
            self.n_samples=data['data'].shape[2]
            self.n_chans=data['data'].shape[1]
            if self.preprocessing:
                self.register['labels_trials']=data['labels_trials']
                self.register['sfreq']=data['sfreq']
                self.register['ch_names']=data['ch_names']
                self.register['age']=data['age']
                self.register['group_medidator']=data['group_medidator']
                self.register['gender']=data['gender']
                self.register['id_participant']=data['id_participant']
            else: 
                self.register['labels_trials']=data['labels']
            
    def load_sampling_rate(self, fs):
        self.register['sfreq']=fs
    
    def load_info_channels(self, nchans,channels_names):
        self.register['nchans']=nchans
        self.register['ch_names']=channels_names
        
    def load_participants_description(self,participant,path=""):
        data_participants=pd.read_csv(path,sep='\t')
        df=data_participants[data_participants['participant_id']== participant]
        self.register['age']=df['age'].values[0]
        self.register['gender']=df['gender'].values[0]
        self.register['id_participant']=participant
        self.register['group_medidator']=df['group'].values[0]
    
    def give_data(self,data=""):
        return self.register[data]
    
    def save_data(self,save_path=""):
        data_save={'data':self.register['data'],
                   'labels_trials':self.register['labels_trials'],
                   'sfreq':self.register['sfreq'],
                   'ch_names':self.register['ch_names'],
                   'age':self.register['age'],
                    'gender':self.register['gender'],
                    'group_medidator':self.register['group_medidator'],
                    'id_participant':self.register['id_participant']}
        sio.savemat(save_path, data_save)
        
        
        
    #------------------------PREPROCESSING--------------------------        

    def filter(self,freq,filter_type,order):
        self.register['data']=self._filter(self.register['data'], freq=freq, sfreq=self.register['sfreq'],
                                                       filter_type=filter_type, order=order)
    def filter_notch(self,f=50.,qf=20.):
        self.register['data']=self._filter_notch(self.register['data'], sfreq=self.register['sfreq'],f=f,quality_factor=qf)
        
        
    def find_bad_channels(self,fig_path="",plot=True):
        X=np.hstack(self.register['data'].copy())
        idx_bad_ch=self._bad_channels(X,savefig=True,path_file=fig_path,plot=plot)
        self.register['bad_ch_dispersion']=[]
        self.register['bad_ch_dispersion_idx']=[]
        if len(idx_bad_ch)>0:
            print("CANALES MALOS:")
            for it in range(len(idx_bad_ch)):
                self.register['bad_ch_dispersion'].append(self.register['ch_names'][idx_bad_ch[it]])
                self.register['bad_ch_dispersion_idx'].append(idx_bad_ch[it])
            print(self.register['bad_ch_dispersion'])
        else: 
            print("No hay canales malos")
            self.register['bad_ch_dispersion']='None'
        return self.register['bad_ch_dispersion']
            
    def data_z_score(self): 
        X=np.hstack(self.register['data'].copy())
        X=self._z_score(X)
        self.register['data'] = np.array(np.hsplit(X,self.n_epochs))
       
    def find_bad_epochs(self,save_fig=False,fig_path=""):
        idx_bad_epochs=self._bad_epochs(self.register['data'],savefig=save_fig,path_file=fig_path)
        self.register['bad_epochs_dispersion_idx']=[]
        if len(idx_bad_epochs)>0:
            self.register['bad_epochs_dispersion_idx']=idx_bad_epochs
            print("Epocas Malas:")
            print(self.register['bad_epochs_dispersion_idx'])
        else: 
            print("No hay Epocas malas")
            self.register['bad_epochs_dispersion_idx']="None"
        return self.register['bad_epochs_dispersion_idx']
    
    def remove_bad_epochs(self):
        if self.register['bad_epochs_dispersion_idx']!="None":
            self.register['data']=self._remove_bad_epochs( self.register['data'],self.register['bad_epochs_dispersion_idx'])
            self.register['labels_trials']=self._remove_bad_trials( self.register['labels_trials'],self.register['bad_epochs_dispersion_idx'])
            
    def compute_ica(self,method="SOBI"):
        self.ica={}
        if method=="SOBI":
            X=np.hstack(self.register['data'])
            self.ica['method']=method
            self._ica(X,self.register['ch_names'],self.register['sfreq'],method=method)
        elif method == 'FastICA':
            X=np.hstack(self.register['data'].copy())
            U , A , S= self.__ica__(X ,self.register['ch_names'],self.register['sfreq'],method='FastICA')
            S=np.array(np.hsplit(S,self.n_epochs))
            self.ica['method']=method
            self.ica['sources']=S
            self.ica['mixing_matrix']=A
            self.ica['unmixing_matrix']=U           
        elif method == 'FastICA-mne':
            X=self.register['data'].copy()
            A , U , W_pca , C_topo , S, ica_mne=self._ica(X, self.register['ch_names'],self.register['sfreq'],nchans=self.register['nchans'],method='FastICA-mne')
            self.ica['sources']=S
            self.ica['mixing_matrix']=A
            self.ica['unmixing_matrix']=U
            self.ica['PCA_matrix']=W_pca
            self.ica['C_topo']=C_topo
            self.ica['ICA_mne']=ica_mne
            self.ica['method']=method
            
    def find_eog_ica(self,method='mne'):
        X=self.register['data'].copy()
        idx=[]
        idx.append(self.register['ch_names'].index('Fp1'))
        idx.append(self.register['ch_names'].index('Fp2'))
        idx.append(self.register['ch_names'].index('Fz'))
        if method=='mne':
            eog0=np.expand_dims(X[:,idx[0],:],axis=1)
            eog1=np.expand_dims(X[:,idx[1],:],axis=1)
            eog2=np.expand_dims(X[:,idx[2],:],axis=1)
            eog3=-(eog0-eog1)
            name_channels=self.register['ch_names']+['EOG000']+['EOG001']+['EOG002']+['EOG003']
            X=np.concatenate((X,eog0,eog1,eog2,eog3),axis=1)
            data={'X':X, 'name_channels':name_channels,'sfreq':self.register['sfreq'],'cant_eog':4,'ICA_mne':self.ica['ICA_mne']}
            method='mne'
        else:
            #Get sources 
            S=np.hstack(self.ica['sources'])
            #Get features
            X=np.hstack(X)
            #Estimating EOG
            eog0=np.expand_dims(X[idx[0],:], axis=0)
            eog1=np.expand_dims(X[idx[1],:], axis=0)
            eog2=np.expand_dims(X[idx[2],:], axis=0)
            eog3=np.expand_dims(X[idx[1],:]-X[idx[0],:], axis=0)
            EOG = np.concatenate((eog0,eog1,eog2,eog3),axis=0)
            data={'EOG':EOG,'X':S,'sfreq':self.register['sfreq']}
            method=''
        idx_eog , zscore_eog = self._find_eog_artifact(data=data,method=method)
        self.ica['sources_eog']=idx_eog
        self.ica['sources_eog_zscore']=zscore_eog
        return self.ica['sources_eog']
        
    def find_emg_ica(self,method='mne'):
        X=self.register['data'].copy()
        if method=='mne':
            X=np.hstack(X)
            data={'ICA_mne':self.ica['ICA_mne'],'X':X,'ch_names':self.register['ch_names'],'sfreq':self.register['sfreq']}
            method='mne'
        else:
            data={'X':X,'sfreq':self.register['sfreq'],'n_lag':int(0.02*100.)}
            method=''
        idx_emg , zscore_emg = self._find_muscle_artifact(data,method=method)
        self.ica['sources_emg']=idx_emg
        self.ica['sources_emg_zscore']=zscore_emg
        return self.ica['sources_emg'],self.ica['sources_emg_zscore']
        
    def reconstruct_data(self): 
        X=self.register['data'].copy()
        components=list(np.unique(self.ica['sources_emg']+self.ica['sources_eog']))
        if self.ica['method'] == 'FastICA-mne':
            data={'X':X,'ch_names':self.register['ch_names'],
                  'sfreq':self.register['sfreq'],'ICA_mne':self.ica['ICA_mne'],
                  'components':components}
            method='mne'
        else: 
            print('NO SE HA IMPLEMENTADO EL ALGORTIMO')   
        self.register['data'] = self._reconstruct_data(data=data,method=method)
        
        
    def plot_eeg_data(self,method='',epochs_ini=0,n_epochs=10,savefig=False,path_fig=""):
        X=self.register['data'].copy()
        ch_names=self.register['ch_names']
        sfreq=self.register['sfreq']
        if method == 'mne':
            plot_eeg_epochs(X,ch_names,sfreq,stdmontage="standard_1020")
        else:
            data_plot = self.register['data'][epochs_ini:epochs_ini+n_epochs,:,:]
            data_plot = np.hstack(data_plot)
            fig = plot_register(data_plot,ch_names,self.n_samples,n_epochs,epochs_ini,return_fig=True)
            if savefig:
                fig.savefig(path_fig+".png")
        
    def plot_eeg_data_psd(self,savefig=False,path_fig=""):
        X=self.register['data'].copy()
        ch_names=self.register['ch_names']
        sfreq=self.register['sfreq']
        plot_eeg_epochs_psd(X,ch_names,sfreq,stdmontage="standard_1020",savefig=savefig,path_fig=path_fig)   
             
    def plot_ica_components(self,savefig=False,path_fig=""):
        if self.ica['method'] == 'FastICA-mne':
            plot_ica_components(self.ica['C_topo'],self.register['ch_names'],self.register['sfreq'],savefig=savefig,path_fig=path_fig)
        else:
            plot_ica_components(self.ica['mixing_matrix'],self.register['ch_names'],self.register['sfreq'],savefig=savefig,path_fig=path_fig)
            
    def plot_ica_sources(self,method='',epochs_ini=0,n_epochs=10,savefig=False,path_fig=""):
        ica_ch=[]
        for it in range(self.ica['sources'].shape[1]):
            ica_ch.append("ICA"+str(it))
        X=self.ica['sources'].copy()
        ch_names=ica_ch
        sfreq=self.register['sfreq']
        if method == 'mne':
            plot_eeg_sources(self.ica['sources'], ica_ch, self.register['sfreq'],savefig=savefig,path_fig=path_fig)
        else:
            data_plot = self.register['data'][epochs_ini:epochs_ini+n_epochs,:,:]
            data_plot = np.hstack(data_plot)
            fig = plot_register(data_plot,ch_names,self.n_samples,n_epochs,epochs_ini,return_fig=True)
            if savefig:
                fig.savefig(path_fig+".png")
        
    def plot_ica_scores(self,savefig=False,path_fig=""):
        if self.ica['method'] == 'FastICA-mne':
            fig0=self.ica['ICA_mne'].plot_scores(scores=self.ica['sources_eog_zscore'], title='ICA EOG scores', figsize=(12,12))
            fig1=self.ica['ICA_mne'].plot_scores(scores=self.ica['sources_emg_zscore'], title='ICA EMG scores', figsize=(12,12))
            if savefig:
                fig0.set_size_inches((6,6))
                fig1.set_size_inches((6,6))
                fig0.savefig(path_fig+"ICA_scores_eog.png")
                fig1.savefig(path_fig+"ICA_scores_emg.png")
        else:
            print('No se ha implementado')
        
        
   

    


