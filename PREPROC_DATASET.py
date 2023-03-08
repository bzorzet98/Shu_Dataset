#%% Importamos las librerias
from shu_dataset import ShuDataset

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Leemos los datos y adaptamos la base de datos para obtener las epocas acondicionadas
path_root = "/home/bruno/Academico/Doctorado/Neuro_Fairness/Shu_Dataset/"
dataset_field="DATASET/"
info_channels="task-motorimagery_channels.tsv"
info_eeg_setup="task-motorimagery_eeg.json"
info_electrodes_position="task-motorimagery_electrodes.tsv"
info_channels=pd.read_csv(dataset_field+info_channels,sep='\t')
info_participants = "participants.tsv"
info_eeg_setup=open(dataset_field+info_eeg_setup,)
info_eeg_setup=json.load(info_eeg_setup)
#   COMPLETADOS 
# participants=["sub-001","sub-002","sub-003","sub-004","sub-005"]
# participants=["sub-006","sub-007","sub-008","sub-009","sub-010"]
# participants=["sub-011","sub-012","sub-013","sub-014","sub-015"]
# participants=["sub-016","sub-017","sub-018","sub-019","sub-020"]
participants=["sub-021","sub-022","sub-023","sub-024","sub-025"]
# participants=["sub-001","sub-002","sub-003","sub-004","sub-005",
#               "sub-006","sub-007","sub-008","sub-009","sub-010",
#               "sub-011","sub-012","sub-013","sub-014","sub-015",
#               "sub-016","sub-017","sub-018","sub-019","sub-020",
#               "sub-021","sub-022","sub-023","sub-024","sub-025"]
sessions=["ses-01","ses-02","ses-03","ses-04","ses-05"]

#%% #-------------------------- CONFIGURACIONES DEL PREPROCESAMIENTO ---------------------------------- #

# FILTRADO
# Filtro Pasa-Banda
l_freq=1. 
h_freq=43.
order=5
# Filtro Notch
f_notch=50.
q_factor=20.

# METODO DE ANALISIS DE COMPONENTES INDEPENDIENTES
ica_method='FastICA-mne'

# IMPRIMIR FIGURAS EN LA PANTALLA
flag_figures=True

#TABLA DE DATOS DE FILTRADO 
data=[['High-Pass','IIR','Butterworth',str(l_freq),str(order)],
      ['Low-Pass','IIR','Butterworth',str(h_freq),str(order)],
      ['Notch','IIR','',str(f_notch),str(q_factor)]]
columns=['Filtro','Tipo','Funcion','Fc','Orden']
data_filters = pd.DataFrame(data,columns=columns)

#GUARDAR LOS REGISTROS MODIFICADOS: 
save_register=True

#%% -------------------------------------------------PREPROCESAMIENTO----------------------------------------------------------
for participant in participants:
    database_path = path_root + dataset_field + participant + "/"
    figures_path_root = path_root + "PREPROC_DATASET_FIGURES/" + participant +"/" + participant + "_"  
    tables_path_root = path_root +"PREPROC_DATASET_TABLES/" + participant +"/"
    save_path_root = path_root + "DATASET_PREPROCESSING/" + participant +"/"+participant+"_"
    
    data_sessions = pd.DataFrame(columns=['Sesiones','CH malos z-score', 'Epocas Malas','Método ICA', 'ICA_EOG_idx','ICA_EMG_idx','Epocas Malas'])
    for session in sessions:
        data_path=participant+"_"+session+"_task_motorimagery_eeg.mat"
        fig_path = figures_path_root + session + "_"
        
        #----------------------ACONDICIONAMOS LOS DATOS ------------------------------------#
        register=ShuDataset(path=database_path+data_path)
        #% Cargamos la descripcion de los participantes
        register.load_participants_description(participant=participant,path=dataset_field+info_participants)
        register.load_info_channels(nchans=info_eeg_setup['EEGChannelCount'], channels_names=list(info_channels['name']))
        register.load_sampling_rate(fs=info_eeg_setup['SamplingFrequency'])
        
        #Graficamos los datos 
        #register.plot_eeg_data()
        #Graficamos la densidad espectral
        #register.plot_eeg_data_psd()
        
        #--------------------------PREPROCESAMIENTO--------------------------------#
        # 1. FILTRADO
        register.filter(freq=l_freq, filter_type="highpass", order=order)
        register.filter(freq=h_freq, filter_type="lowpass", order=order)
        register.filter_notch(f=f_notch,qf=q_factor)
            
        if flag_figures:
            name_fig = "EEG_original"
            register.plot_eeg_data(method='',epochs_ini=0,n_epochs=10,savefig=True,path_fig=fig_path+name_fig)
            plt.close()
            register.plot_eeg_data_psd()
            plt.close()
            
        # 2. IDENTIFICACIÓN DE BADS CHANNELS 
        
        bad_ch = register.find_bad_channels(fig_path=fig_path)
        plt.close()
        
        
        # 3. Z-SCOREAMOS LOS DATOS A LO LARGO DE CADA CANAL
        register.data_z_score()
        
        
        # 4. IDENTIFICACIÓN DE BADS EPOCHS, SIN REMOVERLAS DEL POSTERIOR ANÁLISIS
        bad_epochs_0 = register.find_bad_epochs()
        plt.close()
        
        # 5. ANÁLISIS DE COMPONENTES INDEPENDIENTES
        register.compute_ica(method=ica_method)
        
        if flag_figures:
            register.plot_ica_components(savefig=True,path_fig=fig_path)
            plt.close()
            name_fig ="ICA_sources"
            register.plot_ica_sources(savefig=True,path_fig=fig_path+name_fig)
            plt.close()
            
        # 6. ENCONTRAMOS LAS COMPONENTES PERTENECIENTES AL EOG Y A RUIDO MUSCULAR
        idx_eog = register.find_eog_ica()
        idx_emg, _ = register.find_emg_ica()
        if flag_figures:
            register.plot_ica_scores(savefig=True,path_fig=fig_path)
            plt.close()
            
        # 7. REPROYECTAMOS LOS DATOS ORIGINALES ELIMINANDO LAS COMPONENTES DE RUIDO
        register.reconstruct_data()
        
        if flag_figures:
            name_fig = "EEG_reconstruido"
            register.plot_eeg_data(method='',epochs_ini=0,n_epochs=10,savefig=True,path_fig=fig_path+name_fig)
            plt.close()
            register.plot_eeg_data_psd(savefig=True,path_fig=fig_path+"EEG_reconstruido_")
            plt.close()
            
        # 8. IDENTIFICAMOS BADS EPOCHS    
        
        bad_epochs_1 = register.find_bad_epochs(save_fig=True,fig_path=fig_path)
        plt.close()
        
        # 9. REMOVEMOS BADS EPOCHS 
        
        register.remove_bad_epochs()
        
        # 10. GUARDAMOS LOS DATOS PREPROCESADOS
        if save_register:
            save_path = save_path_root + session + "_task_motorimagery_eeg_preprocessing.mat"
            register.save_data(save_path=save_path)
        
        # Completamos los datos en la tabla de guardado 
        serie_session=pd.Series([session,str(bad_ch),str(bad_epochs_0),ica_method,str(idx_eog),str(idx_emg),str(bad_epochs_1)], index=data_sessions.columns)
        data_sessions=data_sessions.append(serie_session,ignore_index=True)
    
    data_filters.to_csv(tables_path_root + participant + '_data_filters.csv')
    data_sessions.to_csv(tables_path_root + participant + "_" + "summary_table.csv")
a=20