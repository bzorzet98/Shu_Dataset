#%% Importamos las librerías
import pandas as pd
import numpy as np
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
from random import sample, shuffle

from scipy.stats import ks_1samp
from scipy.stats import ttest_1samp
from scipy.stats import wilcoxon
from scipy import stats

from fairness.functions import *
from mne.decoding import  CSP
#%% Load data
path_root = "/home/bruno/Academico/Doctorado/Neuro_Fairness/Shu_Dataset/"
dataset_field="001_Dataset_CSP/"
save_figures = True
figures_path_root = path_root + "005_Figures/"
tables_path_root = path_root + "005_Tables/"
dataset_path_root = path_root + dataset_field
participants=["sub-001","sub-002","sub-003","sub-004","sub-005",
              "sub-006","sub-007","sub-008","sub-009","sub-010",
              "sub-011","sub-012","sub-013","sub-014","sub-015",
              "sub-016","sub-017","sub-018","sub-019","sub-020",
              "sub-021","sub-022","sub-023","sub-024","sub-025"]
sessions = ["ses-01","ses-02","ses-03","ses-04","ses-05"]

dataset={}
for participant in participants:
    dataset[participant]={}
    data_path=participant+"_task_motorimagery_eeg_preprocessing_csp.mat"
    data=sio.loadmat(dataset_path_root + data_path)
    for session in sessions:
        dataset[participant][session +'_data_csp']=data[session +'_data_csp']
        dataset[participant][session +'_labels_trials']=data[session +'_labels_trials']
    dataset[participant]['sfreq']=np.squeeze(data['sfreq'])
    dataset[participant]['age']=np.squeeze(data['age'])
    dataset[participant]['gender']=data['gender'][0]
    dataset[participant]['group_medidator']=data['group_medidator'][0]
    dataset[participant]['id_participant']=data['id_participant'][0]
#%% Obtain male and female indexes
index_female = []
index_male = []
participant_female = []
participant_male = []
it = 0
for participant in participants:
    if dataset[participant]['gender'] == 'M':
        index_male.append(it)
        participant_male.append(participant)
    elif dataset[participant]['gender'] == 'F':
        index_female.append(it)
        participant_female.append(participant)
    it+=1
print(f"Participantes hombres: {participant_male}")
print(f"Participantes mujeres: {participant_female}")
#%% Calculate the mean and covariance matrix per participant
it=0
matrix_mean_csp = np.zeros((6,len(participants)))
matrix_cov_csp = np.zeros((6,6,len(participants)))
for participant in participants:
    data_ = np.concatenate((dataset[participant]['ses-01_data_csp'],
                            dataset[participant]['ses-02_data_csp'],
                            dataset[participant]['ses-03_data_csp'],
                            dataset[participant]['ses-04_data_csp'],
                            dataset[participant]['ses-05_data_csp']),axis=0)
    data_mean_ = np.mean(data_ , axis=0)
    data_cov_ = np.cov(data_.T)
    matrix_mean_csp[:,it] = data_mean_
    matrix_cov_csp[:,:,it] = data_cov_
    dataset[participant]['data_csp'] = {'all_data' : data_, 'mean': data_mean_, 'cov': data_cov_}
    it+=1

#%% Find and exclude of the analysis the man whose average deviates from the rest
means_male_csp = matrix_mean_csp[:,index_male]
mean_male = means_male_csp.mean(axis=1,keepdims=True)

dist_males_mean = np.linalg.norm(means_male_csp - mean_male,axis = 0)
idx_male_exclude = dist_males_mean.argmax()
print(f'El sujeto que es excluido del análisis por tener una distancia con respecto a la media es {participant_male[idx_male_exclude]}')
index_male_origin = index_male.copy()
participant_male_origin = participant_male.copy()
index_male.remove(idx_male_exclude)
participant_male.remove(participant_male[idx_male_exclude])

#%% Calculate the Symmetrised_Divergence_KL for all participants
D_KLS=np.zeros((len(participants),len(participants)))
for it0 in range(0,len(participants)-1):
    for it1 in range(it0+1,len(participants)):
        mu_p = matrix_mean_csp[:,it0]
        mu_q = matrix_mean_csp[:,it1]
        sigma_p = matrix_cov_csp[:,:,it0]
        sigma_q = matrix_cov_csp[:,:,it1]
        D_KLS[it0,it1] = dkls(mu_p, mu_q, sigma_p, sigma_q)

#%% Distances female - male
D_KLS_fm = []
N_samples = int(12*11/2)
for idx_m in index_male:
    for idx_f in index_female:
        if D_KLS[idx_m,idx_f]!=0:
            D_KLS_fm.append(D_KLS[idx_m,idx_f])
            
for idx_f in index_female:
    for idx_m in index_male:
        if D_KLS[idx_f,idx_m]!=0:
            D_KLS_fm.append(D_KLS[idx_f,idx_m])
            

#%Distances male - male
D_KLS_mm = []
for idx_m0 in index_male:
    for idx_m1 in index_male:
        if D_KLS[idx_m0,idx_m1]!=0:
            D_KLS_mm.append(D_KLS[idx_m0,idx_m1])
D_KLS_mm=np.array(D_KLS_mm)

#%Distances female - female
D_KLS_ff = []
for idx_f0 in index_female:
    for idx_f1 in index_female:
        if D_KLS[idx_f0,idx_f1]!=0:
            D_KLS_ff.append(D_KLS[idx_f0,idx_f1])
D_KLS_ff=np.array(D_KLS_ff)


#% Repeat N_it times the analysis
N_it = 20

for it in range(N_it):
    #% Select distances randomly
    D_KLS_fm_selected =  sample(D_KLS_fm, N_samples)
    D_KLS_fm_selected = np.array(D_KLS_fm_selected)
    #%% Histogram 
    fig,ax=plt.subplots()
    fig.set_size_inches((6,6))
    cm = plt.cm.get_cmap('Set1')
    ax.hist(D_KLS_mm,label='Hombre - Hombre',density=True,alpha=1,color= cm(0) )
    ax.hist(D_KLS_fm_selected,label='Mujer - Hombre',density=True,alpha=0.8,color= cm(1))
    ax.hist(D_KLS_ff,label='Mujer - Mujer',density=True,alpha=0.6,color= cm(2))

    ax.set_xlabel('$D_{KLS}$')
    ax.set_ylabel('Frecuencias relativas')
    ax.set(title='Histogramas de distancias $D_{KLS}$')
    ax.legend()

    if save_figures:
        fig.set_size_inches((6,6))
        fig.savefig(figures_path_root+f'{it}_DKLS_Histogram.png')


    #%% Test Hypothesis
    alpha = 0.05
    # Norm Test
    test_norm = pd.DataFrame(columns=['statistics','p-value','H0','Description'])
    comparations = ['Female - Male','Female - Female','Male - Male']
    norm_test = ks_1samp(D_KLS_fm_selected,stats.norm.cdf)
    if norm_test.pvalue <= alpha:
        test_norm.loc['Female - Male']=[norm_test.statistic,norm_test.pvalue,False,f'Reject H0, distribution is not normal(p < {alpha})']
    else:
        test_norm.loc['Female - Male']=[norm_test.statistic,norm_test.pvalue,True,f'Accept H0, distribution is normal(p > {alpha})']
    norm_test = ks_1samp(D_KLS_ff,stats.norm.cdf)
    if norm_test.pvalue <= alpha:
        test_norm.loc['Female - Female']=[norm_test.statistic,norm_test.pvalue,False,f'Reject H0, distribution is not normal(p < {alpha})']
    else:
        test_norm.loc['Female - Female']=[norm_test.statistic,norm_test.pvalue,True,f'Accept H0, distribution is normal(p > {alpha})']
    norm_test = ks_1samp(D_KLS_mm,stats.norm.cdf)
    if norm_test.pvalue <= alpha:
        test_norm.loc['Male - Male']=[norm_test.statistic,norm_test.pvalue,False,f'Reject H0, distribution is not normal(p < {alpha})']
    else:
        test_norm.loc['Male - Male']=[norm_test.statistic,norm_test.pvalue,True,f'Accept H0, distribution is normal(p > {alpha})']


    test_norm.to_csv(tables_path_root + f'{it}_test_norm.csv')

    test = pd.DataFrame(columns=['statics','p-value','H0','Description'])
    U_test = stats.mannwhitneyu(D_KLS_fm_selected,D_KLS_mm)
    if U_test.pvalue <= alpha:
        test.loc['Female - Male / Male - Male']=[U_test.statistic,U_test.pvalue,False,f'Reject H0, two distribution have not equal mean(p < {alpha})']
    else: 
        test.loc['Female - Male / Male - Male']=[U_test.statistic,U_test.pvalue,True,f'Accept H0,  two distribution maybe have equal mean(p > {alpha})']

    U_test = stats.mannwhitneyu(D_KLS_fm_selected,D_KLS_ff)
    if U_test.pvalue <= alpha:
        test.loc['Female - Male / Female - Female']=[U_test.statistic,U_test.pvalue,False,f'Reject H0, two distribution have not equal mean(p < {alpha})']
    else: 
        test.loc['Female - Male / Female - Female']=[U_test.statistic,U_test.pvalue,True,f'Accept H0,  two distribution maybe have equal mean(p > {alpha})']


    test.to_csv(tables_path_root + f'{it}_test.csv')

a=20