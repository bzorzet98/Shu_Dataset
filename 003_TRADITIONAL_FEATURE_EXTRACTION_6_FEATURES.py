#%% Importamos las librerias
from shu_dataset import RegistersShuDataset
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Leemos los datos y adaptamos la base de datos para obtener las epocas acondicionadas
path_root = "/home/bruno/Academico/Doctorado/Neuro_Fairness/Shu_Dataset/"
dataset_field="000_Dataset_Preprocessing/"
save_figures = True
save_path = path_root + '003_Dataset_Traditional_Feature_Extraction_6_Feature/'
participants=["sub-001","sub-002","sub-003","sub-004","sub-005",
              "sub-006","sub-007","sub-008","sub-009","sub-010",
              "sub-011","sub-012","sub-013","sub-014","sub-015",
              "sub-016","sub-017","sub-018","sub-019","sub-020",
              "sub-021","sub-022","sub-023","sub-024","sub-025"]

sessions=["ses-01","ses-02","ses-03","ses-04","ses-05"]

#%% Cargamos los datos
registers = RegistersShuDataset()
registers.load_registers(path = path_root + dataset_field, participants = participants, sessions = sessions)

# -------------------- Procesamiento-------------------------#
# 1. Calculamos la potencia de cada epoca en las bandas de frecuencia establecidas
bands_freqs = [(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)]
registers.spectral_band_power(bands_freqs=bands_freqs,mean_across_ch=True)
# 2. Guardamos los datos
registers.save_registers(path=save_path)

a=20