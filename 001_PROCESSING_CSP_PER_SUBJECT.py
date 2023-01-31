#%% Importamos las librerias
from shu_dataset import RegistersShuDataset
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%% Leemos los datos y adaptamos la base de datos para obtener las epocas acondicionadas
path_root = "/home/bruno/Academico/Doctorado/Neuro_Fairness/Shu_Dataset/"
dataset_field="B_Dataset_Preprocessing/"
save_figures = True
figures_path_root = path_root + "Processing_Figures_003/"
save_path = path_root + 'C_Dataset_CSP_003/'
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
# 1. Realizamos CSP
registers.csp_per_subject(savefig=True,path=figures_path_root)
# 2. Graficamos la dispersión de las sesiones con la primer y ultima componente de CSP
registers.plot_scatter_csp_per_subject(path=figures_path_root,savefig=save_figures)
# 3. Guardamos los datos
registers.save_registers(path=save_path)

a=20