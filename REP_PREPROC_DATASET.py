import pandas as pd
import numpy as np
from fairness.report import Report

participants=["sub-001","sub-002","sub-003","sub-004","sub-005",
              "sub-006","sub-007","sub-008","sub-009","sub-010",
              "sub-011","sub-012","sub-013","sub-014","sub-015",
              "sub-016","sub-017","sub-018","sub-019","sub-020",
              "sub-021","sub-022","sub-023","sub-024","sub-025"]
sessions=["ses-01","ses-02","ses-03","ses-04","ses-05"]

for participant in participants:
    figure_path="PREPROC_DATASET_FIGURES/"+participant+"/"
    tables_path="PREPROC_DATASET_TABLES/"+participant+"/"
    #Cargamos las tablas
    data_filters=pd.read_csv(tables_path+participant+'_data_filters.csv',sep=',', index_col=0)
    data_sessions=pd.read_csv(tables_path+participant+"_"+"summary_table.csv",sep=',', index_col=0)


    #Generamos el report
    report=Report("PREPROC_DATASET_REPORTS/Preprocessing_Report_"+participant+".pdf")
    report.title('Preprocesamiento '+participant)
    report.set_hLine()
    report.subtitle('Datos de Filtrado')
    data=np.expand_dims(data_filters.columns.to_numpy(),axis=0)
    data_val=data_filters.values
    data=np.concatenate((data,data_val),axis=0).tolist()
    report.table(data)
    report.set_hLine()
    report.subtitle('Resumen de Preprocesamiento de cada sesión')
    data=np.expand_dims(data_sessions.columns.to_numpy(),axis=0)
    data_val=data_sessions.values
    data=np.concatenate((data,data_val),axis=0).tolist()
    report.table(data)
    report.set_paragraph('Los datos prreprocesados se guardaron en un archivo .mat con el mismo nombre agregando _preprocessing. A estos datos se les agregaron los metadatos de los sujetos.')
    report.set_paragraph('Los datos en el archivo.mat se guardaron como diccionarios de diccionarios con las claves DATA y METADATA.')
    report.set_page_break()
    report.title('Imagenes del preprocesamiento de datos')
    for it in range(len(sessions)):
        report.title(f'Sesión {it+1}')
        report.subtitle('Canales Malos')
        report.set_image(figure_path+participant+"_"+sessions[it]+"_"+"disp_canales.png")
        report.set_page_break()
        report.subtitle('Epocas Malas')
        report.set_image(figure_path+participant+"_"+sessions[it]+"_"+"disp_epocas.png")
        report.set_page_break()
        report.subtitle('Fuentes de ICA')
        report.set_image(figure_path+participant+"_"+sessions[it]+"_"+"ICA_sources.png",width_=1,heigth_=0.85)
        icas_eog=data_sessions.iloc[it,4]
        report.set_paragraph("ICAs EOG: "+icas_eog)
        icas_emg=data_sessions.iloc[it,5]
        report.set_paragraph("ICAs EMG: "+icas_emg)
        report.set_page_break()
        report.subtitle('Componentes de ICA')
        report.set_image(figure_path+participant+"_"+sessions[it]+"_"+"ICA_comp_0.png")
        report.set_image(figure_path+participant+"_"+sessions[it]+"_"+"ICA_comp_1.png")
        report.set_page_break()
        report.subtitle('Scores de los artefactos')
        report.set_image(figure_path+participant+"_"+sessions[it]+"_"+"ICA_scores_emg.png")
        report.set_image(figure_path+participant+"_"+sessions[it]+"_"+"ICA_scores_eog.png")
        report.set_page_break()
        report.subtitle('EEG Original')
        report.set_image(figure_path+participant+"_"+sessions[it]+"_"+"EEG_original.png",width_=1,heigth_=0.85)
        report.set_page_break()
        report.subtitle('EEG Reconstruido')
        report.set_image(figure_path+participant+"_"+sessions[it]+"_"+"EEG_reconstruido.png",width_=1,heigth_=0.85)
        report.set_page_break()
        report.subtitle('Espectro de Potencias EEG Reconstruido')
        report.set_image(figure_path+participant+"_"+sessions[it]+"_"+"EEG_reconstruido_Espectro_Potencias.png")
        report.set_page_break()
    report.build()