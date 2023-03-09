import pandas as pd
import numpy as np
from fairness.report import Report

figure_path="EXP_STATS_DIFF_CSP_SI_ACCORDING_GENDER_FIGURES/"
tables_path="EXP_STATS_DIFF_CSP_SI_ACCORDING_GENDER_TABLES/"
reports_path="EXP_STATS_DIFF_CSP_SI_ACCORDING_GENDER_REPORTS/"
N_it = 20
#Generamos el report
report=Report(reports_path+"Report.pdf")
report.title(f"Búsqueda estadística de diferencias en las distribuciones de datos procesados")
report.set_paragraph('Se evaluó la distancia entre todos los sujetos (utilizando la divergencia de Kullback-Liebler simetrizada). Este calculo , se realizó sobre los registros procesados de la base de datos Shu Dataset, estos registros fueron preprocesados y luego se aplicó Commun Spatial Patterns para extraer 6 caracteristicas de cada trial de imagenería motora.\n Se agruparon las distancias en intra (MUJER-MUJER y HOMBRE-HOMBRE) e inter-categoría (MUJER-HOMBRE).Para comparar las 3 distribuciones a igual número de muestras, se procedió a sub-muestrear tomando 66 pares, correspondiente al conjunto de menor cardinalidad. Dado que poseemos un número dispar de hombres y mujeres, se excluyó del análisis al hombre cuyo valor medio estuviera más alejado del promedio de los hombres, y en el caso de los pares de sexo mixto se sub-muestreó de forma aleatoria sin repetición.')
report.set_paragraph('La extracción de caracteristicas mediante CSP se realizo sujeto independiente, donde se entreno CSP con los datos de 8 hombres y 8 mujeres, es decir, un análisis sujeto independiente.')
report.set_paragraph('Se realizaron 20 experimentos donde se repitió el proceso: SELECCIÓN DE SUJETOS PARA ENTRENAR CSP, PROYECCIÓN DE LOS DATOS, EVALUACION DE DKLS, ELIMINACIÓN DEL HOMBRE CON MAYOR DESVIO, TESTs DE HIPOTESIS')
report.set_paragraph('Se realizaron test de normalidad y test de comparación de medias, para determinar si las distancias inter e intra categoría provienen de distribuciones cuyos valores medios son iguales.')
report.set_paragraph('A continuación se muestran las diferentes repeticiones del experimento, con su histograma y los resultados del test de comparación de distribuciones')
report.set_page_break()
for it in range(N_it):
    #Load tables
    exp_info_table = pd.read_csv(tables_path+f"{it}_exp_info.csv",sep=',', index_col=0)
    test_table = pd.read_csv(tables_path+f"{it}_test.csv",sep=',', index_col=0)
    norm_table = pd.read_csv(tables_path+f"{it}_test_norm.csv",sep=',', index_col=0)

    report.title(f"Iteración {it}:")
    report.set_hLine()
    report.subtitle('Información general')
    exclude_part = exp_info_table.iloc[0]['EXCLUDE']
    train_part = exp_info_table.iloc[0]['TRAIN']
    test_part = exp_info_table.iloc[0]['TEST']
    report.set_paragraph(f'Los participantes de entrenamiento fueron: {train_part}')
    report.set_paragraph(f'El participante excluido fue: {exclude_part}')
    report.set_paragraph(f'Los participantes de testeo: {test_part}')

    report.subtitle('Histograma')
    report.set_image(figure_path+f"{it}_DKLS_Histogram.png",width_=0.4,heigth_=0.4)
    
    report.subtitle('Test de Normalidad')
    columns = norm_table.columns.to_numpy().tolist()
    columns.insert(0,'Distances')
    data=np.expand_dims(np.array(columns),axis=0)
    data_val =np.concatenate((np.array([[norm_table.index[0]],[norm_table.index[1]],[norm_table.index[2]]]),norm_table.values),axis=1)
    data=np.concatenate((data,data_val),axis=0).tolist()
    report.table(data)
    report.subtitle('Test de Mann-Whitney U')
    columns = test_table.columns.to_numpy().tolist()
    columns.insert(0,'Comparation')
    data=np.expand_dims(columns,axis=0)
    data_val =np.concatenate((np.array([[test_table.index[0]],[test_table.index[1]]]),test_table.values),axis=1)
    data=np.concatenate((data,data_val),axis=0).tolist()
    report.table(data)
    report.set_page_break()
    
report.build()