--------------------- SHU DATASET ------------------------------


Esta base de datos cuenta con 25 participantes realizando task de imagenería motora de mano derecha e izquierda

Link del paper: https://www.nature.com/articles/s41597-022-01647-1#Sec6

Base de datos original: https://figshare.com/articles/software/shu_dataset/19228725/1

Ṕara realizar el preprocesamiento, se reestructuro la base de datos. 
En el siguiente link se encuentran las carpetas con los archivos crudos y modificados: 
https://drive.google.com/drive/folders/1p5alQyq7OC5GJTTfLagYQxfirssnBirA?usp=sharing

CARPETAS : 

* 000_Dataset : es la base de datos original con los registros de eeg en extensión .mat. 
    Esta carpeta es usada para los scripts 000_XXXXXX
* 000_Preprocessing_XXXXX: las carpetas que cuentan con las figuras, tablas y reportes obtenidas de los scripts.
* 001_Dataset_Preprocessing: es la base de datos preprocesada por el script 000_PREPROCESSING.py, donde
    se realizo una limpieza de cada uno de los registros. 
* 001_Processing_XXXX: las carpetas que cuentan con las figura y reportes obtenidas de los scripts 001_PROCESSING_XXXXXX.
* 002_Dataset_CSP: es la carpeta que contiene la base de datos con los registros procesados por el script 001_PROCESSING_CSP_PER_SUBJECT.py

