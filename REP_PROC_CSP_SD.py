import pandas as pd
import numpy as np
from fairness.report import Report

participants=["sub-001","sub-002","sub-003","sub-004","sub-005",
              "sub-006","sub-007","sub-008","sub-009","sub-010",
              "sub-011","sub-012","sub-013","sub-014","sub-015",
              "sub-016","sub-017","sub-018","sub-019","sub-020",
              "sub-021","sub-022","sub-023","sub-024","sub-025"]

report=Report("PROC_CSP_SD_REPORTS/Processing_Report.pdf")

figure_path="PROC_CSP_SD_FIGURES/"

#Generamos el report
report.title('Procesamiento ')
report.set_hLine()
    
for participant in participants:   
    report.subtitle('CSP Filters '+participant)
    report.set_image(figure_path+participant+"_CSPfilter.png")
    report.set_page_break()
    report.subtitle('Dispersión de los trials '+participant)
    report.set_image(figure_path+participant+"_disp_per_session.png")
    report.set_page_break()
report.build()  
a=20
