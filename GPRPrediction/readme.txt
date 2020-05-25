1.GPR_GitHub
    This procedure is used for GPR prediction.Here are some tips may help you when you're using it.
    a.The 'date' format is as 2020/03/10.The 'disease_pixels' represent the number of disease pixels detected in CT images.The 'lobe_pixels' represent the number of lobes detected in CT image.
    b.The 'outy' of 34 lines in the program indicates the forecast days.
    c.GPR kernel using in this method is RBF(1.0),you can change it if you have a better idea.
    d.The 'obsnum' of 67 lines in the program indicates the input data used for GPR.
    f.Before the program runs, you need to create a 'correctdatafig' folder and a 'wrongdatafig' folder under the project folder, otherwise the program will not run.
    g.The file path to be set is on line 19,20,37,229-232
    h.The predicted date error is in column 'range' of Table GPtestdatalack1.xlsx.
2.medicine_analysis
    This procedure is used to calculate the patient's most severe date and its effect on the slope.
3.PearsonCOVID-19
    This procedure is used to calculate the Pearson correlation coefficient between the 'the_area_ratio' and the days of medication.
4.plot
    This program is used to draw histogram.
5.Inspection report
    This procedure is used to calculate the correlation between physical examination data and CT data in the test report.
6.calculate_Inspection
    This procedure is used to calculate the correlation of test reports.
7.GPR_Inspection
    This program is used to calculate the GPR curve of inspection report.