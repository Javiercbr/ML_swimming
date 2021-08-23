# Regression analysis of swimmer performance

Regression algorithms for predictin swimmer performance at 18 year old based on records at younger ages. Data have been retrieved from the website [1] as CSV using different filters. The approach bases in [2] but different regressors have been employed.

Set| Number of samples | Covid19 [%]
--- | --- | ---
Training | 5318| 50
Validation | 1486| 48
Test | 755| 53


<p align="center">
  <img src="missing_records.png" width="400" title="c"> 
</p>



<p align="center">
  <img src="swimmers_pred.png" width="400" title="b"> 
</p>


<p align="center">
  <img src="comparative.png" width="400" title="a"> 
</p>


[1] https://www.usaswimming.org/

[2] Jiang Xie,  Junfu Xu,  Celine Nie,  Qing Nie, Machine learning of swimming data via wisdom of crowd and regression analysis, Mathematical Biosciences & Engineering 2017, 14(2): 511-527 doi: 10.3934/mbe.2017031
