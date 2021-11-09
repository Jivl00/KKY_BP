### Heart Disease UCI
#### [Dataset webpage][1]
---

##### Data
- 303 samples
- binary classification (165 positive / 138 negative samples)
- 13 features

##### Features
1. age 
2. sex 
3. chest pain type (4 values) 
4. resting blood pressure 
5. serum cholestoral in mg/dl 
6. fasting blood sugar > 120 mg/dl
7. resting electrocardiographic results (values 0,1,2)
8. maximum heart rate achieved 
9. exercise induced angina 
10. oldpeak = ST depression induced by exercise relative to rest 
11. the slope of the peak exercise ST segment 
12. number of major vessels (0-3) colored by flourosopy 
13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

##### Tasks
- data normalization
- data split (0.8/0.1/0.1)
- baseline MLP
- evaluation (TP, FP, TN, FN, ROC)
- best model grid search

##### Tags
- Keras
- GPU Remote training (SSH)
- .git + GitHub
- .md Documentation
- cross-validation


[1]: https://www.kaggle.com/ronitf/heart-disease-uci