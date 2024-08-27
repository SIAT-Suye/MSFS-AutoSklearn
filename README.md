# MSFS-AutoSklearn
## 1 Introduction
With the advancement of data science, Automated Machine Learning (AutoML) has gained widespread attention due to its ability to lower barriers, save time, and improve efficiency. However, as the dimensionality of data increases, AutoML faces challenges in handling large feature sets. This research proposes an efficient modeling framework that combines a Multi-Stage Feature Selection (MSFS) algorithm with AutoSklearn to address the challenges posed by high-dimensional data.

## 2 Background
In the context of the booming data science field, the successful application of Machine Learning (ML) across various domains has garnered widespread attention. However, traditional ML workflows often involve complex and time-consuming steps such as data preprocessing, feature engineering, model selection, and hyperparameter tuning. Automated Machine Learning (AutoML) aims to simplify these processes, lowering the application barrier and enhancing the deployment capabilities of model algorithms. Effective feature selection has become crucial in ensuring the efficient operation of AutoML when facing high-dimensional data challenges.

## 3 Feature Selection Algorithm
The proposed MSFS algorithm consists of three stages:

### 3.1 MIG Filtering: 
Calculates the mutual information gain between features and the target variable to select strongly correlated features.
### 3.2 RFECV Selection: 
Utilizes Recursive Feature Elimination with Cross-Validation (RFECV) to iteratively select important features based on feature inter-correlations.
### 3.3 Voting Aggregation Mechanism: 
Determines the final feature subset through a voting mechanism, optimizing the model's stability.
![image](https://github.com/user-attachments/assets/270283b9-5d3b-45fb-aa53-7aedb90d2321)

## 4 Experiments and Results
Using the AutoSklearn framework to model forest fire risk, the effectiveness of the MSFS algorithm was validated. Experimental results indicate that MSFS outperforms other feature selection algorithms in both feature selection and predictive performance, significantly enhancing the success rate and accuracy of AutoSklearn.

## 5 Environment Requirements
This study was conducted on the Kaggle platform (https://www.kaggle.com/), a globally renowned platform for data science and machine learning competitions. The experimental environment was configured with a Linux system, running version 5.15.133+. The Python interpreter was version 3.7.12, compiled by GCC 9.4.0. Additionally, various Python packages were installed, including ConfigSpace-0.4.21, AutoSklearn-0.15.0, distro-1.9.0, emcee-3.1.4, liac-arff-2.5.0, pynisher-0.6.4, pyrfr-0.8.3, scikit-learn-0.24.2, and smac-1.2.

## 6 Code
see msfs-autosklearn.ipynb or https://www.kaggle.com/code/gaolang/msfs-autosklearn
