<h1 style="text-align: center;">An Efficient Task Implementation Modeling Framework with Multi-Stage Feature Selection and AutoML: A Case Study in Forest Fire Risk Prediction</h1>

# Paper: https://www.mdpi.com/2933614

## What is the MSFS-AutoSklearn？
With the advancement of data science, Automated Machine Learning (AutoML) has gained widespread attention due to its ability to lower barriers, save time, and improve efficiency. However, as the dimensionality of data increases, AutoML faces challenges in handling large feature sets. This research proposes an efficient modeling framework that combines a Multi-Stage Feature Selection (MSFS) algorithm with AutoSklearn to address the challenges posed by **high-dimensional data**.

![image](https://github.com/user-attachments/assets/a10b3fc2-6b45-4a30-b930-3d51963f7970)

The MSFS-AutoSklearn framework breaks down the barriers between specialized technology and practical applications by lowering technical thresholds, increasing efficiency, accelerating model deployment, reducing talent demands, and fostering innovation.

## What is the MSFS?
AutoML requires the screening and tuning of a large number of models to enhance the diversity of model selection and increase the probability of successful model operation. This imposes higher demands on the effectiveness and stability of feature subsets. Therefore, it is crucial to construct an FS algorithm that can comprehensively consider **the correlation between features and the target variable**, as well as **the interrelationships between features**, while maintaining **high stability**. This will enable AutoML to efficiently process high-dimensional datasets, thus facilitating the efficient implementation of complex tasks. 
We propose MSFS algorithm. It includes three stages: mutual information gain (MIG), recursive feature elimination with cross-validation (RFECV), and a voting aggregation mechanism, ensuring comprehensive consideration of feature correlation, importance, and stability.

### MIG Filtering: 
Calculates the mutual information gain between features and the target variable to select strongly correlated features.
### RFECV Selection: 
Utilizes Recursive Feature Elimination with Cross-Validation (RFECV) to iteratively select important features based on feature inter-correlations.
### Voting Aggregation Mechanism: 
Determines the final feature subset through a voting mechanism, optimizing the model's stability.
![image](https://github.com/user-attachments/assets/270283b9-5d3b-45fb-aa53-7aedb90d2321)

## Environment Requirements
This study was conducted on the Kaggle platform (https://www.kaggle.com/), a globally renowned platform for data science and machine learning competitions. The experimental environment was configured with a Linux system, running version 5.15.133+. The Python interpreter was version 3.7.12, compiled by GCC 9.4.0. Additionally, various Python packages were installed, including ConfigSpace-0.4.21, AutoSklearn-0.15.0, distro-1.9.0, emcee-3.1.4, liac-arff-2.5.0, pynisher-0.6.4, pyrfr-0.8.3, scikit-learn-0.24.2, and smac-1.2.

## Quick View
You can execute the "msfs-autosklearn.ipynb" notebook to obtain the feature matrix post MSFS processing, which includes only the selected features and the indices of the features that were ultimately chosen.
### Note
1. The input data includes X (Feature matrix, typically a two-dimensional array where each row represents a sample and each column represents a feature) and y (Target variable, usually a one-dimensional array that represents the label or category of each sample)
2. Combine the optimal feature subset obtained from the MSFS algorithm with the labels (target variable) to form a new dataset. Then, split this dataset into 70% for the training set and 30% for the testing set. Use AutoSklearn to validate the effectiveness of the feature subset selected by MSFS, including **correlation**, **capability for handling collinear features**, **the importance of the feature subset**, **stability**, **richness of model selection by AutoSklearn**, and **success rate of model execution**.

## Code
see msfs-autosklearn.ipynb or https://www.kaggle.com/code/gaolang/msfs-autosklearn

## Contact
🙋 If you have any question or want to use the code, please contact ye.su@siat.ac.cn.

🌟 If you find this resource helpful, please consider to star this repository and cite our research.

**Cite: Su, Y.; Zhao, L.; Li, H.; Li, X.; Chen, J.; Ge, Y. An Efficient Task Implementation Modeling Framework with Multi-Stage Feature Selection and AutoML: A Case Study in Forest Fire Risk Prediction. Remote Sens. 2024, 16, 3190. https://doi.org/10.3390/rs16173190**
