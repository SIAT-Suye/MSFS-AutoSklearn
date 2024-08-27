# Paper Title：An Efficient Task Implementation Modeling Framework with Multi-Stage Feature Selection and AutoML: A Case Study in Forest Fire Risk Prediction

## 1 Introduction
With the advancement of data science, Automated Machine Learning (AutoML) has gained widespread attention due to its ability to lower barriers, save time, and improve efficiency. However, as the dimensionality of data increases, AutoML faces challenges in handling large feature sets. This research proposes an efficient modeling framework that combines a Multi-Stage Feature Selection (MSFS) algorithm with AutoSklearn to address the challenges posed by high-dimensional data.

## 2 Feature Selection Algorithm
The proposed MSFS algorithm consists of three stages:

### 2.1 MIG Filtering: 
Calculates the mutual information gain between features and the target variable to select strongly correlated features.
### 2.2 RFECV Selection: 
Utilizes Recursive Feature Elimination with Cross-Validation (RFECV) to iteratively select important features based on feature inter-correlations.
### 2.3 Voting Aggregation Mechanism: 
Determines the final feature subset through a voting mechanism, optimizing the model's stability.
![image](https://github.com/user-attachments/assets/270283b9-5d3b-45fb-aa53-7aedb90d2321)

## 3 Environment Requirements
This study was conducted on the Kaggle platform (https://www.kaggle.com/), a globally renowned platform for data science and machine learning competitions. The experimental environment was configured with a Linux system, running version 5.15.133+. The Python interpreter was version 3.7.12, compiled by GCC 9.4.0. Additionally, various Python packages were installed, including ConfigSpace-0.4.21, AutoSklearn-0.15.0, distro-1.9.0, emcee-3.1.4, liac-arff-2.5.0, pynisher-0.6.4, pyrfr-0.8.3, scikit-learn-0.24.2, and smac-1.2.

## 4 How the MSFS algorithm uses the dataset ？
Input：
X: Feature matrix, typically a two-dimensional array where each row represents a sample and each column represents a feature.

y: Target variable, usually a one-dimensional array that represents the label or category of each sample.

k_mi：The number of features to select using mutual information gain. Specifies how many features to select.

estimator：The estimator (model) used for RFECV. It can be any model that conforms to the Scikit-learn interface, such as decision trees, random forests, etc.

cv（optional, default is 5）：The number of folds for cross-validation. Specifies the number of folds to use in RFECV.

n_repeats（optional, default is 10）：The number of repetitions for the feature selection process. Increases the stability and reliability of the results by running the feature selection multiple times.

Output：
The feature matrix after feature selection, containing only the selected features and the indices of the features that were ultimately selected.

## 5 MSFS Algorithm Training and Testing Approach
Training：The MSFS algorithm iteratively performs the feature selection process. In each iteration, it performs two steps: MIG filtering (stage 1) and RFECV selection (stage 2), resulting in a set of selected features. However, RFECV filtering is only executed if the number of features chosen in stage 1 exceeds one; otherwise, the algorithm proceeds to the next iteration by saving the indices of the stage 1 feature subset. This cycle continues until n_repeats iterations are completed, with the indices of the feature subsets from each iteration being compiled in a list. Ultimately, the optimal feature subset is identified through a voting aggregation mechanism (stage 3).

Testing：Combine the optimal feature subset obtained from the MSFS algorithm with the labels (target variable) to form a new dataset. Then, split this dataset into 70% for the training set and 30% for the testing set. Use AutoSklearn to validate the effectiveness of the feature subset selected by MSFS, including correlation, importance of the feature subset, stability, richness of model selection by AutoSklearn, and success rate of model execution.

## 6 Code
see msfs-autosklearn.ipynb or https://www.kaggle.com/code/gaolang/msfs-autosklearn

## 7 Citation
