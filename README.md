<h1 style="text-align: center;">EEG Signal Quality Analysis by Beacon Biosignals</h1>

<div style="display: flex; justify-content: center; align-items: center; margin-bottom: 50px;">
    <img src="Images/Logo_CentraleSupelec.png" alt="Image 1" style="width: 30%; margin-right: 10px;" />
    <img src="Images/Beacon_Biosignals_Logo.jpg" alt="Image 2" style="width: 30%;" />
</div>

<div style="text-align: center;">
    <a href="#introduction">Introduction</a> •
    <a href="#data-treatment">Data Treament</a> •
    <a href="#feature-extraction">Feature Extraction</a> •
    <a href="#feature-selection">Feature Selection</a> •
    <a href="#data-augmentation">Data Augmentation</a> •
    <a href="#methodology">Methodology</a> •
    <a href="#results">Results</a> •
    <a href="#conclusion">Conclusion</a>
</div>


## Introduction  
This project aims to classify EEG signals into two categories:  
- *Good quality (Class 1)*: Sleep-related patterns are clear without significant artifacts.  
- *Bad quality (Class 0)*: Artifacts obscure the patterns, making interpretation difficult.  

The data, obtained using the Dream Headband at 250 Hz, consists of 2-second labeled windows. The goal is to develop a model for automatic classification of signal quality, inspired by the Kaggle competition: [EEG Signal Quality Analysis](https://www.kaggle.com/competitions/eeg-signal-quality-analysis-by-beacon-biosignals).  


## Data Treatment  
The raw dataset contains 261,755 samples from 5 EEG channels. Signals were filtered using a *Butterworth Band-Pass Filter* (0.1–50 Hz). Each signal was split into 500 data points per 2-second window for analysis.


## Feature Extraction  
Features were divided into four categories, yielding 60 features in total:  
1. *Time Domain Features*: Statistical measures like mean, variance, and skewness.  
2. *Nonlinear Features*: Capture signal complexity and long-term dependencies.  
3. *Frequency Features*: Analyze energy in specific frequency bands (e.g., delta, theta).  
4. *Entropy Features*: Measure randomness and regularity in the signal.  


## Feature Selection  
We used the *Fisher Score* to rank features and create subsets for testing. A Random Forest model with 5-fold cross-validation was used to evaluate feature subsets. Results showed that including all features provided the best Kappa score.


## Data Augmentation  
To reduce overfitting, we applied *bootstrap resampling*, generating 40,000 additional samples. This increased the training dataset to 301,755 samples, ensuring better generalization.


## Methodology  
The primary model used was *Random Forest*, consisting of 50 trees with a maximum depth of 10 to prevent overfitting. Other models, such as XGBoost, logistic regression, SVM, and CNNs, were also tested but did not outperform Random Forest.  

Performance was evaluated using metrics such as *Accuracy* and *Cohen’s Kappa*, with 5-fold cross-validation applied for robust results.  


## Results  
The Random Forest model achieved the following performance metrics:  
- *Train Accuracy*: 90.61% ± 0.27%  
- *Test Accuracy*: 89.84%  
- *OOB Score*: 89.76%  
- *Kappa Score*: 78.89% ± 0.58%  


## Conclusion  
This project provided hands-on experience with preprocessing, feature extraction, and machine learning model development for time-series data. Random Forest proved to be the most effective model, but there is room for improvement in generalization and artifact detection.  

This is the link to our [Project report](Docs/EEG_Signals_analysis_Kaggle_project_PASCHE_POULET.pdf)

