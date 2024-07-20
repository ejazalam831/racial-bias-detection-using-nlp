# Racial Bias Detection in News Articles

## Project Overview
This project is a culmination of our Milestone Project for the Master of Applied Data Science program at the University of Michigan. Our team employed advanced Natural Language Processing (NLP) techniques to detect and quantify racial bias in news articles. The goal of this project is to contribute to fair and inclusive reporting practices by providing a data-driven analysis of how news articles discuss and portray racial dynamics.

By leveraging both supervised and unsupervised machine learning methods, we aim to shed light on the subtle and overt ways racial bias can manifest in media narratives. 

## Data Source
The primary data source is the "Navigating News Narratives: A Media Bias Analysis Dataset" from Zenodo.org. The dataset contains over 3.7 million rows, from which a subset of 41,769 records related to racial bias was selected for analysis.

## Key Features
- Text preprocessing and feature engineering
- Supervised learning models: BERT, LSTM, SVM, SGD
- Unsupervised learning models: LDA for topic modeling, K-means for clustering
- Visualization techniques: word clouds, parallel coordinates plots, silhouette plots

## Main Findings
1. Supervised Learning:
   - BERT model achieved the highest performance with an F1 score of 0.84 ± 0.0154
   - Feature importance analysis revealed key words indicative of bias
   - Sensitivity analysis showed model stability across different hyperparameters

2. Unsupervised Learning:
   - LDA identified 5 distinct topics related to racial discussions
   - K-means clustering determined 17 clusters as optimal
   - Visualization techniques like PCA and t-SNE helped in understanding cluster distributions

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Gensim
- NLTK
- TensorFlow/Keras

## Future Work
- Fine-tune the BERT model for improved performance
- Explore advanced feature engineering techniques
- Analyze a larger dataset for more robust results

## Ethical Considerations
The project addresses potential biases in data and model interpretations, emphasizing the importance of responsible data analysis when dealing with sensitive topics like racial bias.

## Team Information
Amanda Fear, Ejaz Alam, Nikolay Jamgaryan

## Acknowledgments
- Zenodo.org for hosting the "Navigating News Narratives: A Media Bias Analysis Dataset"

For a detailed analysis, please refer to the full project report [here.](https://github.com/ejazalam831/racial_bias_detection_using_nlp/blob/main/Project%20Report/Project%20Report.pdf)
