# Fake-News-Detector

Autohrs: Mingzhi Zhong, Yu Ji, Ziwei Chen and Bokai Li

Abstract

In this project, we integrated two datasets from Chegg to develop a fake news detector using three machine learning methods: logistic regression, support vector machine and fully-connected neural networks.
After a series of tuning, we achieved the highest accuracy of 97\% on the test set using fully-connected neural network. Data and our implementation is available at https://drive.google.com/drive/folders/1UuvHnCrx5AzARPqRPDgG6qLekAIL2By3?usp=sharing

Introduction

The development of the internet has enabled convenient creation and rapid transmission of information. 
However, unrestricted distribution of information leads to the creation of fake news articles that can cause negative impacts on society.
However, detecting fake news manually can be extremely challenging and time-consuming since it involves analyzing content, wording and even visual presentation. 
Thus, fake news detection is an important topic in natural language processing and benefits the society.
In this project, we integrated two datasets from Kaggle into a single dataset containing 10000 real news and 10000 fake news.
Three machine learning methods, including logistic regression, support vector machine and fully-connected neural networks, were applied to analyze data and build a fake news detector.

Prior Work

Early algorithms, such as linguistic approaches and network approaches, seek to make decisions about the authenticity of the news based on other information instead of the content of news alone. In linguistic approaches, syntax analysis, semantic analysis, rhetorical analysis, and other techniques are used to find patterns of deception in the news. In network approaches, network information, such as message metadata, hyperlinks and social network information, are aggregated to identify deceptive contents. However, both methods require extensive prior knowledge in specific domains and manual feature extraction. Thus, automated feature engineering using machine learning and deep learning methods has gained increasing attention in fake news detection. 

However, automated feature engineering has its own pitfall. Many methods extract features that are event-specific, which makes them incapable of detecting deceptive contents on newly arrived events. To address this issue, Wang et al. (2018) built an Event Adversarial Neural Network (EANN) to extract event-invariant features by removing event-specific features captured by the event discriminators. This method outperforms previous state-of-art techniques in fake news detection and shows the power of deep learning method in terms of feature extraction. 

Data Preprocessing

We found two datasets on Chegg. The first one is Getting Real about Fake News\cite{fake news}, which contains contents and headers of over 12,000 fake news articles from real-world websites, identified by the BS Detector Chrome Extension. The second one is All The News\cite{real news}, which contains real news articles from reliable sources, such as the New York Times, CNN, etc. We selected 10,000 examples from each dataset (20,000 in total) and created a vocabulary of size 209,429 and a mapping from each word to its unique index. Using this mapping, we created a fixed-length vector of dimension [1, 209429] for each news article by counting the frequency of each word appeared in both the header and the main content. We used 0 to label the fake example and 1 to label the real example. Finally, we randomly shuffled the whole dataset and splited the dataset into a training set, a validation set and a test set based on a ratio of 8:1:1.


