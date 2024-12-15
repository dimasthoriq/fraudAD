# Self-Supervised Learning for Credit Card Fraud Detection
Credit card fraud detection is an important application of machine learning in the financial industry. Despite suffering from an extremely imbalanced class problem and fraudulence labels are often unavailable, most studies treat fraud detection as a supervised classification problem, which often results in a poorly performing and unrealistic detector. We propose a simple tailored method to detect credit card frauds by using a simple semi-supervised training loss and Mahalanobis distance-based fraud score to obtain a robust detection system. The proposed model records a 0.9717 for AUROC, 0.7603 for AUPRC, and 0.0816 on FPR at 95% TPR, outperforming and/or being comparable with the current state-of-the-art and other existing approaches. This study further justifies the compatibility of semi-supervised learning frameworks for credit card fraud detection and shows its potential for realistic real-world business applications.

# Dataset
ULB Machine Learning Group credit card fraud detection [Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

# Experimental Results
Quantitative comparison:

![quantitatives](https://github.com/user-attachments/assets/bab1feba-afa1-47c0-8b3c-a183bd43060c)

Qualitative visualization:

![qualitatives](https://github.com/user-attachments/assets/648814c2-a2a9-4670-889e-d264b4459d12)


# Reference(s)
Awoyemi et. al. Credit card fraud detection using machine learning techniques: A comparative analysis. In IEEE International Conference on Computing Networking and Informatics, 2017. https://ieeexplore.ieee.org/document/8123782

Chen et. al. A simple framework for contrastive learning of visual representations. In ICML, 2020. https://arxiv.org/abs/2002.05709

Forough et. al. Ensemble of deep sequential models for credit card fraud detection. Applied Soft Computing, 99, 2021. https://www.sciencedirect.com/science/article/pii/S1568494620308218

Khatri et. al. Supervised machine learning algorithms for credit card fraud detection: A comparison. In IEEE International Conference on Cloud Computing, Data Science, and Engineering, 2020. https://ieeexplore.ieee.org/document/9057851

Lee et. al. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. In NeurIPS, 2018. https://arxiv.org/abs/1807.03888

Pozzolo et. al. Credit card fraud detection: A realistic modeling and a novel learning strategy. IEEE Transactions on Neural Networks and Learning Systems, 29(8):3784–3797, 2018. https://ieeexplore.ieee.org/document/8038008

Ruff et. al. Deep one-class classification. In ICML, 2018. https://proceedings.mlr.press/v80/ruff18a.html

Ruff et. al. Deep semi-supervised anomaly detection. In ICLR, 2020. https://arxiv.org/abs/1906.02694

Sehwag et. al. Ssd: A unified framework for self-supervised outlier detection. In ICLR, 2021. https://arxiv.org/abs/2103.12051

Zhou et. al. Feature encoding with autoencoders for weakly supervised anomaly detection. IEEE Transactions on Neural Networks and Learning Systems, 33(6):2454–2465, 2022. https://arxiv.org/abs/2105.10500

Zong et. al. Deep autoencoding gaussian mixture model for unsupervised anomaly detection. In ICLR, 2018. https://bzong.github.io/doc/iclr18-dagmm.pdf
