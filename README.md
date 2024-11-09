# Attention Mechanism Empowered Dual Active Bridge Converter Performance Modeling with Enhanced Interpretability and Lighter Data
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)
  [![Static Badge](https://img.shields.io/badge/TabNet-1?style=for-the-badge&logo=pytorch&logoColor=%23EE4C2C)](https://dreamquark-ai.github.io/tabnet/generated_docs/README.html)
  <img height="31" width="31" src="./image/SHAP.png" /> [![Static Badge](https://img.shields.io/badge/SHAP-1?color=FFDEB4)](https://shap.readthedocs.io/en/latest/index.html)


## Description
***
**DDM-ETS**, a novel data-driven modeling approach for enhancing accuracy and interpretability in power converter modeling, specifically tailored for **dual active bridge (DAB) converters**. It starts with **exploratory data analysis**, **TabNet** for surrogate model building, and followed by **Shapley Additive Explanations (SHAP)** for explainability analysis.

### I.Comparison between existing AI data-driven approaches and DDM-ETS
Current AI data-driven modeling approaches have notable room for enhancement due to data quality issues and the use of algorithms with poor accuracy and interpretability.To address existing limitations, a novel data-driven modeling with exploratory data analysis and TabNet **(DDM-ETS)** is proposed for performance modeling. The first stage is to remove outliers and invalid values and explore the relationships between the features of the data through exploratory data analysis. In the  second stage, the processed data and TabNet are utilized to train  data-driven models for ZVS and efficiency. In the final stage, SHAP, a post-hoc model explanation method, combined with  mask, helps to further understand how each feature influences the model predictions.
<div align="center">
  <img src="./image/DDM-ETS.png" alt="MMD-ETS">
</div>

### II.Architecture of TabNet
TabNet is composed of feature transformer, attentive transformer and mask.The feature transformer is used to convert the input features (P, V2, Din) into an embedded space. The attentive transformer uses self-attentive mechanism to select the most important features and pass to mask. The mask implements sparse selection of features, allowing the model to dynamically focus on the most important features for the current decision. Additionally, by aggregating these masks, a comprehensive understanding of the global importance of features can be obtained. 
<div align="center">
  <img src="./image/TabNet.png" alt="TabNet">
</div>

## Project description
***
The code of this project is open source, and since the dataset will be used for further scientific research, it is not possible to disclose all of the data, and we only provide a part of the example dataset that has been filtered and deprivatized for privacy and data security reasons. This example dataset is intended to help users understand the code running and model training process, but differs from the full dataset actually used. We encourage users to train with their own dataset and make the necessary hyperparameter adjustments to get the best results.
