# MNIST-CNN-Classifier

This repository contains Python scripts for analyzing the performance of `three CNN models` trained on the MNIST dataset. It includes functionality to evaluate and compare models using training and testing metrics.

## Features

- **3 CNN Models**: There are 3 CNN models with a varied combination of activation and evaluation functions. 
- **Comparison Table**: Outputs a clean table comparing training and testing accuracy and loss for each model.
- **Model Histories**: Reads training histories saved as JSON files.
- **Extensible**: Easily add more models by updating the `history_files` dictionary.

## How to Use

**Run main.py**: To train the models.

```bash
python main.py
```
**Run summary.py**: The table will be printed to the console:
```bash
python summary.py
```

```
Model Performance Comparison
==================================================
Model               Metric    Train     Test      
--------------------------------------------------
model_relu          Accuracy  0.9943    0.9877    
                    Loss      0.0178    0.0530    
model_leaky_relu    Accuracy  0.9937    0.9883    
                    Loss      0.0049    0.0083    
model_elu           Accuracy  0.9938    0.9876    
                    Loss      0.0183    0.0504  
```

**Run visualise.py**: There are 5 types of visualisations in the Visualisations folder:
```bash
python visualise.py
```
- Confusion Matrix
- Graphs
- Layer-wise Heatmaps (purely for showcasing)
- Precision-Recall
- ROC (Receiver Operating Characteristic)

![image](https://github.com/user-attachments/assets/f042ba92-23fd-497c-a0e5-9a3a79425f0d)
