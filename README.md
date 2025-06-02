# CSE 151B Competition Spring 2025 - Climate Emulation
## Team Name: BigData (3rd place)
## Overview
This repository contains the model and additional code used for the Kaggle competition. The competition results can be found on [Kaggle](https://www.kaggle.com/competitions/cse151b-spring2025-competition/overview) and all the code in this repository was designed to utilize the competition starter code [repository](https://github.com/salvaRC/cse151b-spring2025-competition). This repository also contains more details about the competition such as the dataset, evaluation metrics, etc. 
## Setup
<ol>
    <li>Clone the starter code [repository](https://github.com/salvaRC/cse151b-spring2025-competition) and follow the setup instructions</li>
    <li>Replace/add corresponding files from thie repository into the starter code repository</li>
    <li>Inside `models.py` add the following import</li?>
    ```
    from src.GRUModel import GRUModel
    ```
    <li>Update `get_model` function input channels to account for added seasonal encoding</li>
    ```
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars) + 1
    ```
    <li>Add conditional to create `GRU` model</li>
    ```
    elif cfg.model.type == "GRU":
        model = GRUModel(**model_kwargs)
    ```
</ol>