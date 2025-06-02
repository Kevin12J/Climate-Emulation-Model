# CSE 151B Competition Spring 2025 - Climate Emulation
## Team Name: BigData (3rd place)
## Overview
This repository contains the model and additional code used for the Kaggle competition. The competition results can be found on [Kaggle](https://www.kaggle.com/competitions/cse151b-spring2025-competition/overview) and all the code in this repository was designed to utilize the competition starter code [repository](https://github.com/salvaRC/cse151b-spring2025-competition). This repository also contains more details about the competition such as the dataset, evaluation metrics, etc. 
## Setup
<ol>
    <li>
        Clone the starter code <a href="https://github.com/salvaRC/cse151b-spring2025-competition" target="_blank">repository</a> and follow the setup instructions.
    </li>
    <li>
        Replace/add corresponding files from the <code>StarterCodeUpdates</code> folder into the starter code repository.
    </li>
    <li>
        Inside <code>models.py</code>, add the following import:
        <pre><code>from src.GRUModel import GRUModel</code></pre>
    </li>
    <li>
        Update the <code>get_model</code> function input channels to account for the added seasonal encoding:
        <pre><code>model_kwargs["n_input_channels"] = len(cfg.data.input_vars) + 1</code></pre>
    </li>
    <li>
        Add a conditional to create the GRU model:
        <pre><code>elif cfg.model.type == "GRU":
    model = GRUModel(**model_kwargs)</code></pre>
    </li>
    <li>
        Run <code>main.py</code>
    </li>
</ol>
