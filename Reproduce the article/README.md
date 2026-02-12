# Reproduction Folder

This folder contains all necessary files to reproduce the results.

------------------------------------------------------------------------

## 📌 File Description

### 📊 Data Files

-   **train.xlsx**\
    Training dataset used for model fitting.

-   **val.xlsx**\
    Validation dataset used for model evaluation.

-   **Total_3m (or Total).xlsx**\
    Full dataset.

------------------------------------------------------------------------

### 🧠 Model Files

-   **PINN_net.py**\
    Defines the neural network architecture used for the site
    model.\
    This file contains the structure of the Physics-Informed Neural
    Network (PINN).

-   **Site (such as Tanggula, ILU).pth**\
    Pre-trained model weights for the site.\
    This file stores the trained parameters and allows direct inference
    without retraining.

------------------------------------------------------------------------

### ▶️ Main Execution Script (Taking the Tanggula site as an exemplar)

-   **Tanggula.py**

    This is the main script for running the trained model.

    To execute:

    ``` bash
    python Tanggula.py
    ```

    The script will:

    -   Load the trained model (`Tanggula.pth`)
    -   Initialize the network structure from `PINN_net.py`
    -   Run inference or evaluation
    -   Generate output results (e.g., SVG figures)

------------------------------------------------------------------------

### 🖼 Output Figures (Taking the Tanggula site as an exemplar)

-   **0m.svg**
-   **0.2m.svg**
-   **1.75m.svg**
-   **2.8m.svg**
-   **3m.svg**

These figures represent model predictions or evaluation results at
different soil depths.

------------------------------------------------------------------------

## 🔄 Reproducibility Workflow

1.  Ensure dependencies are installed (see root `requirements.txt`).

2.  Navigate to the `Tanggula` folder.

3.  Run:

    ``` bash
    python Tanggula.py
    ```

4.  Review generated outputs and figures.

------------------------------------------------------------------------

## 📝 Notes

-   This folder is site-specific and independent of other locations.
-   The pre-trained model (`.pth`) enables direct reproduction of
    results without retraining.
-   If retraining is required, refer to the main `Training framework/`
    directory in the root repository.


