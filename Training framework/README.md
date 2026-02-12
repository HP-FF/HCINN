# Training and Validation Guide 

This folder demonstrates how to train and validate the HCINN model using the Hoher Sonnblick
3 site as an example.

------------------------------------------------------------------------

## 📌 File Description

### 📊 data/

This folder contains the input datasets:

-   **train.xlsx** -- Training dataset\
-   **val.xlsx** -- Validation dataset\
-   **Total.xlsx** -- Full dataset

⚠️ When applying the model to other sites, ensure the data format
remains consistent.

------------------------------------------------------------------------

### 🧠 Core Model Components

-   **PINN_net.py**\
    Defines the neural network architecture of the HCINN model.

-   **Loss.py**\
    Contains all loss functions, including physics-informed constraints
    and data-driven components.

-   **Sampling.py**\
    Implements sampling strategies for training

-   **Data_Loader.py**\
    Responsible for loading and preprocessing the input datasets.

------------------------------------------------------------------------

### ▶️ Training

-   **train.py**

    This is the core training script.

    To start training:

    ``` bash
    python train.py
    ```

    Training outputs (including trained model weights) will be
    automatically saved to:

        model/

------------------------------------------------------------------------

### 🔍 Validation / Inference

-   **val.py**

    This script performs inference or validation using the trained
    model.

    To run validation:

    ``` bash
    python val.py
    ```

    Output results will be saved to:

        results/

------------------------------------------------------------------------

## 🔄 Applying the Model to Other Sites

You can apply this framework to other stations or regions in two ways:

### 1️⃣ Train from Scratch

-   Replace the dataset in the `data/` folder.

-   Ensure the new dataset follows the same format as the original
    files.

-   Run:

    ``` bash
    python train.py
    ```

This will train a new model for the new site.

------------------------------------------------------------------------

### 2️⃣ Transfer Learning (Recommended)

-   Load a previously trained model from the `model/` folder.
-   Fine-tune it using new site-specific data.

This approach is recommended because:

-   It reduces training time.
-   It improves convergence stability.
-   It leverages previously learned physical representations.

------------------------------------------------------------------------

## ⚙️ Model Flexibility

The HCINN framework is fully modular and flexible.

You are encouraged to:

-   Modify the network architecture in `PINN_net.py`
-   Redesign loss terms in `Loss.py`
-   Implement new sampling strategies
-   Adapt data loading procedures

The source code is designed for extensibility and can be adapted to
various frozen-ground or cryosphere-related applications.

------------------------------------------------------------------------

## 🌍 Final Remarks

We welcome researchers to apply and extend the HCINN model to other
sites and environmental conditions!
