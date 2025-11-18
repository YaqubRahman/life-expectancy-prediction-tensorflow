# Life Expectancy Prediction with Neural Networks

This project implements a **regression neural network** using TensorFlow/Keras to predict **life expectancy** of countries based on health, economic, and social indicators from the **World Health Organization (WHO) Global Health Observatory dataset (2000–2015)**.

The goal is to explore how factors such as immunization, mortality rates, income composition, and human development index contribute to life expectancy, and to build a predictive model that generalizes across countries.

---

## 📊 Dataset

- Source: WHO Global Health Observatory (2000–2015)
- Features include:
  - **Immunization factors**
  - **Mortality factors**
  - **Economic factors**
  - **Social factors**
  - **Other health-related factors**
- Target: **Life expectancy (years)**

---

## ⚙️ Project Workflow

1. **Data Loading & Exploration**

   - Load dataset with `pandas`
   - Inspect columns, data types, and summary statistics
   - Drop non-generalizable identifiers (e.g., `Country`)

2. **Preprocessing**

   - One-hot encode categorical features
   - Split into training and test sets (`train_test_split`)
   - Normalize numerical features with `ColumnTransformer`

3. **Model Building**

   - Sequential neural network with:
     - Input layer (matching feature count)
     - Hidden layer: 64 units, ReLU activation
     - Output layer: 1 unit (regression target)

4. **Compilation**

   - Optimizer: Adam (learning rate = 0.01)
   - Loss: Mean Squared Error (MSE)
   - Metric: Mean Absolute Error (MAE)

5. **Training & Evaluation**
   - Train for 40 epochs, batch size = 1
   - Evaluate on test set
   - Report final MSE and MAE

---

## 📌 Requirements

- Python 3.10 or 3.11 (TensorFlow not yet stable on 3.12)
- Libraries:
  - `pandas`
  - `scikit-learn`
  - `tensorflow`
