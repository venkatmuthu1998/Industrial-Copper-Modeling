# 🔩 Industrial Copper Price & Status Prediction

An end-to-end machine learning application built using Streamlit to predict:

📈 Selling Price of Industrial Copper
✅❌ Deal Status (Won / Lost)

The application loads pre-trained ML models, takes user inputs through a web UI, and produces 
real-time predictions for business decision support.


# 🚀 Features

🔍 Deal Status Prediction
    --Classifies a deal as Won or Lost
💰 Selling Price Prediction
     --Predicts copper selling price using regression
📅 Automatic date feature extraction
⚡ Fast inference using cached ML pipelines
🎛 Interactive Streamlit UI with sidebar navigation


# 🧠 Machine Learning Models

1.Classification Model
  --Predicts deal status (Won / Lost)

2.Regression Model
  --Predicts log-transformed selling price
  --Output is exponentiated to get actual price

Both models are loaded from serialized .pkl files.


# 🛠 Tech Stack

Python
Streamlit
NumPy
Scikit-learn (trained models)
Pickle
Pandas (during training)


# 📁 Project Structure

├── Copper.py.py                  # Streamlit application
├── models/
│   ├── Classification_model.pkl
│   └── Regression_Model.pkl
├── Copper_Set.csv                # Training dataset
├── Industrial_Copper.ipynb       # Model training & EDA
├── Industrial_Copper_Colab.csv   # Cleaned dataset
└── README.md


# 🔄 Application Workflow

1.Predict Status
  --User inputs deal-related features
  --Dates converted into day/month/year
  --Classification model predicts Won / Lost

2.Predict Selling Price
  --User inputs order details
  --Regression model predicts log-price
  --Log value converted back to actual price


# 🏗 Architecture Diagram (Logical)                     


┌─────────────────────────┐
│  Copper Dataset (CSV)   │
│  - Industrial_Copper    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Data Preprocessing      │
│ - Cleaning              │
│ - Log Transformations   │
│ - Feature Engineering   │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Model Training (Offline)│
│ - Classification Model  │
│ - Regression Model      │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Pickle Serialized Models│
│ (.pkl files)            │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Streamlit Application   │
│ - User Inputs           │
│ - Date Feature Extract  │
│ - Model Inference       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Prediction Output       │
│ - Deal Status           │
│ - Selling Price         │
└─────────────────────────┘



# 🔁 Execution Flow Diagram


Start App
   │
   ▼
Load Streamlit UI
   │
   ▼
Load Cached ML Models
   │
   ▼
User Selects Prediction Type
   │
   ├── Predict Status
   │   └── Classification Model
   │
   └── Predict Selling Price
       └── Regression Model
   │
   ▼
Date Feature Extraction
   │
   ▼
Model Prediction
   │
   ▼
Display Result in UI


# 👤 Author

Venkatesan M

