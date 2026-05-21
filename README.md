# 🚀 Space Mission Success Prediction System

> An end-to-end Machine Learning project that predicts whether a space mission will succeed or fail, wrapped in a stunning space-themed Streamlit dashboard.

---

## 📌 Project Overview

This project uses historical space mission data to train a machine learning model that predicts mission outcomes (Success / Failure). Users can interact with a real-time Streamlit dashboard to:

- Input mission parameters and receive instant predictions
- Explore data through interactive visualisations
- Review model performance metrics
- Upload CSVs for bulk predictions
- Download prediction reports

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 Multi-model training | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| 🏆 Auto best-model selection | Picks the highest F1 score automatically |
| ⚙️ Hyperparameter tuning | RandomizedSearchCV with 5-fold stratified cross-validation |
| 🎯 Real-time prediction | Enter mission details → instant result + confidence gauge |
| 📊 Interactive charts | Plotly visualisations with dark space theme |
| 🔬 SHAP explainability | Feature-level prediction explanation |
| 📁 Bulk prediction | Upload CSV → download predictions |
| 📋 Prediction history | Track all predictions in-session |

---

## 🛠️ Technologies

- **Python 3.10+**
- **Streamlit** — web dashboard
- **scikit-learn** — ML models & preprocessing
- **XGBoost** — gradient boosting
- **Plotly** — interactive visualisations
- **SHAP** — model explainability
- **joblib** — model serialisation
- **pandas / numpy** — data processing

---

## 📂 Project Structure

```
space_mission_prediction/
├── data/
│   └── Space_Corrected.csv        # Raw dataset
├── models/
│   ├── best_model.pkl             # Trained model artefact
│   └── model_meta.json            # Metrics & metadata
├── notebooks/
│   └── EDA.ipynb                  # Exploratory analysis (optional)
├── app.py                         # Streamlit dashboard
├── train_model.py                 # Model training pipeline
├── utils.py                       # Helper functions
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/space-mission-predictor.git
cd space_mission_prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place your dataset

Put `Space_Corrected.csv` inside the `data/` folder.

### 5. Train the model

```bash
python train_model.py
```

This will:
- Load & clean the data
- Engineer features
- Train 5 ML models
- Tune the best model with RandomizedSearchCV
- Save `models/best_model.pkl` and `models/model_meta.json`

### 6. Launch the dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📈 Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~67% | ~75% | ~71% |
| Random Forest | ~63% | ~70% | ~65% |
| Gradient Boosting | ~63% | ~71% | ~68% |
| Decision Tree | ~61% | ~68% | ~63% |
| XGBoost | ~58% | ~65% | ~61% |

> Actual results vary with the real Space_Corrected.csv dataset.

---

## 🌐 Deployment

### Streamlit Cloud (Free)

1. Push the project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Add secrets / environment if needed → **Deploy!**

> Make sure `models/best_model.pkl` is committed, or add a startup script to run `train_model.py` before launch.

### HuggingFace Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** as the SDK
3. Push your files to the Space repository
4. HuggingFace auto-deploys on push

### Render

1. Create a **Web Service** on [render.com](https://render.com)
2. Build command: `pip install -r requirements.txt && python train_model.py`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 🔮 Future Improvements

- [ ] Integrate live NASA / SpaceX API data
- [ ] Add LSTM for time-series forecasting
- [ ] Natural-language mission description input (NLP)
- [ ] Multi-class prediction (Success / Partial / Failure / Prelaunch)
- [ ] User authentication & persistent history
- [ ] Automated retraining pipeline (MLOps)

---

## 📸 Screenshots

> Add screenshots of the dashboard here after first run.

---

## 📄 License

MIT License — free to use, modify and distribute.

---

Made with ❤️ by a Space Enthusiast | Powered by scikit-learn & Streamlit 🚀
