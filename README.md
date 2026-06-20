# Quickdraw Test

Repository created to learn GitHub workflows.

# 🌸 Iris Predictor

A machine learning application that classifies iris flowers into three species using a Random Forest classifier. Built with Streamlit for an interactive web interface.

## 📋 Overview

This project demonstrates how to build, train, and deploy a machine learning model using Python. It uses the classic Iris dataset to predict which species an iris flower belongs to based on four measurements: sepal length, sepal width, petal length, and petal width.

## 🎯 Features

- **Interactive Web Interface**: Built with [Streamlit](https://streamlit.io/) for easy interaction
- **Machine Learning Model**: Random Forest Classifier trained on the Iris dataset
- **Real-time Predictions**: Input flower measurements and get instant predictions
- **Confidence Visualization**: See the model's confidence level for each species
- **Model Information**: View accuracy metrics and feature importance
- **Species Guide**: Learn about the characteristics of each iris species

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Dataset**: Iris Flower Dataset (150 samples)
- **Features**: 4 measurements
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Classes**: 3 species
  - Setosa
  - Versicolor
  - Virginica
- **Accuracy**: ~96.7%

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Haythemidi02/iris-predictor.git
   cd iris-predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Train the model** (if not already trained)
   ```bash
   python train_model.py
   ```

2. **Start the Streamlit app**
   ```bash
   streamlit run apppp.py
   ```

3. **Open your browser** and navigate to `http://localhost:8501`

## 📦 Requirements

- `streamlit>=1.31.0` - Web framework for data apps
- `scikit-learn>=1.4.0` - Machine learning library
- `pandas>=2.2.0` - Data manipulation
- `numpy>=1.26.3` - Numerical computing
- `matplotlib>=3.8.2` - Data visualization
- `seaborn>=0.13.1` - Statistical data visualization
- `joblib>=1.3.2` - Model serialization

## 📁 Project Structure

```
iris-predictor/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── train_model.py         # Model training script
├── apppp.py              # Streamlit application
├── models/               # Trained model storage
│   └── iris_model.pkl    # Serialized Random Forest model
├── scaler.pkl            # Fitted StandardScaler
└── .devcontainer/        # Development container configuration
    └── devcontainer.json
```

## 🔧 How It Works

### Training Phase (`train_model.py`)
1. Loads the Iris dataset
2. Scales features using StandardScaler
3. Splits data into training (80%) and testing (20%) sets
4. Trains a Random Forest model with optimized hyperparameters
5. Evaluates performance using cross-validation and test accuracy
6. Saves the model and scaler for later use

### Prediction Phase (`apppp.py`)
1. Loads the trained model and scaler
2. Accepts user input for iris flower measurements
3. Scales the input using the saved scaler
4. Makes predictions and calculates confidence scores
5. Displays results with visualizations and explanations

## 📈 Model Performance

The Random Forest model achieves approximately **96.7% accuracy** on the test set. The model provides:
- Class predictions (which species the flower belongs to)
- Probability scores for each species
- Feature importance rankings

## 🌺 Iris Species

- **Setosa**: Smallest flowers with short, wide petals. Very distinct from other species.
- **Versicolor**: Medium-sized flowers with moderate petal length and width. Can overlap with Virginica.
- **Virginica**: Largest flowers with long, narrow petals. Often confused with Versicolor.

## 💡 Usage Tips

1. Enter measurements in centimeters for accurate predictions
2. The app will show your prediction and confidence level
3. Review the model's accuracy metrics in the sidebar
4. Use the species guide to learn about each type

## 🐳 Development Container

This project includes a `.devcontainer` configuration for VS Code development containers:

```bash
# In VS Code, use "Reopen in Container" to develop in an isolated environment
# The container automatically installs dependencies and runs the app
```

## 📚 Resources

- [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) - Wikipedia
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Created by [Haythemidi02](https://github.com/Haythemidi02)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs by opening an issue
- Suggest improvements or new features
- Submit pull requests with enhancements

## 📞 Support

If you have questions or issues, please create an issue on the [GitHub repository](https://github.com/Haythemidi02/iris-predictor/issues).

---

**Happy Predicting! 🎉**
