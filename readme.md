# Stock Price Predictor with Machine Learning
*🌟 Leveraging Machine Learning for Equity and Index Price Predictions 🌟*

This project explores the application of machine learning to predict trends in the financial market. Using datasets for both equities and indices, this project processes data, calculates technical indicators, and trains models to forecast stock price movements. Developed with Python and using libraries like yfinance, pandas, and xgboost, the Stock Price Predictor aims to provide valuable insights for stock market participants.

# Table of Contents-

Project Overview

Uses and Significance

How the Project Works

Installation

Directory Structure

Data Collection and Processing

Model Training

Prediction and Evaluation

Future Plans

Acknowledgments


# Project Overview -

The Stock Price Predictor focuses on forecasting stock prices by processing historical data and calculating technical indicators to capture financial market patterns. Leveraging XGBoost regression models with hyperparameter tuning, the project aims to enhance prediction accuracy for both equity and index data.

**Key Features-**

*Comprehensive Indicator Analysis*: Calculates key indicators such as SMA, EMA, RSI, MACD, and more.

*Efficient Model Training*: Optimized model training with hyperparameter tuning using GridSearchCV.

*Flexible Data Processing*: Capable of handling both equity and index datasets with customizable intervals.


# Uses and Significance -

This system serves as a powerful tool for analysts, investors, and researchers seeking to understand and anticipate market behavior. By combining multiple technical indicators and using robust machine learning techniques, this project provides insights that can support trading and investment decision-making.


# How the Project Works -

**The Stock Price Predictor is organized into three main components:**

*Data Collection*: Historical data is fetched using yfinance for specified intervals, enabling flexibility in market analysis periods.

*Data Processing and Indicator Calculation*: Multiple technical indicators like - SMA, EMA, RSI(Relative Strength Index), CCI(Commodity Channel Index), Momentum, Super Trend, ADX(Average Directional Index),VWAP(Volume Weighted Average Price), Williams %R, OBV(On Balance Volume), ATR(Average True Range), Stochastic Indicator, Bollinger Bands and MACD are calculated to highlight market patterns.

*Model Training and Prediction*: XGBoost models are trained on the processed data to predict future price trends.


# Installation -

**Prerequisites-**

Ensure you have Python 3.7+ installed. Install the required libraries with:

pip install yfinance pandas numpy scikit-learn xgboost

**Required Libraries-**

*yfinance*: For fetching historical financial data.

*pandas*: For data manipulation and processing.

*numpy*: For numerical operations.

*scikit-learn*: For model training and evaluation.

*xgboost*: For advanced regression modeling.


# Directory Structure -

*Organize your project directory as follows:*

├── data                    (Folder for equity and index data files)

├── equities.ipynb          (Notebook for processing and predicting equity data)

├── indices.ipynb           (Notebook for processing and predicting index data)

└── README.md


# Data Collection and Processing -

**Data Collection-**

Data is collected using the fetch_stock_data() function, which retrieves historical stock data for specified intervals and periods via yfinance. It supports multiple intervals for flexible analysis.

**Data Processing-**

The process_stock_data() function prepares the data by calculating technical indicators like SMA, EMA, RSI, MACD, and Bollinger Bands, and many other, providing insights into market trends to inform the model.


# Model Training-

*To train the prediction model:*

Load and preprocess the data, splitting it into training and testing sets using train_test_split.

Train the XGBoost model with hyperparameter tuning using GridSearchCV to optimize model performance.

**Model Architecture-**

The model utilizes XGBoost regression, a technique well-suited for time-series prediction in financial data. Using gradient boosting, it minimizes errors to improve predictive accuracy.


# Prediction and Evaluation -

Open the respective notebook (equities.ipynb or indices.ipynb) to run predictions. Model performance is evaluated using metrics such as the R2 Score, allowing adjustments to enhance accuracy.


# Future Plans -

Expanding Functionality: Future improvements include testing additional machine learning models and incorporating real-time data for live predictions.


# Acknowledgments -

This project was inspired by the potential of machine learning to provide actionable insights into financial markets.

