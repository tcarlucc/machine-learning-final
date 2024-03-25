# Stock Evaluation Using Pattern Recognition

---
#### CS3540 Machine Learning Final Project
- Nicholas Altland
- Thomas Carlucci
- Julian Henry

*This project uses algorithms to recognize patterns in stock trends and calculate future trends by testing multiple
machine learning models*

---
#### Needed Libraries:
```
pip install numpy
pip install matplotlib
pip install mplfinance
pip install pandas
pip install scikit-learn
pip install yfinance
```
---


#### Code Breakdown

**main.ipynb**:

This file is where a majority of the work is done. It downloads data from yfinance and runs functions from the nuerotrader
file to create trend line values. This information is saved into the training and testing datasets. We generate an image
to showcase our data, then train an SVM model and print our results.

Important function calls:
* treadline_breakdown_dataset: Calculates the resistance and support of the dataset
* fit_upper_trendline: Calculates a regression line to fit the upper trend
* fit_lower_trendline: Calculates the regression line to fit the lower trend

**stock_trend_classifier.ipynb**

This file is not used to generate any current code. It was left over from when we were using data pulled from Kaggle.
It contains several functions used for visualization and data collection using webscrapping, which could be 
useful in the future.

**training.py**

Used to create images for presentation

**treadline_automation.ipynb**

Jupyter Notebook version of .py file of the same name in neurotrader file. Has no function in greater program, only used
for testing preliminary results and attempts to optimize the hard coded gradient descent algorithm

**treadline_calc.py**

Not used in anything? Can delete?

**neurotrader folder**

Programs found in this folder are pulled from several public repositories, 
found [here](https://github.com/neurotrader888?tab=repositories). These programs do the preliminary calculations
that produce the tread lines, such as finding the perceptually important points and adjusting the tread lines to the 
high and low values. These are an invaluable resource, and helped inspire this project. 












