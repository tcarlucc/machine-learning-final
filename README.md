# Stock Evaluation Utilizing Algorithmic Pattern Recognition

---
#### CS3540 Machine Learning Final Project
- Nicholas Altland
- Thomas Carlucci
- Julian Henry

*This project uses algorithms to recognize patterns in stock trends and calculate future trends by testing multiple
machine learning models*

---
### INSTRUCTIONS:

Our project is organized into three different sections for the three different machine learning models we created:

1) CNN: This folder contains three different pre-trained CNN's. To run them, make sure to replace the directory with your own code where the sample images are stored. More images can be generated
by "pattern_generator.ipynb" 
2) Flag_NN: This folder contains the machine learning model to detect flags. The data for it is in the home directory.
3) svm_breakout.ipynb: This file is our SVM model. It is in the main directory.

All requirments to run our projects can be found in the "requirments.txt" file. 

---

#### Public Code Acknowledgement:

**neurotrader folder**

Programs found in this folder are pulled from several public repositories, 
found [here](https://github.com/neurotrader888?tab=repositories). These programs do the preliminary calculations
that produce the tread lines, such as finding the perceptually important points and adjusting the tread lines to the 
high and low values. These are an invaluable resource, and helped inspire this project. 


**tradingpatterns**
"Original work by [Preetam Sharma] is licensed under CC BY-NC-SA 4.0 and can be found at (https://github.com/white07S/TradingPatternScanner)."

Programs found in this folder are pulled from the public repository found above. 
As per the liscence agreement, all credit is given to the authors Preetam Sharma and Mateusz Gemzicki, and all edits/changes to their
programs have been marked as such in the comments.

These files were used to detect patterns in our data, which were subsequently turned into images for our CNN.

---



> _Resources:_
> - [matplotlibs documentations](https://matplotlib.org/)
> - [Neurotrader888 Repository](https://github.com/neurotrader888?tab=repositories)
> - [Numpy](https://numpy.org/doc/stable/user/whatisnumpy.html)
> - [Pandas](https://pandas.pydata.org/)
> - [Scikit-Learn](https://scikit-learn.org/stable/)
> - [yfinance project](https://pypi.org/project/yfinance/)
