# diabetes-risk-predictor ![Travis Build](https://img.shields.io/travis/nikhilraghava/diabetes-risk-predictor.svg)

The diabetes-risk-predictor is a Keras based neural network that was trained using the Pima Indians diabetes data set. The original owners of this dataset are the National Institute of Diabetes and Digestive and Kidney Diseases (United States). The data set was obtained from the [UCI machine learning datasets database](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes). The contents of the dataset are as follows (list indicies match the column number).

1. Number of times pregnant 
2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test 
3. Diastolic blood pressure (mm Hg) 
4. Triceps skin fold thickness (mm) 
5. 2-Hour serum insulin (mu U/ml) 
6. Body mass index (weight in kg/(height in m)^2) 
7. Diabetes pedigree function 
8. Age (years) 
9. Class variable (0 - non-diabetic or 1 - diabetic) 

Download the dataset as a *csv* file into your working directory, you can also use the dataset in this repository as it is already in the *csv* format. The config file contains the configuration files that have the weights in a *h5* file format and the neural network model itself has been converted to a *JSON* file format to enable faster execution without having to train the model. Running the trained model should give you an accuracy of 99.4%. If you wish to achieve a higher accuracy you can do two things:

+ Play around with the hidden layers
+ Train the neural network for a longer period of time (adjust the epochs)

Doing the above will certainly help you achieve a higher accuracy over time but the accuracies do change as the neural network is trained (because of progressive random changes of the weights). The graph below shows the changes in accuracy levels as the neural network are being trained (you can obtain the graph below by running the *visualizer.py* file). The accuracy levels change because the weights are constantly being updated as the neural network is trained.

![Accuracy Graph](https://cldup.com/THXXlEV_nB.png)
