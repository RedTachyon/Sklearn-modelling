# yazabi-sklearn-project

This repo contains two modules detailed in the assignment: data\_preprocessing.py and train\_and\_test.py, and several convenience modules for training and evaluating models.

Important functions: 
- get\_all\_data from data\_preprocessing.py allows you to extract all data as a feature matrix.

- run\_search from grid\_search.py allows you to do grid search on a model.
                     
- train\_and\_validate from train\_and\_test.py allows you to train a model and evaluate its performance on the test set
                     
eval\_model.py contains a convenient CLI interface for calling train\_and\_validate. Use it like this: ./eval\_model.py svm -n 
