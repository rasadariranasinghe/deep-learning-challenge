# Deep Learning Challenge: Alphabet Soup Charity Funding Prediction

## Overview of the Analysis

Alphabet Soup is a nonprofit foundation looking to develop a tool that can help it select applicants for funding with the highest chances of success in their ventures. Using a dataset of over 34,000 organizations that have received funding in the past, this project aims to build a binary classifier using machine learning and neural networks. The goal is to predict whether an applicant will be successful if funded by Alphabet Soup.

This project follows the steps of data preprocessing, model building, optimization, and evaluation to achieve a target prediction accuracy greater than 75%.


## Data Preprocessing

### Target and Features
- **Target Variable:** `IS_SUCCESSFUL` (indicates whether the funding was used effectively)
- **Feature Variables:** 
  - `APPLICATION_TYPE`: Type of application submitted
  - `AFFILIATION`: Industry affiliation
  - `CLASSIFICATION`: Government classification
  - `USE_CASE`: Purpose for funding
  - `ORGANIZATION`: Type of organization
  - `STATUS`: Active status
  - `INCOME_AMT`: Income classification
  - `SPECIAL_CONSIDERATIONS`: Special considerations for funding
  - `ASK_AMT`: Amount requested

### Dropped Variables
- **EIN:** Employer Identification Number (Identification column)
- **NAME:** Organization name (Identification column)

### Steps for Preprocessing
1. Dropped `EIN` and `NAME` columns as they are not relevant features for the prediction model.
2. Encoded categorical variables using `pd.get_dummies()`.
3. Aggregated rare categories in columns with more than 10 unique values into a new category called `Other`.
4. Split the data into feature matrix `X` and target array `y`.
5. Scaled the training and testing datasets using `StandardScaler()`.

## Model Compilation, Training, and Evaluation

### Neural Network Architecture
1. **Input Layer:** Number of input features equals the number of preprocessed columns.
2. **Hidden Layers:**
   - First hidden layer with 16 neurons and the ReLU activation function.
   - Second hidden layer with 12 neurons and the ReLU activation function.
3. **Output Layer:** A single neuron with the Sigmoid activation function for binary classification (successful or not successful).

### Model Compilation
- Loss function: `binary_crossentropy`
- Optimizer: `adam`
- Metrics: Accuracy

### Training and Evaluation
- The model was trained for diiferent number of epochs .
- The performance of the model was evaluated using the test data, yielding the following results:
  - **Test Accuracy:** Approx. 72%

### Saving the Model
- The trained model was saved as `AlphabetSoupCharity.h5` in HDF5 format.

## Model Optimization

To improve the model performance, various optimization methods were applied:

1. **Adjusting the Number of Neurons:**
   - Increased the number of neurons in the hidden layers to improve learning capacity.
   
2. **Adding More Hidden Layers:**
   - Introduced more hidden layers to capture more complexity in the data.
   
3. **Changing Activation Functions:**
   - Experimented with `tanh` activation function for hidden layers to compare performance with `ReLU`.
   
4. **Modifying Epochs and Batch Sizes:**
   - Increased the number of epochs and adjusted batch sizes to find a balance between underfitting and overfitting.

### Results of Optimized Model
- The optimized model achieved an accuracy of approximately **73%**, meeting the target threshold.
- The optimized model was saved as `AlphabetSoupCharity_Optimization.h5`.

## Results

### Data Preprocessing:
- **Target Variable:** `IS_SUCCESSFUL`
- **Features:** All other columns excluding `EIN` and `NAME`
- **Removed Variables:** `EIN`, `NAME`
- **Transformation:** Categorical variables were encoded, and rare categories were aggregated.

### Model Performance:
- **Neurons:** Initially used 16 and 12 neurons in two hidden layers. Increased neurons in the optimization phase.
- **Activation Functions:** `ReLU` and `Sigmoid`, with optimization attempts using `tanh`.
- **Epochs and Layers:** Started with 100 epochs and 2 layers. Optimization used additional hidden layers and epochs.
- **Achieved Overall Accuracy:** Approx. 73% after optimization.
  
### Steps Taken to Improve Performance:
1. Increased neurons in hidden layers.
2. Added a third hidden layer.
3. Tested different activation functions.
4. Experimented with epoch count and batch size.

## Summary and Recommendations

The neural network model achieved an accuracy of 73%, and could not meet the goal for predicting the success of funding applications. Although the performance is satisfactory, there may be alternative models that could provide better results.

### Recommendation for Alternative Models:
1. **Random Forest Classifier:** Could be used to provide feature importance and deal better with categorical variables.
2. **Gradient Boosting:** Might offer better predictive accuracy with its focus on optimizing errors.
3. **Support Vector Machines (SVM):** Could be another powerful classifier for this binary classification problem.

By exploring these models, it is possible to further improve the predictive accuracy and robustness of the classification system.

---



