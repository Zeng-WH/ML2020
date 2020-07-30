## ML Lecture 2: Where does the error come from

### 1. Bias vs Variance

<img src=".\bias.PNG" style="zoom:80%;" />

### 2. What to do with large bias?

- **Diagnosis:**
  - If your model can not even fit the training examples, then you have large bias.
  - If you can fit the training data, but large error on testing data, then you probably have large variance.

- **For bias, redesign your model:**
  - Add more features as input
  - A more complex model

- **For variance:**
  - More data: Very effective, but not always practical.
  - Regularization.

### 3. Cross Validation

将数据集分成Training set, Validation set, Testing set之后，利用Validation set挑出合适的模型之后，不建议Using the results of public testing data to tune your model. You are making public set better than private set.

### 4. N-fold Cross Validation

<img src=".\nfold.PNG" alt="nfold" style="zoom:80%;" />

