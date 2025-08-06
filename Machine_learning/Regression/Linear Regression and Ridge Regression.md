## ==Introduction==

Linear Regression is a [Supervised machine learning algorithm](https://www.notion.so/Supervised-And-Unsupervised-Machine-Learning-Model-1f479bd9bdc08062871fd01b0adeae6a?pvs=21) where the predicted output is continuous and has a constant slope. Its used to predict values within a continuous range, (e.g. sales, price) rather than trying to classify them into categories(e.g. cat, dog)

## ==Type of Linear Regression==

### ==Simple Regression==

Simple linear regression uses traditional slope-intercept form, where m and b are the variables our algorithm will try to “_learn “_ to produce the most accurate predictions. $x$ represents our input data and $y$ represents our prediction.

$y = mx + b$

$m = Weigth$

$b = Bais$

### ==Multivariable regression==

A more complex, multi-variable linear equation might look like this, where w represents the coefficients, or weights, our model will try to learn.

$f(x,y,z)=w1x+w2y+w3z$

The variables $x,y,z$ represent the attributes, or distinct pieces of information, we have about each observation. For sales predictions, the attributes might include a company’s advertising spend on radio, TV, and newspapers.

$Sales=w1Radio+w2TV+w3News$

In this example :

- weights: the coefficient for the Radio independent variable. In machine learning we call coefficients _weights._ Usually it is the slope or gradient of the line, indicating how steep the line is
- Bias: the intercept where our line intercepts the y-axis. In machine learning we can call intercepts bias. Bias offsets all predictions that we make.
    
    ![[image.png|image.png]]
    
    ![[image 1.png]]
    

  

## ==Cost function==

Cost function is a mathematical function that measures the difference between a model’s predictions and the actual values.

### ==MSE==

MSE (Mean Square Error) measures the average squared difference between an observation’s actual and predicted values. The output is a single number representing the cost or score, associated with our current set of weights. Our goal is to minimize MSE to improve the accuracy of our model.

$MSE = \frac{1}{N} \sum_{i=1}^{n} (y_i - ŷ)^2$

**where,** $ŷ=mx+b$

```Python
n = 5
sum = 0 
for i in range(n):
	sum += ((actual_value[i] - predicted_value[i])**2)

Mean_Squared_Error = sum/n
```

### ==MAE==

Mean Absolute Error (MAE) is the average distance between the actual and calculated values. It is also known as scale-dependent accuracy as it calculates error in observations taken on the same scale used to predict the accuracy of the machine of the machine learning model.

$\text{MAE} = \frac{1}{n} \sum_{i=1}^n \left| y_i - \hat{y}_i \right|$

```Python
n = 5
sum = 0
# for loop 
for i in range(n):
	sum += abs(actual_values[i] - predicted_vales[i])
	
Mean_absolute_error = sum/n
```

  

### ==Root MSE==

It is used to find the actual error. By taking the root of Mean Square Error.

### ==R2 score==

The R2 score, also known as the coefficient of determination, is a statistical measure used to assess the goodness of fit of a regression model. It measure how well the regression model fits the dependent.

$R^2 = 1 - \frac{SSR}{SST}$

  

$SST(Total Sum of Squares) = \sum_{i=1}^{n} (y_i - \bar{y})^2$

$SSR (Residual Sum of Squares) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

$\bar{y} = Mean of actual value$

### ==R2 score==

The adjusted R2 score is a modified version of the R2 score that adjusted for the number of predictors in a regression model. While the R2 score measures the proportion of variance in the dependent variable that can be explained by the independent variables, it has a limitation: it can increase simply by adding more predictors to the model, even if those predictors do not improve the model’s performance.

$\text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - p - 1} \right)$

where:

- ( R^2 ) is the regular ( R^2 ) score,
- ( n ) is the number of observations (data points),
- ( p ) is the number of predictors (independent variables) in the model.

## ==Assumption of Linear Regression==

→ Linear relationship between input and output.

→ There is no multi-colinearity: In multiple linear regression, the independent variables should not be highly correlated with each other. High multicollinearity can make it difficult to determine the individual effect of each independent variable on the dependent variable.

→ Homoscedasticity: The variance of the residuals is constant across all levels of the independent variable(s). In other words, the spread of the residuals should be roughly the same for all predicted values.

→ Errors forms a normal distribution.

  

### 🧠 ==What is Ordinary Least Squares?==

Imagine you’re trying to draw the **best straight line** through a bunch of dots (data points) on a graph. That line should be as close as possible to all the dots — not too far above or below them.

OLS is a method that helps you find that **best line** by **minimizing the total “mistake” —** how far of the line from each dot. It does this by **squaring the differences** (so negatives don’t cancel positives) and **adding them up —** that’s called the **Residual Sum of Squares (RSS)**.

The lower the RSS, the better the line fits your data.

$\boldsymbol{\beta} = (X^\top X)^{-1} X^\top y$

### 😕 ==What is Multicollinearity?==

Let’s say you’re trying to predict someone’s weight based on their height, waist, and shirt size.

But guess what? Height, waist size, and shirt size are all kind of telling the same story — they are related to each other.

This is where the problem starts.

When your input features (like height, waist, and shirt size) are **very similar to each other**, your model gets **confused**. It doesn’t know **which one to trust more**, and the line it draw become **unstable** and overly sensitive.

Even small changes in your data can cause huge changes in the model’s answers — and that’s bad, because your modal becomes unreliable.

### 🛠 ==What Can You Do About It?==

- Remove one of the similar features (if two are almost saying the same thing, keep just one).
- Combine features (e.g., average them, or use techniques like [PCA](https://www.notion.so/Loss-Function-Cost-Function-23979bd9bdc0806684c2d3d61f426c81?pvs=21))
- Use a smarter version of OLS called **Ridge** or **Lasso** that’s more stable when features are similar.  
      
    

### ==Ridge==

This model tries to draw the **best-fit line or plane** through your data (that’s called regression)

It tries to keep the predictions as **close to the real values as possible** — that’s the **least squares loss part**.

|Parameter|What it means (in plain English)|
|---|---|
|**alpha=1.0**|This is the **strength of the penalty** (regularization). Bigger `alpha` means **more shrinkage** (simpler model). Smaller means the model behaves more like plain linear regression. ➕ More alpha = more protection from overfitting.|
|**fit_intercept=True**|Should the model **learn the intercept** (starting point on the y-axis)? Usually, you want this to be `True`.|
|**copy_X=True**|Should scikit-learn **make a copy of your input data** so it doesn't modify your original data? Set to `True` to keep your input safe.|
|**max_iter=None**|Maximum number of steps the solver (optimizer) will take to find the best solution. Useful if you're using iterative solvers (like "sparse_cg", "sag", etc.).|
|**tol=0.0001**|Tolerance for stopping the solver: If the improvement is **smaller than this**, stop early. (Faster training!)|
|**solver='auto'**|Which **algorithm** should the model use to find the best solution? `'auto'` means scikit-learn will choose the best one based on your data. Options: `'auto'`, `'svd'`, `'cholesky'`, `'sparse_cg'`, `'sag'`, `'saga'`, `'lbfgs'`.|
|**positive=False**|If set to `True`, the model will force all the **coefficients to be positive** (no negative influence allowed). Useful for things like predicting **prices** or **counts**, where negative values don't make sense.|
|**random_state=None**|Controls randomness for reproducible results. Use a number like `random_state=42` if you want consistent output every time you run it.|

### 🧱 ==It also adds a "penalty" for making the model too complex (big weights)==

- This is called **L2 regularization**.
- it’s like saying: “**Don’t trust any one feature too much**”

This helps the model:

- Avoid overfitting (memorizing noise)
- Stay **simple and general** (better for new data)

### ==Non-Negative Least Squares==

when you build a linear regression model, its gives you a formula like:

$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$

The model picks values for β1,β2,… (called coefficients) so that the formula fit your data as closely as possible.

### 🤔 ==But What If…==

Sometimes, these **coefficients** can come of **negative**, which might not make sense in real life.

### 🧠 ==Real-world examples:==

- You’re predicting prices of products →prices shouldn’t be negative.
- You’re modeling **frequency of** something (like how many times something happens) → frequency can’t be negative.
- You’re working with **weights** of ingredients → weight can’t be negative.

### ✅ ==The Solution: Non-Negative Least Squares==

we tell the model:

==“Hey! When you’re picking those coefficients, only allow positive number (or zero). No negatives allowed!”==

In Python, when using ==LinearRegression== ==from== ==scikit-learn====, you can do this by setting:==

```Python
LinearRegression(positive=True)
```

### ==Ordinary Least Squared and Ridge Regression Practical==

```Python
from sklearn import linear_model

# ditting the model
reg = linear_model.LinearRegression()
reg.fit([[0,0], [1,1], [2,2]], [0,1,2])

# co-efficient
reg.coef_

# intercept
reg.intercept_
```

### ==Data Loading and Preparation==

Load the diabetes dataset. For simplicity, we only keep a single feature in the data. Then, we split the data and training and test sets.

```Python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X = X[:, [2]] # Use only one feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=False)
```

### ==**Linear regression model**==

**we create a linear regression model and fit it on the training data. Note that by default, an intercept is added to the model. We can control this behavior by setting the** ==**fit_intercept**== **.**

```Python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(X_train, y_train)
```

### ==**Model evaluation**==

**We evaluate the model's performance on the test set using the mean squared error and the coefficient of determination.**

```undefined
from sklearn.metrics import mean_squared_error, r2_score
y_pred = regressor.predict(X_test)

print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coefficient of determination:{r2_score(y_test, y_pred):.2f}")
```

Mean squared error: 2548.07  
Coefficient of determination:0.47  

### ==**Plotting the results**==

**Finally, we visualize the results on the train and test data.**

```Python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2, figsize=(10,5), sharex=True, sharey=True)

ax[0].scatter(X_train, y_train, label="Train data points")
ax[0].plot(
    X_train,
    regressor.predict(X_train),
    linewidth=3,
    color="tab:orange",
    label="Model Predictions",
)
ax[0].set(xlabel="Feature", ylabel="Target", title="Train set")
ax[0].legend()

ax[1].scatter(X_test, y_test, label="Test data points")
ax[1].plot(
    X_test,
    y_pred,
    linewidth=3,
    color="tab:orange",
    label="Model predictions"
)
ax[1].set(xlabel="Feature", ylabel="Target", title="Test set")
ax[1].legend()
plt.show()
```

![[image 2.png|image 2.png]]

**OLS on this single-feature subset learn a linear function that function that minimizes the mean squared error on the training data. We can see how well (or poorly) it generalize by looking at the R^2 Score and mean squared error on the test set. In higher dimensions, pure OLS often overfits, especially if the data is noisy. Regularization techniques (like Ridge or Lasso) can help reduce that.**

### ==**Ordinary Least Squares and Ridge Regression Variance**==

We sample only two data points, then repeatedly add small Gaussian noise to them and refit both OLS and Ridge. We sample only two data points, then repeatedly add small Gaussian noise to them and refit both OLS and Ridge. We plot each new line to see how much OLS can jump around, whereas Ridge remains more stable thanks to its penalty term.

```Python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
```

### ==Create a Tiny Dataset==

```Python
X_train = np.c_[0.5, 1].T # this give (2,1)
y_train = [0.5, 1] # this gives (1,2)
X_test = np.c_[0,2].T
```

- You have only **two training points**:
    
    $(0.5,0.5) and (1.0, 1.0)$
    
- `np.c_` is a NumPy shorthand for **column stacking**.

### `X_test` is for prediction from 0 to 2 — the line you want to draw.  
  
==Create Two Models==

```Python
classifier = dict(
ols = linear_model.LinearRegression(),
ridge = linear_model.Ridge(alpha=0.1)
)
```

You’re making a dictionary with two models:

- ==ols== : standard regression
- ==ridge== : same as OLS but adds a penalty to keep things stable

### ==Loop through each model==

For each loop (==ols== and and ==ridge==):

```Python
fig, ax = plt.subplots(figsize=(4,3))
```

make a small graph for each model

### ==Repeat training with Noisy Data==

```Python
for _ in range(6):
    this_X = 0.1 * np.random.normal(size=(2,1)) + X_train
```

You add a tiny bit of random noise to the training data to simulate real-world errors or measurement noise.

```Python
clf.fit(this_X, y_train)
ax.plot(X_test, clf.predict(X_test), color="gray")
ax.scatter(this_X, y_train, s=3, c="gray", marker="o", zorder=10)
```

you:

- Fit the model on this noisy data
- Draw the prediction line (gray)
- show the noisy points (small gray dots)

  
✅ This simulates **what happens if your data isn’t perfect**.  
  

### ==**Final Clean Fit**==

```Python
clf.fit(X_train, y_train)
ax.plot(X_test, clf.predict(X_test), linewidth=2, color="blue")
ax.scatter(X_train, y_train, s=30, c="red", marker="+", zorder=10)
```

  

You:

- Fit the model on the original, clean points
- Draw the final prediction line (thicker blue)
- Mark the real points with big red “+” signs

### ==Customize and Show the Plot==

```Python
ax.set_title(name)  # Show "ols" or "ridge"
ax.set_xlim(0, 2)   # X-axis range
ax.set_ylim(0, 1.6) # Y-axis range
ax.set_xlabel("X")
ax.set_ylabel("y")
fig.tight_layout()
plt.show()
```

![[image 3.png]]

![[image 4.png]]

|`alpha` value|Effect||
|---|---|---|
|`0`|No penalty → behaves like **Ordinary Least Squares** (OLS)||
|Small (e.g. `0.1`)|Small penalty → coefficients shrink slightly||
|Medium (e.g. `1`)|Moderate penalty → more shrinkage||
|Large (e.g. `10`, `100`)|Strong penalty → more simplification, may **underfit**||

###   
🔍 ==How to Choose the Best== ==`alpha`====?==

```Python
from sklearn.linear_model import RidgeCV

alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)

print("Best alpha:", ridge_cv.alpha_)
```

This will train the model using different alpha values and choose the one that works best on validation sets.  
  

### 🧠 ==what's a== ==**solver**====?==

When Ridge regression is training a model, it needs to do some math to find the best line (i.e., best coefficients).

A solver is just the method or algorithm it uses to do that math.

There are several solvers available, and some are faster or better depending on your data.  
  

### ==What does== ==`solver="auto"`== ==mean?==

If you don’t know which solver to choose (and most people don’t need to!), you can just write:

```Python
Ridge(solver="auto")
```

This tells Ridge:

> “you figure out the best way to solve this problem for me.”

It will then check your data, and pick one of the following methods:

- ==“lbfgs”== - good for medium-sized dense data
- ==“cholesky”== ==- fast when the data is not sparse==
- ==“sparse_cg”== - good for very large or sparse datasets

|Solver|Best For|Description (Layman Terms)|
|---|---|---|
|`'svd'`|Small datasets|Uses **Singular Value Decomposition** (SVD), which is very **accurate** but not the fastest. Great when your data has **fewer samples/features**.|
|`'cholesky'`|Small to medium datasets|Solves the equation using **Cholesky decomposition**. Fast and memory-efficient if the data isn’t too large.|
|`'sparse_cg'`|Large sparse data|Uses **Conjugate Gradient** algorithm. Best for **large datasets** that are **sparse** (lots of zeros).|
|`'sag'`|Large datasets|**Stochastic Average Gradient**. Very fast for **large datasets**, especially when features are dense (not sparse). Needs data to be **scaled**.|
|`'saga'`|Large datasets + L1 regularization|Like `sag` but also supports **L1 regularization** (Lasso). Works on both **sparse and dense** data. Needs data to be **scaled** (**standardScaler**).|
|`'lbfgs'`|Small/medium datasets, many outputs|A **quasi-Newton optimizer**. Handles **multitarget** regression and is **robust**, but a bit slower.|

### 📋 ==How does Ridge pick?==

Ridge goes through a checklist like this (simplified):

1. Is the dataset large and sparse? → Use ==“sparse_cg”==
2. Is the dataset small or dense? → Use ==“cholesky”== or ==“lbfs”==

It picks the first solver that matches your data condition.

### ✅ ==In Simple Terms:==

> Ridge looks at your data and says,  
> ”Hmm… based on what I see, this method will solve it best,”  
> — and automatically picks the most efficient solver for you.  

So you don’t have to worry about the math engine — Ridge handles it.

|   |   |
|---|---|
|**Solver**|**Condition**|
|‘lbfgs’|The `positive=True` option is specified.|
|‘cholesky’|The input array X is not sparse.|
|‘sparse_cg’|None of the above conditions are fulfilled.|

###   
🧠 ==**Understanding Collinearity and Ridge Regression (In Simple Terms)**==

  

In this example, we are showing how **Ridge Regression** helps when some features (columns) in your data are **too similar** to each other — a problem called **collinearity**.

### 🛠️ ==How Ridge Regression Helps==

Ridge Regression fixes this by adding a penalty when the model tries to assign large numbers to the features.

we can control how this penalty is using a value called alpha (α):

- If **alpha is small**:
    
    ➤ The penalty is light, and the model behaves like normal linear regression. But this can make the coefficients **unstable**
    

### ==**Plot Ridge coefficients as a function of the regularization**==

we want to see how the Ridge Regression coefficients changes as we adjust the penalty value ( ==alpha== ).

This helps us understand:

- How sensitive the model is to regularization
- How the model becomes simpler or **more stable** as we increase the penalty

```Python
X = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
```

This creates a special kind of matrix called a Hilbert matrix of size 10x10

### 🧠 ==What’s a Hilbert Matrix?==

- A Hilbert matrix is a square matrix where each entry is:
    
    $X[i][j] = \frac{1}{i + j + 1}$
    
- it’s symmetric and its values get smaller as you move right or down.

```Python
y = np.ones(10)
```

This just creates a target/output ==y====:==

```Python
[1.0,1.0,1.0,..,1.0]
```

So every input row of the matrix ==X== ==is mapped to the same output== ==1.0====.==

```Python
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
```

> We want to try 200 different alpha values (penalty strengths)

Think of ==alpha== ==as a== ==**knob**== ==that controls how much the model avoids big weights.==  
  

> ✅ What is ==`np.logspace()`==?  
> ==`np.logspace()`== ==is a Numpy function used to create numbers that are evenly== ==**spaced on a log scale**== ==(logarithmic scale)====`np.logspace(start, stop, num)`====  
> Example:  
>   
> ====`np.logspace(-2, 2, 5)`====  
> This means:  
> give 5 numbers from  
> ====$10^{-2}$== ==to== ==$10^{2}$====  
>   
> ====output:  
>   
> ====`array([ 0.01, 0.1, 1.0, 10.0, 100.0 ])`==

==`alphas = np.logspace(-10, -2, n_alphas)`==

> Create 200 values of alpha between $10^{-10}$ and $10^{-2}$, **spaced logarithmically**

so:

- You’ll try **tiny alphas (very weak penalty)**
- Up to moderate **alphas (stronger penalty)**

This range is good for testing how sensitive the model is:

> ==🛠️== ==`ridge = linear_model.Ridge(alpha=a, fit_intercept=False)`====  
>   
> ====Create a Ridge Regression model using the current alpha.==

- ==alpha=a== ==: Set the penalty to the current value==
- ==fit_intercept=False== : We don’t want the model to learn an intercept (bias); we’re only testing coefficient shrinkage.

### ==Display results==

```Python
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge Coefficients vs Regularization Strength (alpha)")
plt.axis("tight")
plt.legend(
    [f"Feature {i + 1}" for i in range(X.shape[1])], loc="best", fontsize="small"
)
plt.show()
```

> ==ax = plt.gca()  
>   
> ====gca → “Get Current Axes”====  
>   
> ====Get the current plot “axis” (canvas where we draw the chart).  
> You can now customized or add things to it using  
> ====ax== ==.==

> ==ax.plot(alphas, coefs)==  
> Plot how each **coefficient** (weight) changes as you change ==alpha==.

- ==alphas== ==: The x-axis — different values of the penalty.==
- ==coefs== : The y-axis — the corresponding model weight for each feature.

Each line in the charts shows one **feature’s coefficient**.

> ==ax.set_xscale("log")==  
> Make the x-axis (alpha values) logarithmic scale, not linear.  

- Beacuse ==alphas== go from very small (like $10^{-10}$) to larger values (like $10^{-2}$).
- A log scale spread them out **better** for visualization.

> ==ax.set_xlim(ax.get_xlim()[::-1])  
>   
> ====This reverses the x-axis.==

So the plot starts with **small alphas on the right** and **large** on the right and large alphas on the left — this is a common style in regularization plots, showing:

- Left = **strong penalty (coefficients shrink)**
- Right = **no penalty** (coefficients grow wild)

> ==plt.xlabel("alpha")  
>   
> ====Label the x-axis as “====**alpha**====” (the regularization strength)==

> ==plt.ylabel("weights")  
>   
> ====Label the y-axis as “====**weight” —**== ==the Ridge model’s coefficients==

> ==plt.title("Ridge Coefficients vs Regularization Strength (alpha)")  
>   
> ====Add a title to the plot to explain what it’s showing==

> ==plt.axis("tight")==  
> Automictically adjust the plot to fit everything snugly (no extra space)  

> ==plt.legend([...])==  
> Add a **legend** showing which line is which feature.

This part:

```Python
[f"Feature {i + 1}" for i in range(X.shape[1])]
```

is just a like:

```Python
["Feature 1", "Feature 2", ..., "Feature 10"]
```

so you know which line represents which input variable.

![[image 5.png]]