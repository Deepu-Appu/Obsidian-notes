### ==RidgeClassifier==

**RidgeClassifier** is a version of **Ridge Regression** that is used for **classification** instead of **prediction**. This classifier first converts binary target to `{-1, 1} and then treats the problem as a regression task, optimizing the same objective as above.

### 📌 ==How RidgeClassifier Works (in simple words):==

1. if you’re doing **binary classification** (like yes/no or 0/1), it **converts your labels** to `-1` and `+1
2. Then it **pretends** it’s a **regression problem**, and tries to predict number close to -1 or +1 using the same math as Ridge Regression.
3. Finally, it **uses the sign** of the result:
    - If prediction is **positive**, its says **class 1**
    - if prediction is **negative**, it says **class0**

### 🤹 ==For Multiclass (more than 2 classes):==

- It treats it like predicting **multiple numbers at once**
- Then it choose the class with the **highest score** (highest predicted value)

It might seem questionable to use a (penalized) Least Square loss to fit a classification model instead of the more traditional logistic or hinge losses. However, in practice, all those models can lead to similar cross-validation scores in term of accuracy or precision/recall, while the penalized least square loss used by the `RidgeClassifier allows for a very different choice of the numerical solvers with distinct computational performance profiles.

This classifier is sometimes referred to as a `Least Square Support Vector Machine` with a linear kernel