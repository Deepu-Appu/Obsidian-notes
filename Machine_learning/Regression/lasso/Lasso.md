**Lasso** is a type of linear **regression** — a method that tries to **predict** to **a result using several input features** (like predicting house prices from size, location, etc.)

But unlike **normal regression, Lasso has a special power**:

> 🔍 It **automatically removes features** that aren’t very important.

### 📉 ==How?==

Lasso does this by **shrinking** the coefficients (the “weight” assigned to each feature). If a feature isn’t useful, Lasso can **shrink its weight all the way to zero** — effectively removing it from the model.

> Think of it like a coach picking only the most valuable players for a game and **benching the rest**.

### 📦 Why is this useful?

- ✅ Makes your model **simpler and faster**
- ✅ Helps **avoid overfitting**
- ✅ Great when you have **a lot of features** and want to find which ones truly matter

### 📸 ==What is "compressed sensing"?==

Imagine you want to **rebuild an image (like an X-ray)**, but only have a few blurry parts. Compressed sensing uses math tricks like Lasso to reconstruct the **full image** using the most important **pieces of information** — even when some data is missing.

### 🧮 ==Mathematic formulae==

$\min_{w} { \frac{1}{2n_{\text{samples}}} ||X w - y||_2 ^ 2 + \alpha ||w||_1}$

### ==📌 1.== ==Prediction Error Term:==

This is the objective function that **Lasso Regression** tries to minimize. It has two parts:

- Measures how well your model’s predictions match the actual values.
- ==$Xw$== ==is the predicted value (input and weight).==
- y is the actual value.
- ==$||Xw - y||^2$== ==means== ==**squared error**==
- The ==1/(2n)== ==part is just for mathematical convenience (averaging over all samples)==

### 📌 ==2.== ==Penalty Term (L1 regularization):==

$\alpha \left\lVert w \right\rVert_1$

- ==$\left\lVert w \right\rVert_1$== means the **sum of absolute values of the coefficients**.
- ==$a$== is a **regularization parameter** (a positive number you choose).
    
    It controls **how much penalty** you want to apply
    

### 📌 ==Why is this useful?==

This is the **L1 regularization** term used in **Lasso Regression**

- Adds a penalty to large weights,
- Helps **prevent overfitting**
- And often makes some weights **exactly zero,** which leads to **sparse mode** (feature selection).

### 🧮 ==Lasso Regression Loss Function==

The **loss function** is what the model tries to **minimize** during training.

==**Lasso Loss :**== ==$\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \left\lVert w \right\rVert_1$==

- ==**MSE**== = Mean Squared Error (how wrong the predictions are)

### 🧠 ==What does this mean?==

1. **MSE** says:
    
    👉 “Fit the data accurately as possible”
    
2. **L1 penalty** says:
    
    👉 “Use fewer features — shrinks unnecessary weights towards zero”
    
3. $a$ controls the balance:
    - If $a$ = 0: model becomes regular linear regression (no penalty)
    - if $a$ is too big: model might drop too many features and underfit

|Model|Uses Linear Equation?|Regularization|Effect|
|---|---|---|---|
|Linear|✅ Yes|❌ None|Fits all coefficients freely|
|Ridge (L2)|✅ Yes|✅ Adds α∥w∥22\alpha \\|w\\|_2^2|Shrinks large coefficients|
|Lasso (L1)|✅ Yes|✅ Adds α∥w∥1\alpha \\|w\\|_1|Shrinks and can zero out some coefficients (feature selection)|

### 🔧 ==What is Coordinate Descent (Layman’s Explanation)?==

Imagine you’re trying to find the lowest point in a valley (i.e., **minimize a function**). But instead of moving freely in all directions, you takes steps **along one axis at a time** — like only walking **east-west**, **then north-south,** then back **east-west**, and so on — adjusting one coordinate (weight) while holding others fixed.

### 📌 ==Why Coordinate Descent for Lasso?==

- At each step, it optimizes one coefficient ==$w_j$== at a time,
- Applies a soft **thresholding rule** (pushes small weights toward zero),
- Efficiently finds a **sparse solution** — many weights become **exactly zero**, which is the **key feature of lasso**.

### ==Compressive sensing: tomography reconstruction with L1 prior (Lasso)==

we’re trying to **rebuild a picture** (like a black-and-white X-ray) — but instead of seeing the whole picture at once, you only get a **few shadowy slices** taken from different angles.

This process of taking **angled shadow snapshots** to rebuild an image is called **Tomography**, and it’s what machine like **CT scanners** do in hospitals.

### 🧩 ==The Problem==

To reconstruct the full **image perfectly**, you’d ideally need lots of projections (those angled shadows).

But in real life:

- Taking too many projection is **slow**, **costly**, or **dangerous** (due to radiation).
- So we want to **reconstruct the image from as few projections** as possible  
    

### 🧠 ==The Smart Trick:== ==Sparsity & Compressive Sensing==

- Many images (especially medical or scientific ones) have lots **of empty space or flat regions**.
- Only the **edge or boundaries of** objects have meaningful information
- So the image is **sparse** — meaning **most of the pixels are zero**, and only a few matter.

This lets us use **Compressive Sensing**:

> “If the image is mostly **empty (sparse)**, we can rebuild it using less data by filling in the blanks **smartly**”

### 🧮 ==How Do We Fill in the Blanks?==

**We solve an equation** that tries to:

- Match the **data** (the shadow we **captured**)
- and also **keep the image sparse** (only a few bright pixels)

To do that, we use a method called the **Lasso**, which:

- **Keeps things accurate** to the projection data,
- And penalizes **non-zero to force sparsity** using something called **L1 regularization**

This is like saying:

> “Only add brightness where it’s **really needed** — otherwise keep it black”

### 🥇 ==Why L1 (Lasso) is Better Than L2 (Ridge) Here:==

- **L1(Lasso)** encourages **exact zeros**, so it gives us **sharp edges and clean results**.
- **L2** (**ridge**) spreads the brightness too much, **causing blurry and incorrect pixels** (**called artifacts**)

Think of L1 as a strict editor who deletes all unnecessary stuff.

L2 is more like **someone who smudges everything** to make it look smoother — but that **hurts clarity**.

### 🔍 ==Final Result==

- Using L1 (Lasso): **We rebuild the image almost perfectly**, even **with noise**.
- Using L2 (Ridge): The results is **blurry**, **with errors**, **especially in the corners** (**which were seen in fewer projections**)