# Nearest Centroid Classifier — Matrix Explanation

This document explains the **Nearest Centroid Classifier** in a **matrix-based, student-friendly way**, including what each step does and why we do it. Perfect for learning, teaching, or implementing a simple custom classifier in Python.

---

## 1. Representing the Data (What & Why)

### What we do

We represent the dataset as a **matrix**:

```
X = [
 [x_11, x_12, ..., x_1d],
 [x_21, x_22, ..., x_2d],
 ...
 [x_n1, x_n2, ..., x_nd]
]
```

and the labels as:

```
y = [y_1, y_2, ..., y_n]^T
```

### Why we do this

* Machine learning algorithms work on **vectors and matrices**
* Each **row = one data sample**
* Each **column = one feature**
* Allows mathematical operations like mean and distance

---

## 2. Identifying the Classes (What & Why)

### What we do

We extract all unique class labels:

```
C = {c_1, c_2, ..., c_K}
```

### Why we do this

* Know how many classes exist
* Compute **one representative vector per class**
* Each class will later compete to “own” a new data point

---

## 3. Computing Class Centroids (Training Phase)

### What we do

For each class `c_k`, collect all samples belonging to it:

```
X_ck = {x_i | y_i = c_k}
```

Compute the **mean (average) vector**:

```
mu_k = (1 / |X_ck|) * sum(x_i in X_ck)
```

### Why we do this

* Centroid represents the **center of the class in feature space**
* Summarizes all samples of a class using **one vector**
* Training is very fast (no optimization required)

> This is what `fit()` does in code.

---

## 4. Storing All Centroids in a Matrix

### What we do

Stack all centroids into one matrix:

```
M = [
 mu_1^T,
 mu_2^T,
 ...,
 mu_K^T
]
```

### Why we do this

* Makes distance computation **systematic and efficient**
* Each row corresponds to **one class**
* This matrix is the **entire trained model**

> No weights, no parameters—just centroids.

---

## 5. Representing Test Data

### What we do

New (unseen) samples as a matrix:

```
X_test = [
 z_1^T,
 z_2^T,
 ...,
 z_t^T
]
```

### Why we do this

* Predictions use the **same feature space**
* Allows comparison between test points and centroids

---

## 6. Computing Distances (Prediction Phase)

### What we do

For each test sample `z_i`, compute distance to each centroid:

```
D_ik = || z_i - mu_k ||_2
```

Distance matrix:

```
D = [
 [D_11, D_12, ..., D_1K],
 [D_21, D_22, ..., D_2K],
 ...
 [D_t1, D_t2, ..., D_tK]
]
```

### Why we do this

* Distance measures **similarity**
* Smaller distance → more similar
* Point belongs to class whose **center is closest**

> This corresponds to `np.linalg.norm(X - centroid)` in Python.

---

## 7. Making the Final Decision

### What we do

For each test sample, choose the class with minimum distance:

```
y_hat_i = argmin_k D_ik
```

### Why we do this

* Implements the **nearest-prototype rule**
* Closest centroid “wins”

---

## 8. Final Output

### What we do

Return the predicted label vector:

```
y_hat = [y_hat_1, y_hat_2, ..., y_hat_t]^T
```

---

## Why this model is useful

* ✔ Extremely simple and intuitive
* ✔ Very fast to train (just averaging)
* ✔ Easy to visualize in 2D/3D feature space
* ✘ Assumes classes are well-separated
* ✘ Cannot model complex decision boundaries

### One-Line Intuition

> Assign a sample to the class whose mean feature vector is closest.
