Nearest Centroid Classifier — Matrix Explanation

(What we do & Why we do it)

1. Representing the Data (What & Why)
What we do

We represent the dataset as a matrix:

𝑋
=
[
𝑥
11
	
𝑥
12
	
⋯
	
𝑥
1
𝑑


𝑥
21
	
𝑥
22
	
⋯
	
𝑥
2
𝑑


⋮
	
⋮
	
⋱
	
⋮


𝑥
𝑛
1
	
𝑥
𝑛
2
	
⋯
	
𝑥
𝑛
𝑑
]
X=
	​

x
11
	​

x
21
	​

⋮
x
n1
	​

	​

x
12
	​

x
22
	​

⋮
x
n2
	​

	​

⋯
⋯
⋱
⋯
	​

x
1d
	​

x
2d
	​

⋮
x
nd
	​

	​

	​


and the labels as:

𝑦
=
[
𝑦
1


𝑦
2


⋮


𝑦
𝑛
]
y=
	​

y
1
	​

y
2
	​

⋮
y
n
	​

	​

	​

Why we do this

Machine learning algorithms work on vectors and matrices

Each row = one data sample

Each column = one feature

This allows us to apply mathematical operations (mean, distance, etc.)

2. Identifying the Classes (What & Why)
What we do

We extract all unique class labels:

𝐶
=
{
𝑐
1
,
𝑐
2
,
…
,
𝑐
𝐾
}
C={c
1
	​

,c
2
	​

,…,c
K
	​

}
Why we do this

We need to know how many classes exist

The algorithm will compute one representative vector per class

Each class will later compete to “own” a new data point

3. Computing Class Centroids (Training Phase)
What we do

For each class 
𝑐
𝑘
c
k
	​

, we collect all samples that belong to it:

𝑋
𝑐
𝑘
=
{
𝑥
𝑖
∣
𝑦
𝑖
=
𝑐
𝑘
}
X
c
k
	​

	​

={x
i
	​

∣y
i
	​

=c
k
	​

}

Then we compute the mean (average) vector:

𝜇
𝑘
=
1
∣
𝑋
𝑐
𝑘
∣
∑
𝑥
𝑖
∈
𝑋
𝑐
𝑘
𝑥
𝑖
μ
k
	​

=
∣X
c
k
	​

	​

∣
1
	​

x
i
	​

∈X
c
k
	​

	​

∑
	​

x
i
	​

Why we do this

The centroid represents the center of the class in feature space

It summarizes all samples of a class using one vector

This reduces the entire class to a single prototype

Training becomes very fast (no optimization, no gradients)

👉 This is exactly what fit() does in your code.

4. Storing All Centroids in a Matrix
What we do

We stack all centroids into one matrix:

𝑀
=
[
𝜇
1
⊤


𝜇
2
⊤


⋮


𝜇
𝐾
⊤
]
M=
	​

μ
1
⊤
	​

μ
2
⊤
	​

⋮
μ
K
⊤
	​

	​

	​

Why we do this

Makes distance computation systematic and efficient

Each row corresponds to one class

This matrix is the entire trained model

👉 No weights, no parameters—just centroids.

5. Representing Test Data
What we do

New (unseen) samples are also written as a matrix:

𝑋
𝑡
𝑒
𝑠
𝑡
=
[
𝑧
1
⊤


𝑧
2
⊤


⋮


𝑧
𝑡
⊤
]
X
test
	​

=
	​

z
1
⊤
	​

z
2
⊤
	​

⋮
z
t
⊤
	​

	​

	​

Why we do this

Predictions must use the same feature space

Allows us to compare test points with centroids mathematically

6. Computing Distances (Prediction Phase)
What we do

For each test sample 
𝑧
𝑖
z
i
	​

, compute distance to each centroid:

𝐷
𝑖
𝑘
=
∥
𝑧
𝑖
−
𝜇
𝑘
∥
2
D
ik
	​

=∥z
i
	​

−μ
k
	​

∥
2
	​


This forms a distance matrix:

𝐷
=
[
𝐷
11
	
𝐷
12
	
⋯
	
𝐷
1
𝐾


𝐷
21
	
𝐷
22
	
⋯
	
𝐷
2
𝐾


⋮
	
⋮
	
⋱
	
⋮


𝐷
𝑡
1
	
𝐷
𝑡
2
	
⋯
	
𝐷
𝑡
𝐾
]
D=
	​

D
11
	​

D
21
	​

⋮
D
t1
	​

	​

D
12
	​

D
22
	​

⋮
D
t2
	​

	​

⋯
⋯
⋱
⋯
	​

D
1K
	​

D
2K
	​

⋮
D
tK
	​

	​

	​

Why we do this

Distance measures similarity

Smaller distance → more similar

We assume a point belongs to the class whose center is closest

Euclidean distance is simple and intuitive geometrically

👉 This corresponds to np.linalg.norm(X - centroid).

7. Making the Final Decision
What we do

For each test sample, choose the class with minimum distance:

𝑦
^
𝑖
=
arg
⁡
min
⁡
𝑘
𝐷
𝑖
𝑘
y
^
	​

i
	​

=arg
k
min
	​

D
ik
	​

Why we do this

This implements the nearest-prototype rule

The centroid acts as the class representative

The closest centroid “wins”

8. Final Output
What we do

We return the predicted label vector:

𝑦
^
=
[
𝑦
^
1


𝑦
^
2


⋮


𝑦
^
𝑡
]
y
^
	​

=
	​

y
^
	​

1
	​

y
^
	​

2
	​

⋮
y
^
	​

t
	​

	​

	​

Why this model is useful (Student takeaway)

✔ Extremely simple

✔ Very fast to train

✔ Easy to visualize

✘ Assumes classes are well-separated

✘ Cannot model complex boundaries