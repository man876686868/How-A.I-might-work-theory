# Main Theory: How AI Might Work

## 1. Your Theory — Expanded and Made Explicit

### Core Claim
Model parameters (weights & biases) are not “memory slots” holding facts. They are coefficients of functions — linear maps, translations, and non-linear activations — that shape a high-dimensional transformation \( x \mapsto \hat{y} \). Intelligence is the emergent behavior of these composed transformations.

### Forward Pass as Function Application
A network computes:
\[
\hat{y} = \text{softmax}(W_2 \, f(W_1 x + b_1) + b_2)
\]

This is a composition of:
- A linear map,
- A nonlinearity \( f \) (which folds/warps geometry),
- Another linear map,
- And a final projection to probabilities.

### Parameters as Geometry
- \( W_1, b_1 \): Define a linear embedding + shift of input space into a hidden manifold. \( f \) then warps that manifold nonlinearly.
- \( W_2, b_2 \): Read out a linear functional on that warped manifold to produce logits.

Thus, parameters determine the shape, curvature, and relative positions of regions in representation space.

### Training
Backpropagation and Stochastic Gradient Descent (SGD) modify these parameters via gradients computed from the loss. Gradients are vectors in parameter space that reshape the linear maps, gradually changing the geometry to better separate/correlate inputs to desired outputs.

### Emergence
Repeated nonlinear composition across layers creates features and higher-order statistical detectors that were not explicitly encoded — i.e., emergent abilities.

### Tradeoffs / Dynamics
Capacity (parameters) vs regularization, flat vs sharp minima, memorization vs generalization — all are geometric properties of parameter space and the loss landscape.

---

## 2. Unpacking the Symbolic Forward Equation

We start with:
\[
\hat{y} = \text{softmax}(W_2 \, f(W_1 x + b_1) + b_2)
\]

### Definitions and Shapes (General):
- \( x \in \mathbb{R}^{n_{\text{in}}} \): Input vector.
- \( W_1 \in \mathbb{R}^{n_{\text{hidden}} \times n_{\text{in}}}, b_1 \in \mathbb{R}^{n_{\text{hidden}}} \): First layer weights and biases.
- \( f: \mathbb{R}^{n_{\text{hidden}}} \to \mathbb{R}^{n_{\text{hidden}}} \): Elementwise nonlinearity (e.g., tanh, ReLU).
- \( W_2 \in \mathbb{R}^{n_{\text{out}} \times n_{\text{hidden}}}, b_2 \in \mathbb{R}^{n_{\text{out}}} \): Second layer weights and biases.
- \( \text{softmax}: \mathbb{R}^{n_{\text{out}}} \to \Delta_{n_{\text{out}} - 1} \): Maps logits to probabilities.

---

## 3. Concrete Worked Example

### Dimensions
- Input dimension: \( n_{\text{in}} = 2 \).
- Hidden dimension: \( n_{\text{hidden}} = 3 \).
- Output dimension: \( n_{\text{out}} = 2 \) (binary classification).

### Chosen Numeric Parameters
\[
x = \begin{bmatrix} 0.5 \\ -1.0 \end{bmatrix}, \, 
W_1 = \begin{bmatrix} 0.2 & -0.4 \\ 0.7 & 0.1 \\ -0.5 & 0.9 \end{bmatrix}, \

b_1 = \begin{bmatrix} 0.1 \\ -0.2 \\ 0.05 \end{bmatrix}
\]
\[
W_2 = \begin{bmatrix} 0.5 & -1.0 & 0.3 \\ -0.2 & 0.4 & 0.6 \end{bmatrix}, \
b_2 = \begin{bmatrix} 0.0 \\ 0.1 \end{bmatrix}
\]
Target label (one-hot): \( y = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \).

Loss: Cross-entropy: \( L = -\log(\hat{y}_0) \).

### 3.1 Forward Pass
#### Step A: Linear Projection to Hidden Pre-Activations
\[
z_1 = W_1 x + b_1 = \begin{bmatrix} 0.6 \\ 0.05 \\ -1.10 \end{bmatrix}
\]

#### Step B: Activation
\[
h = \tanh(z_1) \approx \begin{bmatrix} 0.537 \\ 0.050 \\ -0.800 \end{bmatrix}
\]

#### Step C: Logits
\[
z_2 = W_2 h + b_2 \approx \begin{bmatrix} -0.022 \\ -0.468 \end{bmatrix}
\]

#### Step D: Softmax Probabilities
\[
\hat{y}_i = \frac{e^{z_{2,i}}}{\sum_j e^{z_{2,j}}}, \, \hat{y} \approx \begin{bmatrix} 0.610 \\ 0.390 \end{bmatrix}
\]

#### Step E: Loss
\[
L = -\log(\hat{y}_0) \approx 0.495
\]

---

## 4. Backpropagation
### Gradients for Top Linear Layer
\[
\frac{\partial L}{\partial W_2} = d_{z_2} \cdot h^\top, \, 
\frac{\partial L}{\partial b_2} = d_{z_2}
\]

Numerically:
\[
d_{W_2} \approx \begin{bmatrix} -0.210 & -0.019 & 0.312 \\ 0.210 & 0.019 & -0.312 \end{bmatrix}, \, 
d_{b_2} \approx \begin{bmatrix} -0.390 \\ 0.390 \end{bmatrix}
\]

---

## 5. Geometry and Emergence
- **Initial Mapping**: \( W_1 \) projects \( x \) into a hidden coordinate \( z_1 \). Nonlinearity \( \tanh \) warps space to \( h \). \( W_2 \) creates decision hyperplanes.
- **Error Signal**: Gradients reshape geometry via \( W_1, W_2 \) updates.
- **Emergence**: Composition of linear maps + nonlinear folds morph input space into separable structures.

---

## 6. Practical Recommendations
- Visualize decision boundaries in \( h \)-space.
- Run perturbation studies to measure sensitivity.
- Test activation saturation to observe gradient attenuation.

---

## 7. Appendix: Raw Numeric Table
| Variable     | Value                     |
|--------------|---------------------------|
| Input \( x \) | \([0.5, -1.0]
| \( z_1 \)     | \([0.6, 0.05, -1.10]
| \( h \)       | \([0.537, 0.050, -0.800]
| \( z_2 \)     | \([-0.022, -0.468]
| \( \hat{y} \) | \([0.610, 0.390]
| Loss \( L \)  | \( 0.495 \)
