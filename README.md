# AI Curve Fitter

A web-based symbolic regression and curve-fitting application that automatically discovers mathematical equations from data.

The system analyzes numerical datasets, generates a large library of mathematical transformations, evaluates candidate models, and returns the best-fitting analytical equation along with performance metrics and interactive visualizations.

---

## Features

### Automatic Equation Discovery

Instead of manually choosing a model, the application automatically searches through:

* Linear relationships
* Polynomial functions
* Trigonometric functions
* Exponential functions
* Logarithmic functions
* Hyperbolic functions
* Rational expressions
* Composite mathematical transformations
* Sparse symbolic regression models

The goal is to identify a compact mathematical expression that best describes the data.

---

## Advanced Function Library

The engine generates hundreds of candidate features including:

### Polynomial Features

```text
x
x²
x³
x⁴
x⁵
x⁶
(x+1)²
(x−1)²
```

### Trigonometric Functions

```text
sin(x)
cos(x)
tan(x)
sec(x)
csc(x)
cot(x)

sin(2x)
sin(3x)
sin(5x)

cos(2x)
cos(3x)
cos(5x)
```

### Inverse Trigonometric Functions

```text
asin(x)
acos(x)
atan(x)
```

### Hyperbolic Functions

```text
sinh(x)
cosh(x)
tanh(x)

asinh(x)
acosh(x)
atanh(x)
```

### Exponential Functions

```text
exp(x)
exp(-x²)

2^x
3^x
5^x
10^x

exp(sin(x))
exp(cos(x))
```

### Logarithmic Functions

```text
log(1+|x|)
log2(1+|x|)
log10(1+|x|)
```

### Special Functions

```text
erf(x)
erfc(x)

gamma(x)
gammaln(x)

Bessel J0
Bessel J1
Bessel Y0
Bessel Y1
```

### Signal Processing Functions

```text
sinc(x)
gaussian(x)
sigmoid(x)
```

### Composite Features

Examples:

```text
sin(x²)

x·sin(x²)

x²·cos(x)

sin(exp(x))

cos(log(1+|x|))
```

The system can also create interaction terms between generated features.

---

## Exact Pattern Detection

Before performing symbolic regression, the application attempts to detect exact relationships.

### Linear Detector

Recognizes:

```math
y = mx + b
```

### Polynomial Detector

Automatically identifies exact polynomial relationships up to configurable degree limits.

Example:

```math
y = 3x^2 + 2x + 1
```

### Sinusoidal Detector

Attempts to identify:

```math
y = A \sin(Bx + C) + D
```

with automatic parameter estimation.

---

## Mathematical Expression Parser

Users may enter mathematical expressions directly in the input data.

Examples:

```text
sin(pi/4)

ln(2)

sqrt(5)

e^2

cos(pi)
```

The frontend evaluates these expressions automatically before fitting.

---

## Interactive Visualization

### 2D Curve Fitting

Displays:

* Original data points
* Predicted curve
* Interactive zooming
* Interactive panning

Built using Plotly.

### 3D Surface Fitting

For multi-variable datasets:

```text
x1, x2 → y
```

the application generates:

* 3D scatter plots
* Surface plots
* Interactive rotation and exploration

---

## Model Performance Metrics

For every fitted model the system reports:

### Training R²

Measures fit quality on training data.

### Validation R²

Measures generalization performance on unseen data.

Example:

```text
Training R²: 0.9987
Validation R²: 0.9979
```

---

## Export Features

### Copy Equation

Copies the discovered symbolic equation to the clipboard.

### Python Export

Exports the discovered model as a Python function.

Example:

```python
def model(x):
    return 3*x + 2
```

---

## Technology Stack

### Backend

* Flask
* NumPy
* SciPy
* SymPy
* Scikit-Learn

### Frontend

* HTML5
* JavaScript
* Plotly.js

---

## Project Structure

```text
curve_fitter/
│
├── app.py
├── templates/
│   └── index.html
├── requirements.txt
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/sanjay-181006/curve_fitter.git
cd curve_fitter
```

### Create Virtual Environment

```bash
python -m venv venv
```

#### macOS / Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install flask numpy scipy sympy scikit-learn
```

---

## Running the Application

Start the Flask server:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

in your browser.

---

## Example Dataset

Input:

```text
1,3
2,5
3,7
4,9
5,11
```

Discovered equation:

```text
y = 2x + 1
```

---

## Multi-Variable Example

Input:

```text
1,2,5
2,3,8
3,4,11
4,5,14
```

Where:

```text
x1,x2,y
```

The system automatically searches for a surface model.

---

## Applications

* Scientific data analysis
* Experimental modeling
* Engineering curve fitting
* Symbolic regression research
* Mathematical discovery
* Rapid prototyping of predictive equations
* Educational demonstrations of regression techniques

---

## Future Improvements

* Genetic Programming based symbolic regression
* Neural symbolic regression
* CSV drag-and-drop support
* Equation complexity controls
* LaTeX equation export
* Confidence intervals
* Residual analysis dashboard
* Model comparison dashboard

---

## Author

**Sanjay Selvakumar**

Machine Learning • Artificial Intelligence • Symbolic Regression • Scientific Computing • Mathematical Modeling

LinkedIn: https://linkedin.com/in/sanjay-selvakumar-55129b293
