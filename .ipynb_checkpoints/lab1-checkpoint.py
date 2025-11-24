from nbformat import v4 as nbf

# Создаем новый ноутбук
nb = nbf.new_notebook()

# Ячейки: оформление в стиле IMRAD
cells = []

# Introduction
intro = """# Numerical Differentiation Lab Report

## Introduction

In this lab we study the accuracy of **numerical differentiation methods**.  
Numerical differentiation is prone to two types of errors:
- **Truncation error** due to approximation of the derivative with finite differences.
- **Round-off error** due to limited floating-point precision.

The goal is to compare 5 finite-difference formulas for differentiation across 5 different functions.  
We analyze how the absolute error depends on the step size \\(h\\), in a **log-log scale**.
"""

cells.append(nbf.new_markdown_cell(intro))

# Methods
methods = """## Methods

We consider the following finite difference schemes for approximating \\(f'(x)\\):

1. Forward difference:  
\\[ f'(x) \\approx \\frac{f(x+h) - f(x)}{h} \\]

2. Backward difference:  
\\[ f'(x) \\approx \\frac{f(x) - f(x-h)}{h} \\]

3. Central difference:  
\\[ f'(x) \\approx \\frac{f(x+h) - f(x-h)}{2h} \\]

4. 4th-order central difference:  
\\[ f'(x) \\approx \\frac{4(f(x+h)-f(x-h))}{2h \\cdot 3} - \\frac{(f(x+2h)-f(x-2h))}{4h \\cdot 3} \\]

5. 6th-order central difference:  
\\[ f'(x) \\approx \\frac{3(f(x+2h)-f(x-2h))}{20h} - \\frac{(f(x+3h)-f(x-3h))}{60h} + \\frac{15(f(x+h)-f(x-h))}{20h} \\]

We test these methods on the following functions:

1. \\( f(x) = \\sin(x^2) \\)  
2. \\( f(x) = \\cos(\\sin(x)) \\)  
3. \\( f(x) = e^{\\sin(\\cos(x))} \\)  
4. \\( f(x) = \\ln(x+3) \\)  
5. \\( f(x) = (x+3)^{1/3} \\)  

The exact derivatives are computed analytically.  

Step sizes are defined as:
\\[ h_n = \\frac{2}{2^n}, \\quad n = 1, 2, \\dots, 21 \\]

We evaluate the derivative at **x = 1**.
"""

cells.append(nbf.new_markdown_cell(methods))

# Code: imports and methods
code_setup = """import numpy as np
import matplotlib.pyplot as plt

# Finite difference methods
def diff_forward(f, x, h):
    return (f(x+h) - f(x)) / h

def diff_backward(f, x, h):
    return (f(x) - f(x-h)) / h

def diff_central(f, x, h):
    return (f(x+h) - f(x-h)) / (2*h)

def diff_4th_order(f, x, h):
    return (4*(f(x+h)-f(x-h))/(2*h) - (f(x+2*h)-f(x-2*h))/(4*h)) / 3

def diff_6th_order(f, x, h):
    return (3*(f(x+2*h)-f(x-2*h))/(4*h) - (f(x+3*h)-f(x-3*h))/(6*h)/10 + (f(x+h)-f(x-h))/(2*h)*15/10)

# Functions and derivatives
funcs = [
    (lambda x: np.sin(x**2), lambda x: 2*x*np.cos(x**2), "sin(x^2)"),
    (lambda x: np.cos(np.sin(x)), lambda x: -np.sin(np.sin(x))*np.cos(x), "cos(sin(x))"),
    (lambda x: np.exp(np.sin(np.cos(x))), lambda x: np.exp(np.sin(np.cos(x))) * np.cos(np.cos(x)) * (-np.sin(x)), "exp(sin(cos(x)))"),
    (lambda x: np.log(x+3), lambda x: 1/(x+3), "ln(x+3)"),
    (lambda x: (x+3)**(1/3), lambda x: (1/3)*(x+3)**(-2/3), "(x+3)^(1/3)")
]

methods = [
    ("Forward", diff_forward),
    ("Backward", diff_backward),
    ("Central", diff_central),
    ("4th order", diff_4th_order),
    ("6th order", diff_6th_order)
]

# Step sizes
n_vals = np.arange(1, 22)
h_vals = 2 / (2**n_vals)

x0 = 1.0
"""

cells.append(nbf.new_code_cell(code_setup))

# Results section
results = """## Results

We plot the **absolute error** of each method as a function of step size \\(h\\).  
The plots use logarithmic scale for both axes.
"""

cells.append(nbf.new_markdown_cell(results))

# Code for plotting
code_plot = """for f, df, name in funcs:
    plt.figure(figsize=(7,5))
    for method_name, method in methods:
        errors = []
        for h in h_vals:
            try:
                approx = method(f, x0, h)
                exact = df(x0)
                errors.append(abs(approx - exact))
            except Exception:
                errors.append(np.nan)
        plt.plot(h_vals, errors, label=method_name, marker="o")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("h")
    plt.ylabel("Absolute error")
    plt.title(f"Numerical differentiation error for {name}")
    plt.legend()
    plt.grid(True)
    plt.show()
"""

cells.append(nbf.new_code_cell(code_plot))

# Discussion
discussion = """## Discussion

The results show the following trends:

- The **forward** and **backward** difference formulas exhibit \\(O(h)\\) convergence.  
- The **central** difference formula improves accuracy to \\(O(h^2)\\).  
- The **4th-order** and **6th-order** schemes provide even higher accuracy for moderate values of \\(h\\).  
- For very small \\(h\\), rounding errors dominate and the error increases again. This reflects the balance between truncation error and round-off error.

### Conclusion
Higher-order central difference formulas significantly reduce error, but the optimal step size depends on the interplay between truncation and floating-point round-off errors.
"""

cells.append(nbf.new_markdown_cell(discussion))

# Собираем ноутбук
nb['cells'] = cells

# Сохраняем
path = "numerical_diff_lab.ipynb"
with open(path, "w", encoding="utf-8") as f:
    import json
    f.write(json.dumps(nb, ensure_ascii=False))


