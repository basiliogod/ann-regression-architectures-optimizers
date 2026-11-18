# Neural Network Architectures & Optimization Algorithms in Regression Tasks

**Authors:** Guilherme Basílio (2100123) & Carlos Santos (2128623) <br>
**Institution:** University of Madeira <br>
**Course:** Artificial Neural Networks (2026/2027) <br>
**Date:** November 2026

---

An end-to-end comparative machine learning framework implemented **from scratch in MATLAB / GNU Octave** (without external machine learning toolboxes) to investigate the interplay between feedforward neural network topology (width vs. depth) and first- and second-order optimization algorithms across two distinct regression paradigms.

The pipeline spans exploratory data analysis, domain-specific feature engineering, strict anti-leakage partitioning, analytical backpropagation verified via centered finite differences, validation-driven hyperparameter grid search, and **540 paired experimental training runs** ($2\text{ datasets} \times 3\text{ architectures} \times 3\text{ optimizers} \times 30\text{ seeds}$).

## Project Scope & Experimental Framework

### 1. Problem Formulation & Datasets
To evaluate model capacity and optimizer dynamics under contrasting signal-to-noise regimes, two benchmark regression tasks were studied:
- **Concrete Compressive Strength (Static Tabular Regression):** Predicting the compressive strength ($\text{MPa}$) of high-performance concrete from $8$ physical/chemical mixture components and curing age ($N = 1\,030$ observations). Governed by deterministic, strongly non-linear hydration laws.
- **Daily Minimum Temperatures in Melbourne (Time-Series Forecasting):** One-step-ahead forecasting of daily minimum temperatures $T(t+1)$ ($^{\circ}\text{C}$) over a $10$-year period ($1981\text{--}1990$, $N = 3\,650$ daily records). Characterized by a smooth annual seasonal cycle superimposed on high-frequency stochastic weather noise.

### 2. Preprocessing, Feature Engineering & Anti-Leakage Partitioning
- **Concrete Engineering ($d = 10$):** Derived the physical **Water/Cement Ratio** ($\text{Water}/\text{Cement}$, grounded in Abrams' law) and applied a logarithmic transformation **$\ln(\text{Age})$** to linearize rapid early-stage curing gains.
- **Melbourne Engineering ($d = 9$):** Sanitized anomalous `?` formatting artifacts, constructed an autoregressive sliding window of the $p = 7$ most recent temperatures $[T(t), \dots, T(t-6)]$ justified by the Autocorrelation Function (ACF), and embedded continuous cyclical calendar features $\sin(2\pi \cdot \text{doy}_t / 365)$ and $\cos(2\pi \cdot \text{doy}_t / 365)$.
- **Data Partitioning ($70\% / 15\% / 15\%$):**
  - *Concrete:* Fixed random split ($721$ train / $155$ validation / $154$ test), accompanied by a structural mixture analysis revealing that $68.8\%$ of test formulations share ingredients with training samples at different curing ages (*group leakage* discussion).
  - *Melbourne:* Strictly chronological contiguous block split ($2\,550$ train / $546$ validation / $547$ test) to preserve causality.
- **Z-Score Standardization:** Computed mean and standard deviation exclusively on the training subset to prevent data leakage, inverting predictions back to original physical units ($\text{MPa}$ and $^{\circ}\text{C}$) for final test evaluation.

### 3. Neural Network Topologies & Gradient Verification
All models implement fully connected feedforward networks with $\tanh$ hidden activations, a linear output neuron, Mean Squared Error (MSE) loss, and uniform **Glorot/Xavier** weight initialization:
- **Architecture A (*Reference*):** $1$ hidden layer with $10$ neurons ($121$ params on Concrete; $111$ on Melbourne).
- **Architecture B (*Wide*):** $1$ hidden layer with $100$ neurons ($1\,201$ params on Concrete; $1\,101$ on Melbourne).
- **Architecture C (*Deep*):** $10$ hidden layers with $10$ neurons each ($1\,111$ params on Concrete; $1\,101$ on Melbourne).
- **Analytical Gradient Verification:** Validated exact backpropagation gradients against centered finite differences ($\epsilon = 10^{-6}$), achieving maximum relative errors of $1.083 \times 10^{-8}$ (Arq. A), $1.389 \times 10^{-7}$ (Arq. B), and $3.437 \times 10^{-7}$ (Arq. C).

### 4. Training Algorithms & Paired Experimental Protocol
- **Optimizers Implemented from Scratch:**
  1. **Mini-batch Gradient Descent ($B = 32$):** Reshuffled every epoch with tuned learning rate $\eta$.
  2. **BFGS (Quasi-Newton):** Full-batch inverse Hessian approximation $\mathbf{H}_k$ with Strong Wolfe line search ($c_1 = 10^{-4}$, $c_2 = 0.9$).
  3. **Adam ($B = 32$):** Adaptive first and second moment estimates with bias correction ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$) and tuned $\eta$.
- **Hyperparameter Tuning:** Tuned $\eta$ across a $6$-point logarithmic grid $[10^{-4}, 10^{-1}]$ ($5$ repetitions $\times 60$ epochs) using validation MSE median.
- **Paired Final Evaluation:** Executed $30$ paired runs (seeds $1$ to $30$, identical initial weights $\boldsymbol{\theta}_0^{(k)}$ across optimizers) under equal computational budgets measured in *epoch-equivalents* ($250$ for Concrete; $200$ for Melbourne), selecting the weights achieving the lowest validation loss.

---

## Key Experimental Results (Test Set)

### Concrete Compressive Strength ($\text{MPa}$) — Baseline OLS RMSE: `6.417 MPa` ($R^2 = 0.8418$)

| Architecture | Optimizer | Test RMSE (Mean $\pm$ Std) | Median RMSE | Best / Worst | Mean $R^2$ | Avg. Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A (Reference)** | Mini-batch GD | $5.206 \pm 0.278$ | $5.206$ | $4.515 / 5.723$ | $0.8956$ | $2.66\text{ s}$ |
| **A (Reference)** | BFGS | $5.008 \pm 0.264$ | $4.956$ | $4.583 / 5.616$ | $0.9034$ | $0.77\text{ s}$ |
| **A (Reference)** | Adam | $5.161 \pm 0.273$ | $5.173$ | $4.658 / 5.787$ | $0.8974$ | $2.88\text{ s}$ |
| **B (Wide)** | Mini-batch GD | $4.631 \pm 0.191$ | $4.631$ | $4.297 / 5.332$ | $0.9175$ | $10.83\text{ s}$ |
| **B (Wide)** | **BFGS** | **$4.389 \pm 0.150$** | **$4.385$** | **$4.127 / 4.732$** | **$0.9259$** | $29.90\text{ s}$ |
| **B (Wide)** | **Adam** | **$4.440 \pm 0.210$** | **$4.396$** | **$4.012 / 4.958$** | **$0.9241$** | **$11.26\text{ s}$** |
| **C (Deep)** | Mini-batch GD | $4.855 \pm 0.278$ | $4.788$ | $4.423 / 5.525$ | $0.9091$ | $18.52\text{ s}$ |
| **C (Deep)** | BFGS | $4.675 \pm 0.374$ | $4.680$ | $3.908 / 5.426$ | $0.9155$ | $20.98\text{ s}$ |
| **C (Deep)** | Adam | $4.602 \pm 0.386$ | $4.756$ | $3.928 / 5.259$ | $0.9181$ | $20.09\text{ s}$ |

### Melbourne Min Temperatures ($^{\circ}\text{C}$) — Baselines: Persistence `2.574 °C` ($R^2 = 0.5640$) | AR(7) `2.262 °C` ($R^2 = 0.6634$)

| Architecture | Optimizer | Test RMSE (Mean $\pm$ Std) | Median RMSE | Best / Worst | Mean $R^2$ | Avg. Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A (Reference)** | Mini-batch GD | $2.193 \pm 0.021$ | $2.192$ | $2.152 / 2.250$ | $0.6834$ | $21.29\text{ s}$ |
| **A (Reference)** | BFGS | $2.184 \pm 0.017$ | $2.183$ | $2.146 / 2.223$ | $0.6863$ | $2.21\text{ s}$ |
| **A (Reference)** | **Adam** | **$2.175 \pm 0.017$** | **$2.177$** | **$2.145 / 2.214$** | **$0.6887$** | **$23.06\text{ s}$** |
| **B (Wide)** | Mini-batch GD | $2.182 \pm 0.010$ | $2.181$ | $2.165 / 2.206$ | $0.6866$ | $24.52\text{ s}$ |
| **B (Wide)** | BFGS | $2.181 \pm 0.009$ | $2.180$ | $2.164 / 2.203$ | $0.6870$ | $17.56\text{ s}$ |
| **B (Wide)** | Adam | $2.181 \pm 0.013$ | $2.177$ | $2.164 / 2.219$ | $0.6870$ | $25.62\text{ s}$ |
| **C (Deep)** | Mini-batch GD | $2.207 \pm 0.022$ | $2.202$ | $2.166 / 2.251$ | $0.6794$ | $41.94\text{ s}$ |
| **C (Deep)** | BFGS | $2.191 \pm 0.026$ | $2.187$ | $2.155 / 2.266$ | $0.6841$ | $15.59\text{ s}$ |
| **C (Deep)** | Adam | $2.200 \pm 0.017$ | $2.202$ | $2.158 / 2.232$ | $0.6814$ | $42.39\text{ s}$ |

---

## Repository Structure

```text
ann-regression-architectures-optimizers/
│
├── data/
│   ├── Concrete_Data.csv                          # UCI Concrete Compressive Strength dataset converted to csv (N=1030)
│   ├── Concrete_Data.xls                          # Original UCI Concrete Compressive Strength dataset (N=1030)
│   └── daily-minimum-temperatures-in-me.csv       # Melbourne daily minimum temperatures (N=3650)
│
├── src/
│   ├── ann_lib.m                                  # Self-contained library (EDA, FE, MLP, backprop, GD, BFGS, Adam plots)
│   └── main_assignment2.m                         # Master script orchestrating Steps 1 through 8
│
├── results/
│   ├── terminal_logs.txt                          # Copy of all terminal output
│   ├── tuning_results.mat                         # Validation grid search curves and optimal learning rates
│   ├── final_results_concrete.mat                 # 30-run paired metrics and histories for Concrete
│   ├── final_results_melbourne.mat                # 30-run paired metrics and histories for Melbourne
│   └── final_results_all.mat                      # Consolidated experimental archive
│
├── outputs/
│   ├── eda_concrete_histograms.png                # Concrete variable distributions
│   ├── eda_concrete_corr.png                      # Concrete Pearson correlation matrix
│   ├── eda_concrete_scatter.png                   # Concrete feature vs. target scatter plots
│   ├── eda_melbourne_series.png                   # Melbourne time series, distribution & seasonal profile
│   ├── eda_melbourne_acf_scatter.png              # Melbourne ACF, lag-1 scatter & daily jumps
│   ├── tuning_concrete.png                        # Learning rate validation curves (Concrete)
│   ├── tuning_melbourne.png                       # Learning rate validation curves (Melbourne)
│   ├── progress_ep_concrete_archA.png             # Training/validation median & IQR curves (Concrete Arq. A)
│   ├── progress_ep_concrete_archB.png             # Training/validation median & IQR curves (Concrete Arq. B)
│   ├── progress_ep_concrete_archC.png             # Training/validation median & IQR curves (Concrete Arq. C)
│   ├── progress_ep_melbourne_archA.png            # Training/validation median & IQR curves (Melbourne Arq. A)
│   ├── progress_ep_melbourne_archB.png            # Training/validation median & IQR curves (Melbourne Arq. B)
│   ├── progress_ep_melbourne_archC.png            # Training/validation median & IQR curves (Melbourne Arq. C)
│   ├── progress_time_concrete_archA.png           # Convergence vs. wall-clock time (Concrete Arq. A)
│   ├── progress_time_concrete_archB.png           # Convergence vs. wall-clock time (Concrete Arq. B)
│   ├── progress_time_concrete_archC.png           # Convergence vs. wall-clock time (Concrete Arq. C)
│   ├── progress_time_melbourne_archA.png          # Convergence vs. wall-clock time (Melbourne Arq. A)
│   ├── progress_time_melbourne_archB.png          # Convergence vs. wall-clock time (Melbourne Arq. B)
│   ├── progress_time_melbourne_archC.png          # Convergence vs. wall-clock time (Melbourne Arq. C)
│   ├── boxplot_rmse_concrete.png                  # Custom box plots of 30-run test RMSE (Concrete)
│   ├── boxplot_rmse_melbourne.png                 # Custom box plots of 30-run test RMSE (Melbourne)
│   ├── predictions_concrete.png                   # Predicted vs. measured strength & residuals
│   └── predictions_melbourne.png                  # Observed vs. forecast temperature, 90-day zoom & residuals
│
├── docs/
│   ├── Assignment_II_2026-27.pdf                  # Official course assignment specification (Regression Task)
│   ├── report.tex                                 # IEEE conference report in LATEX
│   ├── Report_RNA_Assignment2.pdf                 # Final 10-page IEEE conference report
│   └── Presentation_RNA_Assignment2.pdf           # 15-slide executive presentation
│
└── README.md                                      # Project documentation
```

---

## Execution Guide

### Prerequisites
- **GNU Octave** ($\ge 7.0$) or **MATLAB** ($\ge \text{R2020b}$). No external toolboxes are required.

### Running the Full Pipeline
1. Clone the repository and navigate to the `src/` directory:
   ```bash
   git clone https://github.com/basiliogod/ann-regression-architectures-optimizers.git
   cd ann-regression-architectures-optimizers/src
   ```
2. Run the master script in GNU Octave or MATLAB:
   ```bash
   octave --no-gui main_assignment2.m
   ```
   Or inside the interactive Octave/MATLAB console:
   ```matlab
   run('main_assignment2.m')
   ```
3. All statistical tables, gradient verification checks, and baseline comparisons will print directly to the console, while all `.mat` archives and high-resolution (`300 dpi`) figures will be exported automatically to `results/` and `outputs/`.