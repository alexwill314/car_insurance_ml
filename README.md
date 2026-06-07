# Motor Insurance Pricing with GLMs and Machine Learning

## 1. Project Overview

This project builds a motor insurance pricing workflow using the French Motor Third-Party Liability dataset. 
The goal is to estimate expected claim costs by separately modeling claim frequency and claim severity.

## 2. Business Context

In non-life insurance, the technical pure premium can be decomposed into:

Pure Premium = Expected Claim Frequency × Expected Claim Severity

The project compares traditional actuarial models with machine learning models and evaluates them not only by 
predictive performance, but also by calibration, interpretability and risk segmentation.

## 3. Data

The project uses two datasets comprised of:
- Policy-level data with exposure, claim counts and risk features of 677,991 motor third-part liability policies 
(observed mostly on one year)
- Claim-level severity data with individual claim amounts for 26,639 motor third-part liability policies

The datasets are fetched from openml.
For more info see https://github.com/dutangc/CASdatasets

## 4. Methodology

- Exposure-adjusted exploratory data analysis
- Claim frequency modeling using Poisson regression and Random Forest
- Claim severity modeling using Gamma/log-transformed regression
- Pure premium estimation
- Calibration analysis by risk decile

## 5. Key Findings

The Random Forest slightly outperforms the Poisson GLM baseline.

| Model | Mean Poisson Deviance (MPD) | Deviance Score (D²) |
| :--- | :---: |:-------------------:|
| Poisson GLM (Baseline) | 0.466361 |      0.039818       |
| Random Forest (ML) | 0.457052 |      0.058983       |

Both models show modest explanatory power, which is expected for claim frequency data because claims are sparse and highly random at the individual policy level.

The Poisson GLM provides an interpretable actuarial baseline and accounts for different observation periods using an exposure offset. The Random Forest achieves a slightly lower Poisson deviance and higher D², suggesting that it captures some non-linear effects or interactions.

Overall, the results show that machine learning can provide incremental improvement over a traditional GLM baseline, but the available features explain only a limited share of claim frequency variation.

## 6. Technologies

Python, pandas, scikit-learn, statsmodels, Jupyter