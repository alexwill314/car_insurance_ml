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

The project uses:
- Policy-level data with exposure, claim counts and risk features
- Claim-level severity data with individual claim amounts

## 4. Methodology

- Exposure-adjusted exploratory data analysis
- Claim frequency modeling using Poisson regression and Gradient Boosting
- Claim severity modeling using Gamma/log-transformed regression
- Pure premium estimation
- Calibration analysis by risk decile
- Segment-level error analysis
- Interactive Plotly dashboard

## 5. Key Findings

To be added after the analysis.

## 6. Technologies

Python, pandas, scikit-learn, Plotly, statsmodels, Jupyter