# ProstateCare+

## Introduction
ProstateCare+ is a tool that uses machine learning model to help you assess your risk for prostate cancer with ease and accuracy. The model is trained on a clinical dataset. 

## About Dataset
The data used in this example is sourced from a study conducted by Stamey et al. (1989). The
study aimed to investigate the relationship between the level of prostate-specific antigen (PSA)
and various clinical measures in a group of 97 men who were scheduled to undergo a radical
prostatectomy. PSA is a protein that is produced by the prostate gland, and higher levels of PSA
are often associated with a higher likelihood of having prostate cancer. The dataset provides
valuable information for examining the correlation between PSA levels and other clinical factors
in the context of prostate cancer

## Properties
There are eight (8) features in this dataset, particularly in regression models predicting prostatespecific antigen (PSA) levels
1. lcavol (log cancer volume): This represents the logarithm of the cancer volume. Cancer volume is a measurement of the size of the tumor in the prostate.
2. lweight (log prostate weight): This is the logarithm of the weight of the prostate.
3. Age: The age of the patient.
4. lbph (log benign prostatic hyperplasia): This represents the logarithm of the volume of benign prostatic hyperplasia (BPH). BPH is a noncancerous enlargement of the prostate that can affect urinary function and PSA levels.
5. lcp (log capsular penetration): This denotes the logarithm of the extent to which the cancer has penetrated the capsule of the prostate. Capsular penetration indicates a more advanced stage of cancer.
6. Gleason: The Gleason score is a grading system used to evaluate the prognosis of prostate cancer based on microscopic appearance. Higher Gleason scores indicate more aggressive cancer.
7. pgg45 (percentage of Gleason scores 4 or 5): This is the percentage of tissue samples that have a Gleason grade of 4 or 5. Higher percentages indicate a more aggressive and advanced cancer.
8. lpsa (log PSA level): The logarithm of the prostatespecific antigen (PSA) level in the blood. PSA is a protein produced by both normal and malignant cells of the prostate gland, and its levels are often elevated in men with prostate cancer.
9. Target: The level of PSA in the blood.