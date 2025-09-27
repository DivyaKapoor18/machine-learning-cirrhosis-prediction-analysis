# Dataset Information

## Cirrhosis Patient Survival Prediction Dataset

### Source
- **Repository**: UCI Machine Learning Repository
- **Dataset**: Cirrhosis Patient Survival Prediction
- **URL**: https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1

### Description
This dataset contains medical records of 418 patients with liver cirrhosis, tracking their survival and various clinical parameters over time.

### Variables

#### Demographic Features
- `ID`: Patient identifier
- `Age`: Patient age in days
- `Sex`: Patient gender (M/F)

#### Clinical Features
- `N_Days`: Number of days of follow-up
- `Status`: Patient status (C: censored, CL: censored due to liver transplant, D: death)
- `Drug`: Treatment type (D-penicillamine vs Placebo)

#### Physical Examination
- `Ascites`: Presence of ascites (Y/N)
- `Hepatomegaly`: Enlarged liver (Y/N)
- `Spiders`: Spider angiomas (Y/N)
- `Edema`: Edema status (N, S, Y)

#### Laboratory Values
- `Bilirubin`: Serum bilirubin (mg/dl)
- `Cholesterol`: Serum cholesterol (mg/dl)
- `Albumin`: Serum albumin (mg/dl)
- `Copper`: Urine copper (μg/day)
- `Alk_Phos`: Alkaline phosphatase (U/liter)
- `SGOT`: Serum glutamic oxaloacetic transaminase (U/ml)
- `Tryglicerides`: Triglycerides (mg/dl)
- `Platelets`: Platelets per cubic ml/1000
- `Prothrombin`: Prothrombin time (seconds)

#### Disease Stage
- `Stage`: Histologic stage of disease (1-4)

### Data Quality
- Contains some missing values (handled in analysis)
- Mixed data types (numerical and categorical)
- Real-world clinical data with expected variability