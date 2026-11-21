# NHANES 2017–2018 subset

`nhanes_2017_2018.csv` is a subset of the U.S. National Health and Nutrition
Examination Survey (NHANES), 2017–2018 cycle, published by the CDC's National
Center for Health Statistics. NHANES data are in the public domain.

Adults aged 20+ only. Columns are merged from these public files
(https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/):

- DEMO_J — sex (RIAGENDR), age (RIDAGEYR), race/ethnicity (RIDRETH3)
- GHB_J  — HbA1c % (LBXGH)
- GLU_J  — fasting plasma glucose mg/dl (LBXGLU) and fasting-subsample weight (WTSAF2YR)
- INS_J  — fasting insulin µU/ml (LBXIN)
- BMX_J  — BMI (BMXBMI), waist cm (BMXWAIST)
- DIQ_J  — self-reported diabetes diagnosis (DIQ010)

Prevalence uses the fasting-subsample survey weight (fast_wt) as NCHS recommends;
other statistics are unweighted and labelled as such. This is research and
education software, not a clinical or official statistic.
