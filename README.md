---
title: "Predicting readmission probability for diabetes inpatients"
author: "Cheshta Dhingra"
date: 'Due: April 7, 2017 at 11:59PM'
output:
  pdf_document:
    toc: no
    toc_depth: 2
  html_document:
    number_sections: yes
    self_contained: yes
    toc: no
header-includes:
- \usepackage{fancyhdr}
- \pagestyle{fancy}
- \fancyfoot[CO,CE]{}
- \fancyfoot[LE,RO]{\thepage}
subtitle: STAT 471/571/701, Fall 2017
graphics: yes
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(tidy=TRUE, fig.width=6,  fig.height=5, 
                      fig.align='left', dev = 'pdf')
```
[image_0]: ./images/a.png
[image_1]: ./images/b.png
[image_2]: ./images/c.png
[image_3]: ./images/d.png
[image_4]: ./images/e.png
[image_5]: ./images/f.png
[image_6]: ./images/g.png
[image_7]: ./images/h.png
[image_8]: ./images/i.png
[image_9]: ./images/j.png
[image_10]: ./images/k.png
[image_11]: ./images/l.png
[image_12]: ./images/m.png 


# I. Executive Summary 

## a. Background

Diabetes is a chronic medical condition affecting millions of Americans, but if managed well, with good diet, exercise and medication, patients can lead relatively normal lives. However, if improperly managed, diabetes can lead to patients being continuously admitted and readmitted to hospitals. Readmissions are especially serious - they represent a failure of the health system to provide adequate support to the patient and are extremely costly to the system. As a result, the Centers for Medicare and Medicaid Services announced in 2012 that they would no longer reimburse hospitals for services rendered if a patient was readmitted with complications within 30 days of discharge. 

Given these policy changes, being able to identify and predict those patients most at risk for costly readmissions has become a pressing priority for hospital administrators. In this project, we shall explore how to use the techniques we have learned in order to help better manage diabetes patients who have been admitted to a hospital. Our goal is to avoid patients being readmitted within 30 days of discharge, which reduces costs for the hospital and improves outcomes for patients. The goals of this analysis are:
1.  Identify the factors predicting whether or not the patient will be readmitted within 30 days. 
2.  Propose a classification rule to predict if a patient will be readmitted within 30 days. 

## b. Data  

The original data is from the Center for Clinical and Translational Research at Virginia Commonwealth University. It covers data on diabetes patients across 130 U.S. hospitals from 1999 to 2008. There are over 100,000 unique hospital admissions in this dataset, from ~70,000 unique patients. The data includes demographic elements, such as age, gender, and race, as well as clinical attributes such as tests conducted,emergency/inpatient visits, etc. All observations have five things in common:

1.	They are all hospital admissions
2.	Each patient had some form of diabetes
3.	The patient stayed for between 1 and 14 days.
4.	The patient had laboratory tests performed on him/her.
5.	The patient was given some form of medication during the visit. 

A more detailed summary of each variable can be found below in section IIA. 

## c. Methods

To developed my final model I created models using two different methods: backward selection and LASSO. I attempted to use AIC as a third method but found that it was too computationally expensive for my computers and so I abandoned this method in favor of the other two. I then compared the ROC curves and AUC for the three different models and chose the one I felt had the most predictive power based on the ROC curves, AUC and significance of included variables. I then found the associated misclassification error using the given fact that the cost of mislabelling a readmission is twice the cost of mislabelling a non-readmission. This gave me my final Bayes Rule Classification Threshold. Finally, I tested my model on my subset of test data.

## d. Main Findings 

The key variables of importance were found to be: num_procedures, num_medications, number_emergency, number_inpatient, A1Cresult, metformin, glimepiride, insulin, diabetesMed, disch_disp_modified, adm_src_mod, age_mod, diag1_mod, diag2_mod, diag3_mod. The Bayes Rule Classification Threshold using the risk ratio of 2:1 was 1/3. This means that if the predicted probability of readmission exceeds 1/3, we will predict that that individual gets readmitted. Our misclassification error was about 22%. 

## e. Limitations 

Readmissions is not a flawless indicator of hospital quality. Some of the advantages of this measure are: it has been shown that a high readmission rate suggests poor care, it has good face validity, is relatively easy to identify and we have a large amount of data regarding readmissions. However, it can be difficult to predict, is often shown to lack association with other hospital characteristics associated with quality of care. Most importantly, whether or not a patient is readmitted may be associated with factors outside the hospital's control, such as the patient's medication compliance level after she is discharged.   
Further, it may be interesting to see the 60 day and 90 day readmission data. 

# II.  Data Analysis

```{r library, include=FALSE}
# Libraries
library(dplyr)
library(ISLR)
library(tidyverse)
library(reshape2)
library(gridExtra)
library(car)
library(leaps)
library(glmnet)
library(xtable)
library(bestglm)
library(QuantPsyc)
library(pROC)
```

## a. Data Summary 

### Description of variables

The full dataset used covers ~50 different variables to describe every hospital diabetes admission. In this section we give an overview and brief description of the variables in this dataset.

**a) Patient identifiers:** 

a. `encounter_id`: unique identifier for each admission 
b. `patient_nbr`: unique identifier for each patient 

**b) Patient Demographics:** 

`race`, `age`, `gender`, `weight` cover the basic demographic information associated with each patient. `Payer_code` is an additional variable that identifies which health insurance (Medicare /Medicaid / Commercial) the patient holds.

**c) Admission and discharge details:** 

a.	`admission_source_id` and `admission_type_id` identify who referred the patient to the hospital (e.g. physician vs. emergency dept.) and what type of admission this was (Emergency vs. Elective vs. Urgent). 
b.	`discharge_disposition_id` indicates where the patient was discharged to after treatment.

**d) Patient Medical History:**

a.	`num_outpatient`: number of outpatient visits by the patient in the year prior to the current encounter
b.	`num_inpatient`: number of inpatient visits by the patient in the year prior to the current encounter
c.	`num_emergency`: number of emergency visits by the patient in the year prior to the current encounter

**e)	Patient admission details:**

a.	`medical_specialty`: the specialty of the physician admitting the patient
b.	`diag_1`, `diag_2`, `diag_3`: ICD9 codes for the primary, secondary and tertiary diagnoses of the patient.  ICD9 are the universal codes that all physicians use to record diagnoses. There are various easy to use tools to lookup what individual codes mean (Wikipedia is pretty decent on its own)
c.	`time_in_hospital`: the patient’s length of stay in the hospital (in days)
d.	`number_diagnoses`: Total no. of diagnosis entered for the patient
e.	`num_lab_procedures`: No. of lab procedures performed in the current encounter
f.	`num_procedures`: No. of non-lab procedures performed in the current encounter
g.	`num_medications`: No. of distinct medications prescribed in the current encounter

**f)	Clinical Results:**

a.	`max_glu_serum`: indicates results of the glucose serum test
b.	`A1Cresult`: indicates results of the A1c test

**g)	Medication Details:**

a.	`diabetesMed`: indicates if any diabetes medication was prescribed 
b.	`change`: indicates if there was a change in diabetes medication
c.	`24 medication variables`: indicate whether the dosage of the medicines was changed in any manner during the encounter

**h)	Readmission indicator:** 

Indicates whether a patient was readmitted after a particular admission. There are 3 levels for this variable: "NO" = no readmission, "< 30" = readmission within 30 days and "> 30" = readmission after more than 30 days. The 30 day distinction is of practical importance to hospitals because federal regulations penalize hospitals for an excessive proportion of such readmissions. I regrouped this variable into an indicator for whether or not the patient was readmitted within 30 days. 

See appendix for a summary of the variables in the full dataset. 

### Data cleaning

Since many of the values are missing, we will modify the dataset in the following ways: 

1) `Payer code`, `weight` and `Medical Specialty` are not included since they have a large number of missing values. 

2) Variables such as `acetohexamide`, `glimepiride.pioglitazone`, `metformin.rosiglitazone`, `metformin.pioglitazone` have little variability, and are as such excluded. This also includes the following variables: `chlorpropamide`, `acetohexamide`, `tolbutamide`, `acarbose`, `miglitor`, `troglitazone`, `tolazamide`, `examide`, `citoglipton`, `glyburide.metformin`, `glipizide.metformin`, and `glimepiride.pioglitazone`.

3) Some categorical variables have been regrouped. For example, `Diag1_mod` keeps some original levels with large number of patients and aggregates other patients as `others`. 

4) Observations for which `race` was given to be "?" were omitted for ease of analysis. Similar process for observation where gender was "Unknown/Invalid". In total, 2276 observations were omitted, representing 2.2% of the dataset. 

5) To ease in the analysis we removed `encounter_id` and `patient_nbr` as the large number of various values created issues and provided little predictive power. We are left with 29 variables in the dataset. 

6) The event of interest is **readmitted within < 30 days**. We have recoded those who were readmitted *beyond* 30 days such that they do not get counted under our event of interest. 

```{r, include = FALSE}
#Lets regroup the readmission indicator 
rdata <- read.csv("Data/readmission.csv")
rdata$read <- NA #new readmission variable 
#recoding those who were readmitted after 30 days to "NO" 
rdata$read[which(rdata$readmitted == ">30")] <- as.integer(0)
rdata$read[which(rdata$readmitted == "NO")] <- as.integer(0)
rdata$read[which(rdata$readmitted == "<30")] <- as.integer(1)
#summary(rdata$read)
rdata$readmitted <- NULL #remove original variable

rdata_clean <- filter(rdata, rdata$race != "?") #2273 omitted
rdata_clean <- filter(rdata_clean, rdata_clean$gender != "Unknown/Invalid") #3 omitted
rdata_clean <- rdata_clean %>% dplyr::select(-encounter_id, -patient_nbr)
```

### Graphical Summary (See Appendix)

## b. Analyses 

### Creating a Test set 
I retained 10% of the data (about 10,000 observations) as a testing set to assess my final model. This leaves almost 90,000 observations to train the model. 

```{r, include = FALSE}
#split data into 90:10 training/testing sets
smp_size <- floor(0.9 * nrow(rdata_clean))
set.seed(123)
train_ind <- sample(seq_len(nrow(rdata_clean)), size = smp_size)
rtrain_clean <- rdata_clean[train_ind, ]
rtest_clean <- rdata_clean[-train_ind, ]
```

### Backward Selection

The first method I will use is Backward Selection. Starting with the full model I successively remove the variable with the highest p-value (lowest significance), then run the logistic regression with the remaining variables. This process is repeated until I reach a model which has only variables significant at the 0.05 level. The best model is shown below with its associated Anova test results (`fit_b.best`). 

```{r, include= FALSE, eval = FALSE}
#backward selection 
fit_backward_all <- glm(read~.,rtrain_clean, family=binomial)
Anova(fit_backward_all)
fit_b.1 <- update(fit_backward_all, .~. -pioglitazone)
Anova(fit_b.1)
fit_b.2 <- update(fit_b.1, .~. -race)
Anova(fit_b.2)
fit_b.3 <- update(fit_b.2, .~. -glyburide)
Anova(fit_b.3)
fit_b.4 <- update(fit_b.3, .~. -time_in_hospital)
Anova(fit_b.4)
fit_b.5 <- update(fit_b.4, .~. -gender)
Anova(fit_b.5)
fit_b.6 <- update(fit_b.5, .~. -rosiglitazone)
Anova(fit_b.6)
fit_b.7 <- update(fit_b.6, .~. -num_lab_procedures)
Anova(fit_b.7)
fit_b.8 <- update(fit_b.7, .~. -max_glu_serum)
Anova(fit_b.8)
fit_b.9 <- update(fit_b.8, .~. -adm_typ_mod)
Anova(fit_b.9)
fit_b.10 <- update(fit_b.9, .~. -number_outpatient)
Anova(fit_b.10)
fit_b.11 <- update(fit_b.10, .~. -change)
Anova(fit_b.11) 
fit_b.12 <- update(fit_b.11, .~. -glipizide)
Anova(fit_b.12)
fit_b.13 <- update(fit_b.12, .~. -number_diagnoses)
Anova(fit_b.13) #fit_b.best
```

```{r}
fit_b.best <- glm(read~ num_procedures + num_medications + number_emergency + number_inpatient + A1Cresult + 
                    metformin + glimepiride + insulin + diabetesMed + disch_disp_modified + adm_src_mod + age_mod + 
                    diag1_mod + diag2_mod + diag3_mod, rtrain_clean, family=binomial)
Anova(fit_b.best)
```

```{r, include = FALSE}
# AIC
# xy.rdata <- model.matrix(read ~.+0, rtrain_clean)
# xy.rdata <- data.frame(xy.rdata, rtrain_clean$read)
# fit_aic_all <- bestglm(xy.rdata, family = binomial, method = "exhaustive", IC="AIC", nvmax = 10)
# fit_aic_all$BestModel
# fit_aic_best <- glm(read~..., hd_data.f, family=binomial)
# summary(fit.best)
# Anova(fit.best)
```
![alt text][image_0]

### LASSO in classifications:

Next, I will use LASSO to find the best model variables. 
The regularization techniques used in regression are readily applied to classification problems. For a given lambda we minimize -log liklihood/n + lambda |beta|
To remain consistent in both binary and continuous responses, glmnet() uses the following penalized least squares. 
RSS/(2n) + lambda |beta|

Shown below is the plot of the binomial deviance from the 10-fold cross validated LASSO model. We want to minimize this. 

```{r, echo = FALSE}
#LASSO (alpha = 1) 
X.readmissions <- model.matrix(read~., rtrain_clean)[,-1] #
Y.readmissions <- rtrain_clean$read
set.seed(123)
fit_lasso_cv <- cv.glmnet(X.readmissions, Y.readmissions, alpha=1, family="binomial", nfolds = 10, type.measure = "deviance")
plot(fit_lasso_cv)
```

![alt text][image_1]

![alt text][image_2]

```{r, include = FALSE}
# i): lambda.min
coef.min <-coef(fit_lasso_cv, s="lambda.min") 
coef.min <- coef.min[which(coef.min !=0), ]
as.matrix(coef.min)
# ii): lambda.1se
coef.1se <- coef(fit_lasso_cv, s="lambda.1se")  
coef.1se <- coef.1se[which(coef.1se !=0),] 
as.matrix(coef.1se)
```

```{r, include = FALSE}
fit_lasso_min <- glm(read~ race + gender + time_in_hospital +  num_lab_procedures + num_procedures + num_medications + number_outpatient +  
                       number_emergency + number_inpatient + number_diagnoses + max_glu_serum + A1Cresult + metformin + glimepiride + glipizide + 
                       glyburide + pioglitazone + rosiglitazone + insulin + change + diabetesMed + disch_disp_modified + adm_src_mod + adm_typ_mod + 
                       age_mod + diag1_mod + diag2_mod + diag3_mod,
                       rtrain_clean, family=binomial)
Anova(fit_lasso_min)
#summary(fit_lasso_min)

fit_lasso_1se <- glm(read~ time_in_hospital + num_medications + number_emergency + number_inpatient + number_diagnoses + diabetesMed + disch_disp_modified + 
                       diag1_mod + diag3_mod,
                       rtrain_clean, family=binomial)
Anova(fit_lasso_1se)
#summary(fit_lasso_1se)
```

```{r, echo=FALSE}
# ROC and AUC
fit_backward.roc <- roc(rtrain_clean$read, fit_b.best$fitted, plot=F, col="blue") 
fit_lasso_min.roc <- roc(rtrain_clean$read, fit_lasso_min$fitted, plot=F, col="blue")
fit_lasso_1se.roc <- roc(rtrain_clean$read, fit_lasso_1se$fitted, plot=F, col="blue")
plot(1-fit_backward.roc$specificities, fit_backward.roc$sensitivities, col="red", pch=16, cex=.7, 
     xlab="False Positive", 
     ylab="Sensitivity")
  points(1-fit_lasso_min.roc$specificities, fit_lasso_min.roc$sensitivities, col="blue", pch=16, cex=.6)
  points(1-fit_lasso_1se.roc$specificities, fit_lasso_1se.roc$sensitivities, col="black", pch=16, cex=.6)
  title("Red:Backwards Selection, Blue: LASSO Lambda Min, Black: LASSO Lambda 1SE")
auc(fit_backward.roc) #0.6551 
auc(fit_lasso_min.roc) #0.6561  
auc(fit_lasso_1se.roc) #0.6514 
```

auc(fit_backward.roc) = 0.6551 
auc(fit_lasso_min.roc) = 0.6561  
auc(fit_lasso_1se.roc) = 0.6514  

As we can see, all three models are fairly similar, with similar AUCs but I decided to go with the backward selection model since all of the variables in it are significant at the 0.05 level, which cannot be said about the LASSO models. 

Now that we have selected our final model of read ~ num_procedures + num_medications + number_emergency + number_inpatient + A1Cresult + metformin + glimepiride + insulin + diabetesMed + disch_disp_modified + adm_src_mod + age_mod + diag1_mod + diag2_mod + diag3_mod
                                                    
which was obtained through backward selection, we now need to come up with a reasonable classifier for our model. Based on a quick and somewhat arbitrary guess, it's estimated that it costs twice as much to mislabel a readmission than it does to mislabel a non-readmission. Based on this risk ratio, I will propose a specific classification rule to minimize the cost. Then: 

![alt text][image_3] 

Therefore the Bayes Rule Classification Threshold using this risk ratio would be 1/3. Using this threshold we get a weighted misclassification error (MCE) of 0.22. 

```{r, include=F}
# Summary of Selected Model
anova(fit_b.best)
#summary(fit_b.best)
# Bayes Rule Classifier
fit.backward.pred.bayes=rep("0", length(rtrain_clean$read))
fit.backward.pred.bayes[fit_b.best$fitted > 1/3]="1" 
MCE.bayes.backward=(sum(2*(fit.backward.pred.bayes[rtrain_clean$read == "1"] != "1")) + sum(fit.backward.pred.bayes[rtrain_clean$read == "0"] != "0"))/length(rtrain_clean$read)
MCE.bayes.backward
MCE.bayes.2 <- data.frame(matrix(0, 100, 2))
colnames(MCE.bayes.2) <- c("Threshold", "MCE")

for(i in 1:100) {
MCE.bayes.2[i, 1] <- i/100
fit.best.pred.bayes.2=rep("0", length(rtrain_clean$read))
fit.best.pred.bayes.2[fit_b.best$fitted > i/100]="1" 
MCE.bayes.2[i,2]=(sum(2*(fit.best.pred.bayes.2[rtrain_clean$read == "1"] != "1")) 
           + sum(fit.best.pred.bayes.2[rtrain_clean$read == "0"] != "0"))/
  length(rtrain_clean$read)
}
```

```{r}
ggplot(MCE.bayes.2, aes(x = Threshold, y = MCE)) + geom_point() +
    labs(title = "Threshold vs. MCE", x = "Threshold", y = "MCE")
```

![alt text][image_4]

### Evaluating my model using testing data

Get the fitted prob's using the testing data:
```{r, echo = FALSE}
fit.fitted.test <- predict(fit_b.best, rtest_clean, type="response") # fit1 prob
fit.test.roc <- roc(rtest_clean$read,fit.fitted.test, plot=T )
auc(fit.test.roc)
```
![alt text][image_5]

## c. Conclusion 

The key variables of importance were found to be: num_procedures, num_medications, number_emergency, number_inpatient, A1Cresult, metformin, glimepiride, insulin, diabetesMed, disch_disp_modified, adm_src_mod, age_mod, diag1_mod, diag2_mod, diag3_mod. The Bayes Rule Classification Threshold using the risk ratio of 2:1 was 1/3. This means that if the predicted probability of readmission exceeds 1/3, we will predict that that individual gets readmitted. Our misclassification error was about 22%.

# III.  Citation 
Data obtained from: [Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.] (https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) 

# IV. Appendix

A summary of the full dataset can be found in the writeup.
```{r, echo = FALSE}
#Loading Readmission Data 
fulldata <- read.csv("/Users/che/Documents/STAT471/Data/diabetic.data.csv")
summary(fulldata)
```

Here is a graphical summary of the 28 input variables and the response variable we retained in the working dataset `rdata_clean`. 


Distributions of Demographics 
```{r, echo = FALSE}
grid.arrange(p1, p2, p25, ncol = 3) #demographics
```
![alt text][image_6]

Distributions of Admission Details
```{r, echo = FALSE}
grid.arrange(p3, p4, p5, p6, p10, p26, p27, p28, ncol = 4) #admission details
```
![alt text][image_7]

Distributions of Medical History variables 
```{r, echo = FALSE}
grid.arrange(p7, p8, p9, ncol = 3) #medical history
```
![alt text][image_8]

Distribution of Clinical Results variables 
```{r, echo = FALSE}
grid.arrange(p12, p13, ncol = 2) #clinical results
```
![alt text][image_9]

Distribution of Medication details variables 
```{r, echo = FALSE}
grid.arrange(p11, p14, p15, p16, p17, p18, p19, p20, p21, ncol = 3) #medication details
```
![alt text][image_10]

Distribution of admission/discharge details variables 
```{r, echo = FALSE}
grid.arrange(p22, p23, p24, ncol = 3) #admission/discharge details
```
![alt text][image_11]

Distribution of response variable (readmissions)
```{r, echo = FALSE}
grid.arrange(p29) #response variable (readmission)
```
![alt text][image_12]
