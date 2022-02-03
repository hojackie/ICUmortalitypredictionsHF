# What Factors Influence Life or Death in the ICU? Hospital Mortality Predictions among ICU Patients with Heart Failure: Machine Learning vs. Clinician’s Best Guess

## Introduction
My name is Jackie Ho. I am a pharmacist and data enthusiast that took Springboard’s Data Analytics course. For my capstone I analyzed hospital mortality among ICU patients with heart failure using Python and developed a risk calculator based on model predictions. The data was retrieved from the MIMIC-III database courtesy of Zhou and colleagues.<sup>1</sup> 

### A little about myself
I first learned basic SQL at the VA Palo Alto during my pharmacy residency. I learned how to run simple queries for research projects and slowly started to apply this skill set as a process improvement/quality pharmacist at my workplace. 

Although healthcare data makes up 30% of the world’s data, with pharmacy services among the largest generator of this data, learning data analytics is not a requirement for most healthcare professionals.<sup>2</sup> Due to the complexity of the medication use process, it is increasingly important to have a comprehensive data management strategy as well as data literacy skillset – particularly within the pharmacy department. 

I soon realized my existing SQL and excel skills were limited and my growth as a pharmacy analyst had plateaued. Thus, I embarked on my data journey with Springboard. In my research, Springboard had the most well-rounded program that would help broaden my skills as a data analyst and my understanding of the field outside of healthcare. The true value of the course is in the weekly mentor calls with experienced data analysts – my mentor (Pasquale) helped guide me through projects and expand my thought process and approach to handling data. Thanks to Springboard’s data analytics course I’m one step closer to reaching my goal. 

I hope to one day develop a pharmacy analytics and outcomes team with the goal of leveraging data to drive measurable improvements in patient outcomes, operational efficiency and financial performance. 

### Why is heart failure (HF) a big deal?
In 2014, there were an estimated 1.1 million (4 million) emergency department visits, 1 million (3.4 million) hospitalizations, and 80 thousand (230 thousand) deaths with primary HF (comorbid HF). The estimated total cost of hospitalization for primary HF alone was >$11 billion in 2014.<sup>3</sup> Additionally, congestive HF had the highest hospital readmission rate (26.9%) among all medical and surgical readmissions. <sup>4</sup> Identifying the most poignant risk factors associated with inpatient hospital mortality among HF patients can lead to rapid identification and intervention on these high-risk patients. Machine learning algorithms can help create risk scores which can estimate outcome probability and help clinicians decide on treatment modalities. At the end of this exercise, I will compare the machine learning results vs. a clinician’s (my) best guess of the most important variables. 

## Methods
My initial goal was to explore the data set and complete a simple regression to identify obvious patterns/trends associated with mortality in the dataset. However, after completing exploratory data analysis in Python and Tableau, it was challenging to see any obvious patterns as there were many variables (49) at play. The initial linear regression model using all 49 variables yielded a low AUC of 0.26 which begs the question whether linear regression is the appropriate model. This led me down a path to exploring predictive models using machine learning. The following are steps I used to complete this project.

### Step 1: Preparing the data

#### Drop unnecessary variables
Drop any variables that are unnecessary (e.g., ID numbers).

#### Check for missing variables 
There are various suggested methods on how to work with missing continuous variables. The most common method is to replace missing data with the mean, followed by the median or a random number. In some cases, all three methods were suggested with the end goal to have the final data set match closely in distribution with the original dataset. 

To accomplish this, I created 2 data arrays. One data array for the original data set with missing variables. The second data array with missing variables replaced by the mean (imputed data array). I used a loop to graph each variable in the original vs. imputed data array to see how close the imputed data distribution was to the original. As seen below, replacing missing values with mean worked well for some but not all variables. 
 

![1](https://user-images.githubusercontent.com/87457603/152285176-a8639221-2648-4c24-a8db-f9780ee32c9a.png)
![2](https://user-images.githubusercontent.com/87457603/152285178-a8b215b0-b41a-46b7-bbeb-42aa4d882afd.png)
![3](https://user-images.githubusercontent.com/87457603/152285179-c33bfe72-dcc8-4b20-a89f-ffd74945adeb.png) 

I then took the variables that did not match closely with the imputed data array (using the mean replacement method above) and repeated the process using the median replacement method. Unfortunately, none of the imputed median variables came close to the distribution of the original dataset. 

Lastly, I used the random variable replacement method to replace null values. The random values were selected based on the normal distribution of existing variables. The data distribution using the random variable replacement method matched the original data set very closely!

After this testing, I replaced the missing variables in the dataset with mean and random variables as tested above. 

Another way to approach missing data is to determine if the variable is clinically relevant. The goal is to determine risk of mortality for heart failure patients in the ICU. Is depression status routinely evaluated? Is it clinically relevant? 

### Step 2: Exploratory data analysis

Seaborn vs. Tableau Pair Plot
I used Python’s Seaborn pair plot and heatmap functions to see relationships between each combination of variables. Due to the sheer number of variables it is difficult to see a pattern. Tableau may provide a better visualization then Python in this case. See the Tableau interactive pair plot [here](https://public.tableau.com/shared/TRGKHJ825). 

#### Python’s Seaborn Pair Plot
![4](https://user-images.githubusercontent.com/87457603/152285180-85debdac-04ec-443c-9e50-866285badc5c.png)

#### Tableau’s Pair Plot
![5](https://user-images.githubusercontent.com/87457603/152285182-cd4d73c3-f3a5-447f-840a-7d389f9a5645.png)

#### Continuous Variables (Histograms & Box Plots) 
Using Python’s convenient loop function, I graphed out box plots and histograms to visualize the distribution of continuous variables. At first glance the means of each variable appears to be similar between the alive and dead outcomes group. It was difficult to discern any significant patterns in the histogram distribution. 
![6](https://user-images.githubusercontent.com/87457603/152285183-0052d527-dc3c-4c15-a35e-8a39d376b1a3.png)
![7](https://user-images.githubusercontent.com/87457603/152285184-28c868cb-1ce3-45a0-a38a-30a6d6260ab8.png)
   
#### Categorical Variables (Bar Chart) 
For categorical variables, I used bar charts in Python and a bubble chart in Tableau. As seen below, the alive group appears to have higher % of comorbidities compared to the dead group except in atrial fibrillation where the dead group has a prevalence of 57.9% vs. 43.2% in the alive group. 

I was also curious about the number of comorbidities that each patient had and its affect on mortality. 
At first glance it appears that having more comorbidities results in less death. However, this can be confounded by multiple factors including disease severity, initial reason for admission, sample size, etc. 

This is one of the reasons why a deeper dive on how all the variables contribute to mortality as a whole is needed. Tableau chart [here](https://public.tableau.com/shared/5HF3X7M2B).

![8](https://user-images.githubusercontent.com/87457603/152285186-e2f65bb8-3e09-4621-8366-33b7f7dac71c.png)
![9](https://user-images.githubusercontent.com/87457603/152285187-957782a7-7bab-4055-91c0-db8616418182.png)
 

## Results

### Initial Linear & Logistic Regression Using All Variables
An initial linear regression using all variables resulted in a very low AUC of 0.26 – meaning only ~26% of the observed variation can be explained by the model’s input. Linear regression is likely not the best model for mortality predictions for this dataset. Of note logistic regression yielded an AUC of 0.77, which shows promise. 

Inputting all variables into a model is impractical. In clinical practice, clinicians will likely not have time to input 48 variables into a prediction calculator to assess risk. Most of the time, clinicians may not even have data on all the variables for the patient. Moreover, creating a model with large number of variables can result in overfitting - where the model learns the detail and noise of the current data set vs. new data. So how do we decide how many and which variables should be inputted into the data model?

### Variable Selection Using Forward Stepwise Approach
At this point I went to DataCamp and started reviewing their machine learning course work (see “Suggested Resources” section). 

Using the forward stepwise selection method, I determined a list of candidate variables from most to least significant. The model begins with no variables, then starts adding the most significant variables one after another until all variables have passed through the model (or a stopping rule has been reached). 
![10](https://user-images.githubusercontent.com/87457603/152285188-5a58d9d0-931e-43da-975e-9ffecd05c0be.png)
 
### Train and Test Split
Next, I used the train-test split method to evaluate the machine learning model. Common train-test split percentages used include 50-50, 70-30 and 80-20. I opted to use a 70-30 split. 

### Deciding the Number of Variables
I graphed the AUC of the test and train data set using the variables identified in the forward stepwise approach. As more variables are added, the training dataset (red line) will rise in AUC but the test dataset (blue line) AUC will fall or have diminishing returns due to overfitting. As seen below, diminishing AUC is seen in the test dataset (blue line) at “CHD with MI” variable. Our new model will include all variables before “CHD with MI”. 
![11](https://user-images.githubusercontent.com/87457603/152285189-e69d9139-830a-4257-a671-0ad882e0cc5c.png)
 

### Building and Comparing Predictive Models 
Using the final set of independent variables, I train-test split the data again and ran it through different models using the ensemble approach to determine which model gives you the best AUC with least error. 

![12](https://user-images.githubusercontent.com/87457603/152286916-0f15253e-cf39-4f16-a720-f8b0ded5e12d.PNG)
![13](https://user-images.githubusercontent.com/87457603/152285192-0fb047ea-734b-438b-b0e1-536edbe0dd70.png)
    

Our winner is the logistic regression model with an AUC of 87%. There are of course many more models and boosting techniques to increase the AUC score of machine learning models. This capstone scratches the surface but hopefully provides some insights on the different possibilities with this data set. 

## Discussion 
Real World Application – Interactive Risk Calculator
A risk calculator was created to test out the model in real time. You can use the calculator in the notebook [here](https://github.com/hojackie/ICUmortalitypredictionsHF/blob/main/ICU%20Mortality%20Predictions%20in%20HF%20Patients.ipynb)! 

![14](https://user-images.githubusercontent.com/87457603/152285193-d91974b3-7893-48b4-8e39-cab43910a1fa.png) 

As you can probably imagine, 32 variables are a lot of variables to enter into a calculator. For further refinement of the model – I would recommend minimizing the number of variables and see if you can still maintain good AUC with reduced error rates. 

From a clinical standpoint, some variables did not seem as significant as others (e.g., depression). There are some variables that may be clinically significant not included in the model (e.g., temperature). 

### Forward Stepwise Approach vs. Clinician Picks the Variables
For fun, I decided to pick a few variables that I thought were significant to mortality (a clinician’s best guess) and compare the AUC generated vs. using the forward stepwise approach. I attempted to remove what I thought were “duplicate” variables. For example: if creatinine is high it is likely the patient has some form of renal failure. Thus, I would either choose creatinine or renal failure vs. looking at both variables. I also prioritized variables that were routinely available upon initial admission to the ICU. Here are the results: 
![15](https://user-images.githubusercontent.com/87457603/152285194-3bdfd126-4333-48a4-a8ff-3c96ad0ac934.png) 

The accuracy and error rates are quite similar between the two approaches with the Clinician’s pick having less variables to contend with. 

## Final Thoughts
This capstone definitely challenged me and furthered my growth in data analytics. At the start of this project I did not anticipate needing to learn about predictive modeling or machine learning. This is a good example of how there is no one size fits all when it comes to analyzing data, each problem can be approached from different angles. As this is my first attempt at predictive modeling – I welcome any comments or constructive feedback. 


## Suggested Resources
Here are some resources I used to learn about predictive modeling: 

1.	DataCamp Introduction to Predictive Analytics in Python: [https://app.datacamp.com/learn/courses/introduction-to-predictive-analytics-in-python](https://app.datacamp.com/learn/courses/introduction-to-predictive-analytics-in-python)

2. DataCamp Intermediate Predictive Analytics in Python [https://app.datacamp.com/learn/courses/intermediate-predictive-analytics-in-python](https://app.datacamp.com/learn/courses/intermediate-predictive-analytics-in-python)

3.	DataCamp Ensemble Methods in Python [https://app.datacamp.com/learn/courses/ensemble-methods-in-python](https://app.datacamp.com/learn/courses/ensemble-methods-in-python)

4.	DataCamp Machine Learning with Tree-Based Models in Python [https://app.datacamp.com/learn/courses/machine-learning-with-tree-based-models-in-python](https://app.datacamp.com/learn/courses/machine-learning-with-tree-based-models-in-python)

5.	DataCamp Supervised Learning with scikit-learn [https://app.datacamp.com/learn/courses/supervised-learning-with-scikit-learn](https://app.datacamp.com/learn/courses/supervised-learning-with-scikit-learn)

6.	Understanding Forward and Backward Stepwise Regression: [https://quantifyinghealth.com/stepwise-selection/](https://quantifyinghealth.com/stepwise-selection/)

7.	AUC-ROC Curve in Machine Learning Clearly Explained [https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/#:~:text=The%20Area%20Under%20the%20Curve,the%20positive%20and%20negative%20classes](https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/#:~:text=The%20Area%20Under%20the%20Curve,the%20positive%20and%20negative%20classes)

8.	Google’s Machine Learning Crash Course [https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)


## Acknowledgements
Special thanks to my mentor Pasquale Prosperati for providing guidance on this project.

## References

1.	Li F, Xin H, Zhang J, Fu M, Zhou J, Lian Z. Prediction model of in-hospital mortality in intensive care unit patients with heart failure: machine learning-based, retrospective analysis of the MIMIC-III database. BMJ Open. 2021 Jul 23;11(7):e044779. doi: 10.1136/bmjopen-2020-044779. 

2.	Huesh MD, Mosher TJ. Using it or losing it? The case for data scientists inside health care. NEJM Catalyst website. [https://catalyst.nejm.org/doi/abs/10.1056/CAT.17.049](https://catalyst.nejm.org/doi/abs/10.1056/CAT.17.049). Accessed December 21, 2021.

3.	Jackson SL, Tong X, King RJ, Loustalot F, Hong Y, Ritchey MD. National Burden of Heart Failure Events in the United States, 2006 to 2014. Circ Heart Fail. 2018 Dec;11(12):e004873. doi: 10.1161/CIRCHEARTFAILURE.117.004873. 

4.	Nair R, Lak H, Hasan S, Gunasekaran D, Babar A, Gopalakrishna KV. Reducing All-cause 30-day Hospital Readmissions for Patients Presenting with Acute Heart Failure Exacerbations: A Quality Improvement Initiative. Cureus. 2020 Mar 25;12(3):e7420. doi: 10.7759/cureus.7420. 

## Author
![16](https://user-images.githubusercontent.com/87457603/152285196-1714ecca-ced1-496e-a9c2-69c9c017176d.png)

Jackie Ho, Pharm.D., BCPS, MPH

[LinkedIn](https://www.linkedin.com/in/hojackie/) [Github](https://github.com/hojackie) [Tableau](https://public.tableau.com/app/profile/jackie.ho8166) [GoogleScholar](https://scholar.google.com/citations?user=Jq70VXsAAAAJ&hl=en&oi=ao)

