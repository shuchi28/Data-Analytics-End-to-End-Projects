# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)

# %%
df= pd.read_excel("Game.xlsx")

# %%
df.head()

# %%
df.info()

# %% [markdown]
# Objective and Insights
# These are the instructions that i coppied from excel file
# 
# Determine the outliers for Wages and mentioned the steps, process and logic
# 
# Analyze the distribution for potential column.
# 
# Difference between normal and student t distribution explain it using 'potential' column.
# 
# Difference between normal and standard normal distribution explain it using 'potential' column.
# 
# find the 95%, 90%, and 99%, confidence interval for 'Potential','wage','weight' column.
# 
# find the 95%, 90%, and 99%, confidence interval for 'Potential','wage','weight' column.
# 
# Proove Central Limit Theorom by using 'potential' column of the game_data.
# 
# Pls give any insgights by anlysing the data in your own and make PPT out of it (Currently we are not checking PPT skills so you can paste the graphs and write insights
# 
# I am going to load only the specific columns required to complete the tasks mentioned in the instructions and include 2-3 extra columns for additional analysis

# %%
df.isnull().sum()/len(df)*100

# %%
# Only these columns i will load from fame excel file
col = ['ID','Name', 'Age', 'Overall', 'Club', 'Value', 'Wage', 'Potential', 'Weight']

# %%
df=pd.read_excel("Game.xlsx",usecols=col)

# %%
df

# %%
df.info()

# %%
df.isnull().sum()/len(df)*100

# %% [markdown]
# # Determine the outliers for Wages and mentioned the steps, process and logic

# %%
df['Wage'].describe()

# %%
df['Wage'].head()

# %%
df['Wage']=df['Wage'].str.replace('€','')
df['Wage']=df['Wage'].str.replace('K','000')
df['Wage'] = df['Wage'].str.replace('M', '000000')

# %%
df['Wage']=df['Wage'].astype(float)

# %%
# outlier detection using IQR METHOD (INTER QUANTILE RANGE)

# %%
Q1 = df['Wage'].quantile(0.25)
Q3 = df['Wage'].quantile(0.75)

# %%
Q1

# %%
Q3

# %%
IQR= Q3-Q1
print(IQR)

# %%
# upper limit and lower limit 

upper_limit = Q3+1.5*(IQR)
print(upper_limit)

lower_limit = Q1-1.5*(IQR)
print(lower_limit)

# %%
outliers = df[(df['Wage']>upper_limit) | (df['Wage']<lower_limit) ]

# %%
outliers

# %%
plt.figure(figsize=(20,12))
plt.title("Wage Outlier Distrbution")
sns.boxplot(df['Wage']);

# %%
print(outliers['Wage'].max())
print(outliers['Wage'].min())

# %% [markdown]
# # 2. Analyze the distribution for potential column.

# %%
plt.figure(figsize=(20,10))
plt.title("Potential Distribution")
plt.subplot(1,2,1)
sns.distplot(df['Potential'])

plt.subplot(1,2,2)
plt.title("Potential Distribution")
sns.histplot(df['Potential'],kde=True);

# %%
from scipy import stats as st 

# %%
import pylab

# %%
plt.figure(figsize=(20,10))
plt.title("Potential Distribution")
st.probplot(df['Potential'],dist='norm',plot=pylab);

# %%


# %% [markdown]
# Conclusions:-
# 
# The 'Potential' column contains 18,207 data points.
# 
# The mean potential is approximately 71.31, with a standard deviation of around 6.14.
# 
# The minimum potential is 48, and the maximum potential is 95.
# 
# The histogram and density plot both exhibit a bell-shaped curve, indicating a normal distribution.
# 
# The Q-Q plot shows a nearly straight line, suggesting the data points follow a normal distribution.
# 
# Overall, the 'Potential' column appears to be normally distributed.

# %% [markdown]
# # 3 Difference between normal and student t distribution explain it using 'potential' column

# %% [markdown]
# Normal Distrbution
# 
# ~~ normal distribution which is also known as Gaussian Distribution and it is a countinous probbility distribution with a Bell shaped curve
# 
# ~~ Mean ,median , mode of 'Potential' are nearly equals and that indicates it follow normal distribution
# 
# ~~ Normal distribution should follow Emperical Rule
# 
# ~~ In the 'Potential' column context, the normal distribution would describe how the 'Potential' values are distributed around their mean value.
# 
# ~~ The majority of 'Potential' values would be concentrated around the mean, and the distribution would be symmetric.
# 

# %% [markdown]
# Student t distribution
# 
# ~~ The Student's t-distribution is also a continuous probability distribution, but it has heavier tails than the normal distribution.
# 
# ~~ It is used when the sample size is small or when the population standard deviation is unknown.
# 
# ~~ In the 'Potential' column context, the Student's t-distribution might be relevant if we are dealing with a small sample of 'Potential' values or if the population standard deviation is not known.

# %% [markdown]
# # 4. Difference between normal and standard normal distribution explain it using 'potential' column.

# %% [markdown]
# Normal Distrbution
# 
# ~~ normal distribution which is also known as Gaussian Distribution and it is a countinous probbility distribution with a Bell shaped curve
# 
# ~~ Mean ,median , mode of 'Potential' are nearly equals and that indicates it follow normal distribution
# 
# ~~ Normal distribution should follow Emperical Rule
# 
# ~~ In the 'Potential' column context, the normal distribution would describe how the 'Potential' values are distributed around their mean value.
# 
# ~~ The majority of 'Potential' values would be concentrated around the mean, and the distribution would be symmetric.

# %% [markdown]
# Standard Normal Distribution:
# 
# ~~The standard normal distribution is a specific type of normal distribution with a mean of 0 and a standard deviation of 1.
# 
# ~~To convert data from a normal distribution with any mean and standard deviation to the standard normal distribution, we can use a process called standardization.
# 
# ~~In standardization, we use z-scores, which represent how many standard deviations each 'Potential' value is away from the mean of the original distribution.
# 
# ~~By standardizing the data, the resulting dataset will have a mean of 0 and a standard deviation of 1.

# %%
mean_potential = df['Potential'].mean() 
std_potential = df['Potential'].std() 

# Calculate z-scores for 'Potential' data
z_scores = st.zscore(df['Potential'])

# Plot density plot (KDE) of the 'Potential' data before standardization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(df['Potential'], fill=True)
plt.xlabel('Potential')
plt.ylabel('Density')
plt.title('Original Distribution of Potential')

# Plot density plot (KDE) of the standardized 'Potential' data (Z-Scores)
plt.subplot(1, 2, 2)
sns.kdeplot(z_scores, fill=True)
plt.xlabel('Standardized Z-Scores')
plt.ylabel('Density')
plt.title('Standard Normal Distribution (Z-Scores)')
plt.show()

# %% [markdown]
# # 5 CENTRAL LIMIT THEOREM ON POTENTIAL
# 

# %% [markdown]
# The central limit theorem says that the sampling distribution of the mean will always be normally distributed, as long as the sample size is large enough.

# %%
# Define the sample sizes
sample_sizes = [5, 30, 50]

# Initialize a dictionary to store sample means for each sample size
sample_means_dict = {size: [] for size in sample_sizes}
sample_means_dict

# %%
# Perform sampling and calculate sample means for each sample size
for size in sample_sizes:
    for _ in range(1000):  # Taking 1000 random samples for each size
        sample = df['Potential'].sample(size, replace=True)  # Sampling with replacement(1 data points may be select two times)
        sample_mean = np.mean(sample)
        sample_means_dict[size].append(sample_mean)
        
pd.DataFrame(sample_means_dict)

# %%
# Plot KDE plots of the sample means for each sample size
plt.figure(figsize=(12, 6))
for size in sample_sizes:
    sns.kdeplot(sample_means_dict[size], fill=True, label=f'Sample Size {size}')
plt.xlabel('Sample Mean (Potential)')
plt.ylabel('Density')
plt.title('Central Limit Theorem - Sample Means Distribution')
plt.legend()
plt.show()

# %% [markdown]
# We can see clearly here that the distribution approaches normal as sample size gets larger.
# It is evident from the graphs that as we keep on increasing the sample size from 5 to 50 the histogram tends to take the shape of a normal distribution.

# %% [markdown]
# # Getting top 10 players according to potential

# %%
High_performance_player = df.sort_values(by='Potential',ascending=False)

# %%
H_P_P=High_performance_player.nlargest(10,'Potential')

# %%
H_P_P

# %%
plt.figure(figsize=(20,10))
sns.barplot(H_P_P['Name'],H_P_P['Potential'],data=H_P_P);
for i , v in enumerate(H_P_P['Potential']):
    plt.text(x=i,y=v+29,s=f"{v}")

# %% [markdown]
# # Comparison of Overall and Potential Ratings

# %%
plt.figure(figsize=(20,10))
plt.title("Overall and Potential Ratings")
sns.scatterplot(df['Overall'],df['Potential'],color='black',data=df)
plt.plot();

# %% [markdown]
# player who have high potential also have high overall performance!


