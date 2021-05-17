

import pandas as pd
data = pd.read_csv("gap.tsv", sep='\t')
data.head()

	country 	continent 	year 	lifeExp 	pop 	gdpPercap
0 	Afghanistan 	Asia 	1952 	28.801 	8425333 	779.445314
1 	Afghanistan 	Asia 	1957 	30.332 	9240934 	820.853030
2 	Afghanistan 	Asia 	1962 	31.997 	10267083 	853.100710
3 	Afghanistan 	Asia 	1967 	34.020 	11537966 	836.197138
4 	Afghanistan 	Asia 	1972 	36.088 	13079460 	739.981106

#Exercise 1: "Make a scatter plot of life expectancy across time"
data.plot.scatter(x='year', y = 'lifeExp')

<AxesSubplot:xlabel='year', ylabel='lifeExp'>

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import statsmodels
import seaborn as sns
import sklearn
from sklearn import datasets, linear_model, metrics
import statsmodels.formula.api as smf

data['year'].unique
sns.violinplot(x = 'year', y = 'lifeExp', data = data, bw =.3 )
plt.title("Violin plot of Life Expectancy vs year")
plt.show()

#Question 2
# How would you describe the distribution of life expectancy across countries for individual years? Is it skewed, or not? Unimodal or not? Symmetric around it’s center?

#---Answer---# 
#The distribution of life expectancy is not skewed, unimodal, and not symmetric. There is a general upwards trend of life expectancy. The distribution of life expectancy across countries show that data was collected on specific years, but overall just shows a general upwards trend.

# Question 3: Suppose I fit a linear regression model of life expectancy vs. year (treating it as a continuous variable), and test for a relationship between year and life expectancy, will you reject the null hypothesis of no relationship? (do this without fitting the model yet. I am testing your intuition.)

#---Answer---#
#Yes I will reject the null hypothesis of no relationship because as shown in the graph the general trend seems to show that there is a strong direct relationship between year and life expectancy. In the violin plot, it is very clear with the larger end of the clusters being higher in higher years and lower in lower years. Therefore there would be no reason to claim the null hypothesis was true.

# Question 4: What would a violin plot of residuals from the linear model in Question 3 vs. year look like? (Again, don’t do the analysis yet, answer this intuitively)

#--------Answer---------#
# The violin plot of residuals compared to the linear model in Q3 will show a linear relationship since we also expect a linear relationship between the years and life expectancy

# Question 5: According to the assumptions of the linear regression model, what should that violin plot look like? That is, consider the assumptions the linear regression model you used assumes (e.g., about noise, about input distributions, etc); do you think everything is okay?

#----------Answer---------# 
# The violin plot should show a direct relationship between year and life expectancy, I thinkk everything should be okay. The plot of residuals should be centered closer to 0 as time goes on and have a higher average value than before because the graph for year vs life expectancy has more higher values as time goes on.

# Exercise 2: Fit a linear regression model using, e.g., the LinearRegression function from Scikit-Learn or the closed-form solution, for life expectancy vs. year (as a continuous variable). There is no need to plot anything here, but please print the fitted model out in a readable format.

import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data_cp = data.drop(columns =['country', 'continent','pop', 'gdpPercap'])
data_cp
#get the values
x_value = data_cp.iloc[:, :-1].values
y = data_cp.iloc[:, 1].values
#train the values
X_train, X_test, y_train, y_test = train_test_split(x_value, y, test_size=0.2, random_state=0)
#do a linear regression
linRegress = LinearRegression()
#Fit it to the graph
linRegress.fit(X_train, y_train)
print("Coefficient: %.9f" % linRegress.coef_)
print("Intercept: %.9f" % linRegress.intercept_)
print("Squared Err: %.2f" % np.mean((linRegress.predict(x_test) - y_test) ** 2))
print('Variance: %.2f' % linRegress.score(x_test, y_test))

#make a scatter plot
data.plot.scatter(x='year', y = 'lifeExp')
#add the linear regression line
plt.plot(x_value, Y_pred, color='red')
plt.show()

Coefficient: 0.318266770
Intercept: -570.679472281
Squared Err: 124.43
Variance: 0.22

# Question 6: On average, by how much does life expectancy increase every year around the world?
#---Answer---# 
# Life expectancy increases by 0.31826677 years every year based on the coefficient of the linear regression line given.

# Question 7: Do you reject the null hypothesis of no relationship between year and life expectancy? Why?

#---Answer---#
# Yes, I reject the null hypothesis becasue the p-value is under.05. By rule of p-value rejection hypothesis in statistics, if the p-value is below .05 then it will reject that there is no relationship. In addition, there visually appears to be a direct relationship between life expectancy and years.

# Exercise 3: Make a violin plot of residuals vs. year for the linear model from Exercise 2.
# group data by year
data['year'].unique
#make a violin plot
data['residual'] = data['lifeExp'] - (regressor.intercept_ + regressor.coef_[0]*data['year'])

sns.violinplot(x = 'year', y = 'residual', data = data, bw =.3 )
plt.title("Violin plot of residuals vs year")
plt.show()

# Question 8: Does the plot of Exercise 3 match your expectations (as you answered Question 4)?

#Yes the plot of Exercise 3 matches my expectations because although we are plotting residuals it should still show the probability distr. of the relationship between life expectancy and years. The ones with 

# Exercise 4: Make a boxplot (or violin plot) of model residuals vs. continent.

sns.boxplot(y = 'residual', x = 'continent', data = data)
plt.title("Box Plot of Residuals of Life Expectancy vs Continents")
plt.show()

# Question 9: Is there a dependence between model residual and continent? If so, what would that suggest when performing a regression analysis of life expectancy across time?

#---------Answer------------#
# There is a dependence between model residual and continent. This suggests that over time the linear regression line will not be reflective of life expectancy for specific continents rather than others over time.

# Exercise 5: As in the Moneyball project, make a scatter plot of life expectancy vs. year, grouped by continent, and add a regression line. The result here can be given as either one scatter plot per continent, each with its own regression line, or a single plot with each continent's points plotted in a different color, and one regression line per continent's points. The former is probably easier to code up.
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
all_continents = data['continent'].unique()
for i in all_continents:
    continent_data = data[data["continent"] == i]
    #Remove unnecessary data columns
    continent_data = continent_data.drop(columns =['country', 'continent','pop', 'gdpPercap', 'residual'])

    #get the correct data
    x_value = continent_data.iloc[:, :-1].values
    y = continent_data.iloc[:, 1].values

    #train the linear regression model
    x_trainer, x_test, y_trainer, y_test = train_test_split(x_value, y, test_size=0.2, random_state=0)

    linRegress = LinearRegression()
    linRegress.fit(x_trainer, y_trainer)
    #print(linRegress.intercept_)
    print("Coefficient: %.9f" % linRegress.coef_)
    print("Squared Err: %.2f" % np.mean((linRegress.predict(x_test) - y_test) ** 2))
    print('Variance: %.2f' % linRegress.score(x_test, y_test))
    continent_data.plot.scatter(x='year', y = 'lifeExp')
   # plt.plot(x_value, Y_pred, color='red')
    plt.title(i)
    plt.plot(x_test, linRegress.predict(x_test), color='red')
    plt.show()

Coefficient: 0.457255590
Squared Err: 88.57
Variance: 0.39

Coefficient: 0.229704389
Squared Err: 9.71
Variance: 0.48

Coefficient: 0.286811232
Squared Err: 56.28
Variance: 0.27

Coefficient: 0.355796610
Squared Err: 36.23
Variance: 0.55

Coefficient: 0.199107537
Squared Err: 1.24
Variance: 0.90

# Question 10: Based on this plot, should your regression model include an interaction term for continent and year? Why?

#------------Answer---------------#
# Yes the regression model should include an interaction term for each continent because it is very apparent based off of the individual graphs that the slopes are different depending on the the continent. For example, Europe's linear regression slope is very different from Oceania's

# Exercise 6: Fit a linear regression model for life expectancy including a term for an interaction between continent and year. Print out the model in a readable format, e.g., print the coefficients of the model (no need to plot). Hint: adding interaction terms is a form of feature engineering, like we discussed in class (think about, e.g., using (a subset of) polynomial features here).
all_years = data['year'].unique()
mean_acc = []
year_acc = []
continent_acc = []
for i in all_continents:
    sum = 0
    counter = 0
    for y in all_years:
        mean = data.loc[ (data['continent'] == i ) & (data['year'] == y )].lifeExp.mean(axis =0)
        continent_acc.append(i)
        year_acc.append(y)
        mean_acc.append(mean)

continent_mean_year_data = pd.DataFrame(columns=[])
continent_mean_year_data.insert(0, 'continent', continent_acc)
continent_mean_year_data.insert(1, 'year', year_acc)
continent_mean_year_data.insert(2, 'mean', mean_acc)

temp = []
for index, row in data.iterrows():
    t = continent_mean_year_data.loc[ (continent_mean_year_data['year'] == row['year']) & (continent_mean_year_data['continent'] == row['continent'])]['mean']
    y = 0.5 * (regressor.intercept_ + regressor.coef_[0]*row['year']) + (0.5 * float(t))
    temp.append(y )
data['Cont-Yr-Interaction'] = temp
data

	country 	continent 	year 	lifeExp 	pop 	gdpPercap 	residual 	Cont-Yr-Interaction
0 	Afghanistan 	Asia 	1952 	28.801 	8425333 	779.445314 	-21.776263 	48.445828
1 	Afghanistan 	Asia 	1957 	30.332 	9240934 	820.853030 	-21.836597 	50.743571
2 	Afghanistan 	Asia 	1962 	31.997 	10267083 	853.100710 	-21.762931 	52.661577
3 	Afghanistan 	Asia 	1967 	34.020 	11537966 	836.197138 	-21.331265 	55.007452
4 	Afghanistan 	Asia 	1972 	36.088 	13079460 	739.981106 	-20.854598 	57.130934
... 	... 	... 	... 	... 	... 	... 	... 	...
1699 	Zimbabwe 	Africa 	1987 	62.351 	9216418 	706.157306 	0.634400 	57.530694
1700 	Zimbabwe 	Africa 	1992 	60.377 	10704340 	693.420786 	-2.930934 	58.468755
1701 	Zimbabwe 	Africa 	1997 	46.809 	11404948 	792.449960 	-18.090268 	59.248768
1702 	Zimbabwe 	Africa 	2002 	39.989 	11926563 	672.038623 	-26.501602 	59.907916
1703 	Zimbabwe 	Africa 	2007 	43.487 	12311143 	469.709298 	-24.594935 	61.443987

1704 rows × 8 columns

import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression

data_cp = data.drop(columns =['continent', 'country', 'pop', 'gdpPercap', 'residual', 'lifeExp'])

#get the values
x_value = data_cp.iloc[:, :-1].values
y = data_cp.iloc[:, 1].values

#train the linear regression model
x_trainer, x_test, y_trainer, y_test = train_test_split(x_value, y, test_size=0.2, random_state=0)
linRegress = LinearRegression()
# fit it to the model
linRegress.fit(x_trainer, y_trainer)
#make a scatter plot
plt.scatter(x_test, y_test,  color='blue')
#Add the regression line
plt.plot(x_test, linRegress.predict(x_test), color='red')
#General knowledge of the graph
print("Coefficient: %.9f" % linRegress.coef_)
print("Squared Err: %.2f" % np.mean((linRegress.predict(x_test) - y_test) ** 2))
print('Variance: %.2f' % linRegress.score(x_test, y_test))
plt.show()

Coefficient: 0.318641178
Squared Err: 21.68
Variance: 0.59

# Question 11: Are all parameters in the model significantly different from zero? If not, which are not significantly different from zero?

#----------Answer----------#
# None of the parameters are significantly different from 0. The one that is not significantly different fro zero is the life expectancy mean based for each continent because it was used to make the continent interaction factor.

# Question 12: On average, by how much does life expectancy increase each year for each continent? (Provide code to answer this question by extracting relevant estimates from model fit)

#-----------Answer-------------#
# 1. Asia: 0.45725559
# 2. Europe: 0.22970439
# 3. Africa: 0.28681123
# 4. Americas: 0.35579661
# 5. Oceania: 0.19910754
# Taken from exercise 5

# Exercise 7: Make a residuals vs. year violin plot for the interaction model. Comment on how well it matches assumptions of the linear regression model.

data['year'].unique
data['residual_interaction'] = data['Cont-Yr-Interaction'] - (regressor.intercept_ + regressor.coef_[0]*data['year'])
sns.violinplot(x = 'year', y = 'residual_interaction', data = data, bw =.3 )
plt.title("Violin plot of Residual Interaction vs year")
plt.show()

# ---------------Analysis-----------#
# The residual plot clearly reflects the unbalanced life expectancy based on continent. As you can see there is a clear trend of data separation as you go foward in time. This suggests that there is a huge separation in life expectancy between continents.

 

