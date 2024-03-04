#the following demonstrations in A and C show that (based on empirical evidence):
#a. the OLS with the covariate is equivalent to the CUPED
#-- dependent and independent variance are the same across two methods
#b. if purely random assignment, can just use the treatment and use the cross-sectional comparison (no need for the pre-outocme)


#-------------------------------------------------------------------------------------------#
#A. OLS with clustered standard errors is equivalent to the variance from the delta method  #
# in delta method, need to change the metrics to the same unit (i.e. click through rate)    #
# in OLS, the metrics is the user_level clicks (i.e. number of clicks)                      #
# here, no consider of CUPED yet (simple cross-sectional data)                              #
#-------------------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from scipy import stats

#-------------------------------#
# simulate the actual case      #
#-------------------------------#
user_mean=0.3 #average click rate (click/pape view) in the pop
user_sd=0.15 #standard deviation of the click rate in the pop

users=500
pre_pageview=1000
pageview=1000 #the page array can be treated as user-session, so one user can view multiple pages, can think of it as user-page obs

#assign treatment randomly
treatment=np.random.binomial(1,0.5,users)
treatment_effect=0.25
print(treatment)

user_page = np.random.choice(users, pageview)
# print(user_page)

#inidivual level click rate
user_click_rate=user_mean+np.random.normal(0, user_sd, users)
# print(user_click_rate)

# user_page = np.random.choice(users, pageview)
# print(user_click_rate[user_page])
# print(treatment[user_page])

def experiment(user_click_rate, pageview, treatment, treatment_effect):
    #assign each pageview with a user id (0-499), result array is of length pageview
    user_page = np.random.choice(users, pageview)

    #click rate per page (because each page is assigned to a user)
    #here, user_page shows for each pageview, which user is assigned to it
    #treatment[user_page] shows for each pageview, whether the corresponding user is in the treatment group

    page_click_rate=user_click_rate[user_page] + treatment_effect*treatment[user_page] + np.random.normal(0,.01, pageview)
    
    # Remove <0 or >1
    page_click_rate[page_click_rate>1] = 1
    page_click_rate[page_click_rate<0] = 0

    #based on the click rate prob of each pageview, simulate whether the click is done for each pageview
    clicks_observed = np.random.binomial(1, page_click_rate)

    #return a dataframe with the no. of obs. n_pageview
    return pd.DataFrame({'clicks': clicks_observed, 
                         'user': user_page, 
                         'treatment': treatment[user_page], 
                         'views' : np.ones_like(user_page)})

# Simulate for "true" standard error
#comparing the click-through rates of two treatment groups
diffs = []
for k in range(100):
    o = experiment(user_click_rate, pageview, treatment, treatment_effect)
    gbd=o.groupby(['user','treatment'], as_index=False).agg({'clicks':'sum', 'views': 'sum'})
    A, B = gbd.treatment == 0, gbd.treatment ==1 #This line creates two boolean arrays A and B 
    diffs.append((gbd.clicks[B].sum()/gbd.views[B].sum()) - (gbd.clicks[A].sum()/gbd.views[A].sum()))
print('the true effect and sd of the treatment is', np.array(diffs).mean(), np.array(diffs).std()) # std error

#now prepare the data for the treatment effect estimation
data = experiment(user_click_rate, pageview, treatment, treatment_effect)

#-------------------------------#
# method 1: OLS                 #
#-------------------------------#
df = pd.DataFrame({'Y' : data.clicks/data.views, 'D' : data.treatment, 'g' : data.user})
#ols with clustered standard error
'''The cov_type='cluster' parameter specifies that cluster-robust standard errors should be used. 
The cov_kwds parameter specifies additional keyword arguments for computing the covariance matrix. 
In this case, it specifies that the groups for clustering should be defined based on the values 
in the column 'g' of the DataFrame df, and that no correction should be applied ('use_correction': False).'''

ols_coef=ols('Y~D',df).fit(cov_type='cluster', 
                           cov_kwds={'groups': df['g'], 
                                    'use_correction':False})
ols_coef.params['D'],ols_coef.bse['D'] # see standard  error for D

#-------------------------------#
# method 2: Delta               #
#-------------------------------#
'''
Mathematically, the delta method states that if sqrt(n)*(Xn - μ)
converges in distribution to a normal distribution with mean 0 and variance 
 and if g is a differentiable function, then
 sqrt(n)*(g(Xn) - g(μ)) converges in distribution to a normal distribution with mean 0 and variance σ^2 * (g'(μ))^2.
'''
def var_ratio_delta(clicks, views):

    #ref: https://medium.com/@ahmadnuraziz3/applying-delta-method-for-a-b-tests-analysis-8b1d13411c22
    #ref: https://srome.github.io/Connections-Between-the-Delta-Method-OLS-and-CUPED-Illustrated/

    K = len(clicks)
    X = clicks
    Y = views
    
    # sample mean
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    
    # sample variance (degree of freedom = 1)
    var_x = np.var(X,ddof=1)
    var_y = np.var(Y,ddof=1)
    
    cov_xy = np.cov(X,Y, ddof=1)[0,1] # cov(X-bar, Y-bar) = 1/n * cov(X,Y) #The result of np.cov(X, Y, ddof=1) is a covariance matrix where the element at row 0, column 1 represents the covariance between X and Y.
    
    # based on deng et. al
    result = (var_x/mean_x**2 + var_y/mean_y**2 - 2*cov_xy/(mean_x*mean_y))*(mean_x*mean_x)/(mean_y*mean_y*K)

    return result

# NOTICE - delta method only is a good approximation when the cluster size K is approximately constant and number of users is larger
gbd=data.groupby(['user','treatment'], as_index=False).agg({'clicks':'sum', 'views': 'sum'})
A, B = gbd.treatment == 0, gbd.treatment ==1
gbd.tail()
print(B)

#use the delta_ratio function to get the var (z=x/y) for each group
var_control=var_ratio_delta(gbd.clicks[A], gbd.views.values[A])
var_treatment=var_ratio_delta(gbd.clicks[B], gbd.views.values[B])
print(var_treatment)
print(var_control)

print(np.var(gbd.clicks[B],ddof=1))
var_x = np.var(gbd.clicks[B],ddof=1)
var_y = np.var(gbd.views[B],ddof=1)

#ttest calculation 
def ttest(mean_control,mean_treatment,var_control,var_treatment):
    diff = mean_treatment - mean_control
    var = var_control+var_treatment    
    z = diff/np.sqrt(var)
    p_val = stats.norm.sf(abs(z))*2

    result = {'mean_control':mean_control,
             'mean_treatment':mean_treatment,
             'diff':diff,
             'se':np.sqrt(var),
             'p-value':p_val}
    return pd.DataFrame(result,index=[0])

ttest(gbd.clicks[A].sum()/gbd.views[A].sum(), gbd.clicks[B].sum()/gbd.views[B].sum(), var_control, var_treatment)

#print again the ols results for comparison
ols_coef.params['D'],ols_coef.bse['D'] # see standard  error for D

# #----------------------------------------------------------------------------------------------------#
# #B. OLS with Covariate is equivalent to the CUPED                                                   #
# #notice in this example, we dont care about the click through rate                                  #
# #we care about the number of clicks at the user level                                               #
# #this is to mimic the case when metrics is measured at the user level                               #
# #----------------------------------------------------------------------------------------------------#

# # Generate pre-experiment data and experiment data, using same user click means and treatments, etc.
# before_data = experiment(user_click_rate, pre_pageview, treatment, 0)
# after_data = experiment(user_click_rate, pageview, treatment, treatment_effect)

# # To apply the delta method or cuped, the calculations are at the user level, not page level
# pre_agg = before_data.groupby('user', as_index=False).agg({'clicks': 'sum', 'views' : 'sum'})
# post_agg = after_data.groupby('user', as_index=False).agg({'treatment': 'max', 'clicks': 'sum', 'views' : 'sum'})
# post_agg.head()

# #rename pre-data for the merge
# for col in pre_agg.columns:
#     pre_agg.rename(columns={col: 'pre_' + col}, inplace=True)

# dataB=post_agg.join(pre_agg, on='user', how='left')
# dataB['pre_clicks'] = dataB['pre_clicks'].fillna(dataB.pre_clicks.mean()) # users not seen in pre-period
# dataB['pre_views'] = dataB['pre_views'].fillna(dataB.pre_views.mean()) # users not seen in pre-period

# dataB.head()


# #-------------------------------#
# #CUPED
# #-------------------------------#

# #here, the covariate is actually the pre-outcome (in the same spirit of the DID)
# cv = np.cov(dataB['clicks'], dataB['pre_clicks']) # clicks per user!
# theta = cv[0,1]/cv[1,1]

# #CUPED adjustment of the outcome (here, not the click rate, but the click number)
# # this is the individual level adjustment of the post-outcome
# y_hat = dataB.clicks - theta * (dataB.pre_clicks - dataB.pre_clicks.mean())

# dataB['y_hat'] = y_hat
# means = dataB.groupby('treatment').agg({'y_hat':'mean'})
# variances = dataB.groupby('treatment').agg({'y_hat':'var', 'clicks':'count'}) 

# #standard 2-sample t-test at the user level
# effect=means.loc[1] - means.loc[0]
# standard_error = np.sqrt(variances.loc[0, 'y_hat'] / variances.loc[0, 'clicks'] + variances.loc[1, 'y_hat'] / variances.loc[1, 'clicks'])
# t_score = (means.loc[1] - means.loc[0]) / standard_error
# print('CUPED effect and standard error', effect, standard_error)

# #-------------------------------#
# # Direct OLS 
# #w/ pre-term stuff, 
# #note a constant is added automatically with the formula API
# #-------------------------------#
# s = ols('Y ~ D + X', 
#         pd.DataFrame({'Y': dataB.clicks, 'D': dataB.treatment, 'X': dataB.pre_clicks})).fit()
# s.params['D'],s.bse['D'] # see standard  error for D


#----------------------------------------------------------------------------------------------------------------#
#C. with A and B (ignore B), now we move to the standard case of using OLS (in the same spirit of CUPED)         #
#  with the covariate, and the performance metrics is not the same as treatment status                           #
#----------------------------------------------------------------------------------------------------------------#
dataC=dataB
dataC.head()

# OLS session level with pre-treatment variables
#notice that the y in the regression is still the click number, not the click rate
sC=ols('Y ~ X + D', {'Y': dataC.clicks/dataC.views,  
                     'X' : (dataC.pre_clicks/dataC.pre_views - (dataC.pre_clicks/dataC.pre_views).mean()),
                     'D': dataC.treatment}).fit(cov_type='cluster', cov_kwds={'groups': dataC['user'], 
                     'use_correction':False})
sC.params['D'],sC.bse['D'] # see standard  error for D

#if ignore the pre-data, and only use the treatment status for the post-data just as in A
s1=ols('Y ~ D', {'Y': dataC.clicks/dataC.views,  
                     'D': dataC.treatment}).fit(cov_type='cluster', 
                                                cov_kwds={'groups': dataC['user'], 
                                                'use_correction':False})
s1.params['D'],s1.bse['D'] # see standard  error for D

#similar results because the treatment is assigned randomly, so the pre-treatment variables are not correlated with the treatment status
