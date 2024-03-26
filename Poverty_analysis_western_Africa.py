#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import seaborn as sn
import os
import sklearn as sk
import statsmodels.api as sm

class country:
    def __init__(self,name,year,welfare,conso,menage,individu):
        self.country_name = name
        self.year = year
        self.welfare_data = pd.read_stata(welfare)
        self.conso_data = pd.read_stata(conso)
        self.menage_data = pd.read_stata(menage)
        self.individu_data = pd.read_stata(individu)
        
        self.merge_data = pd.merge(self.individu_data,
               self.welfare_data[['hhid','zref','dtot','hhsize']],
               on = 'hhid',
               how='outer')
        self.merge_data['constant'] =1
    
    def estimation_procedure(self,index):
        wls_model = sm.WLS(self.merge_data[index],self.merge_data['constant'], weights=self.merge_data['hhweight'])
        return wls_model.fit()
        
    def prevalence(self):
        self.merge_data['prevalence']=(self.merge_data['dtot']<
                                       (self.merge_data['zref']*self.merge_data['hhsize'])).map({False:0, True:1})
        return self.estimation_procedure('prevalence')
    
    def gap(self):
        self.merge_data['gap'] = (1-self.merge_data['dtot']/(self.merge_data['zref']*self.merge_data['hhsize']))*self.merge_data['prevalence']
        return self.estimation_procedure('gap')
    
    def severity(self):
        self.merge_data['severity'] = ((1-self.merge_data['dtot']/(self.merge_data['zref']*self.merge_data['hhsize']))**2)*self.merge_data['prevalence']
        return self.estimation_procedure('severity')
    
    def aart_welfare_index(self):
        self.merge_data['aart_welfare'] = (self.merge_data['zref']*self.merge_data['hhsize'])/self.merge_data['dtot']
        return self.estimation_procedure('aart_welfare')
    
    def aart_poverty_index(self):
        if not('aart_welfare' in self.merge_data.columns):
            self.merge_data['aart_welfare'] = (self.merge_data['zref']*self.merge_data['hhsize'])/self.merge_data['dtot']
        
        self.merge_data['aart_poverty'] = self.merge_data.apply(lambda x: max(x.aart_welfare,1),axis=1)
        return self.estimation_procedure('aart_poverty')
        


# In[19]:


Benin = country(name='Benin',year='2018',welfare="Benin_2018/ehcvm_welfare_BEN2018.dta",
                conso="Benin_2018/ehcvm_conso_BEN2018.dta",
               menage="Benin_2018/ehcvm_menage_BEN2018.dta",
               individu="Benin_2018/ehcvm_individu_BEN2018.dta")


# In[20]:


Benin.aart_welfare_index().summary()


# In[21]:


Benin.aart_poverty_index().summary()


# In[13]:


Benin.severity().summary()


# In[ ]:




