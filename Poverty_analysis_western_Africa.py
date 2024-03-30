#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sn
import os
import sklearn as sk
import statsmodels.api as sm

class country:
    def __init__(self,name,code_country,year):
        self.country_name = name
        self.year = year
        self.code_country = code_country
        url_base = 'https://github.com/mermozhk/Poverty/raw/main/'
        #print('/'.join([url_base,name,year,'ehcvm_welfare_'+''.join([code_country,year])])+'.dta')
        self.welfare_data = pd.read_stata('/'.join([url_base,name,year,'ehcvm_welfare_'+''.join([code_country,year])])+'.dta')
        self.welfare_data['dtot_corrected']=self.welfare_data['dtot']/self.welfare_data['def_spa']
        
        self.conso_data = pd.read_stata('/'.join([url_base,name,year,'ehcvm_conso_'+''.join([code_country,year])])+'.dta')
        self.menage_data = pd.read_stata('/'.join([url_base,name,year,'ehcvm_menage_'+''.join([code_country,year])])+'.dta')
        self.individu_data = pd.read_stata('/'.join([url_base,name,year,'ehcvm_individu_'+''.join([code_country,year])])+'.dta')
        
        self.merge_data = pd.merge(self.individu_data,
               self.welfare_data[['hhid','zref','dtot_corrected','hhsize']],
               on = 'hhid',
               how='outer')
        self.merge_data['constant'] =1
    
    
    def estimation_procedure(self,index):
        wls_model = sm.WLS(self.merge_data[index],self.merge_data['constant'], weights=self.merge_data['hhweight'])
        return wls_model.fit()
        
    def prevalence(self):
        self.merge_data['prevalence']=(self.merge_data['dtot_corrected']<
                                       (self.merge_data['zref']*self.merge_data['hhsize'])).map({False:0, True:1})
        return self.estimation_procedure('prevalence')
    
    def gap(self):
        self.merge_data['gap'] = (1-self.merge_data['dtot_corrected']/(self.merge_data['zref']*self.merge_data['hhsize']))*self.merge_data['prevalence']
        return self.estimation_procedure('gap')
    
    def severity(self):
        if not('prevalence' in self.merge_data.columns):
            
            self.merge_data['prevalence']=(self.merge_data['dtot_corrected']<
                                       (self.merge_data['zref']*self.merge_data['hhsize'])).map({False:0, True:1})
            
        self.merge_data['severity'] = ((1-self.merge_data['dtot_corrected']/(self.merge_data['zref']*self.merge_data['hhsize']))**2)*self.merge_data['prevalence']
        
        return self.estimation_procedure('severity')
    
    def aart_welfare_index(self):
        self.merge_data['aart_welfare'] = (self.merge_data['zref']*self.merge_data['hhsize'])/self.merge_data['dtot_corrected']
        return self.estimation_procedure('aart_welfare')
    
    def aart_poverty_index(self):
        if not('aart_welfare' in self.merge_data.columns):
            self.merge_data['aart_welfare'] = (self.merge_data['zref']*self.merge_data['hhsize'])/self.merge_data['dtot_corrected']
        
        self.merge_data['aart_poverty'] = self.merge_data.apply(lambda x: max(x.aart_welfare,1),axis=1)
        return self.estimation_procedure('aart_poverty')
        


# In[2]:


Benin = country(name='Benin',code_country='BEN',year='2018')


# In[ ]:


Benin.aart_welfare_index().summary()


# In[ ]:


Benin.aart_poverty_index().summary()


# In[ ]:


Benin.prevalence().summary()


# In[ ]:


Benin.severity().summary()

