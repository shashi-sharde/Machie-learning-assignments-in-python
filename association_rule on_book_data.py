# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:26:57 2020

@author: shashi
"""


import pandas as pd 
from mlxtend.frequent_patterns import apriori,association_rules
book = pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment of association rules\\book.csv")
    
Freq_item = apriori(book,min_support=0.005, max_len=3,use_colnames = True)
Freq_item.shape
# most freq item on basis of support

Freq_item.sort_values('support', ascending = False, inplace = True)
import matplotlib.pyplot as plt
plt.bar(x=list(range(1,11)), height = Freq_item.support[1:11], color = 'rgmyk');
plt.xticks(list(range(1,11)),Freq_item.itemsets[1:11])
plt.xlabel('item-sets'); plt.ylabel('support')

rules = association_rules(Freq_item,metric = 'lift', min_threshold = 1 )
rules.shape
#number of rules at 0.005 support = 1054
rules.head(10)
#checking with support value 0.010
Freq_item2 = apriori(book,min_support=0.010, max_len=3,use_colnames = True)
Freq_item.shape

Freq_item2.sort_values('support', ascending = False, inplace = True)
rules2 = association_rules(Freq_item2, metric = 'lift', min_threshold = 1)
rules2.shape
#with support value 0.010 we get 962 rules
rules2.head(10) 

plt.bar(x=list(range(1,11)), height = rules2.support[1:11], color = 'rgmyk');
plt.xlabel('item-sets'); plt.ylabel('support')
# checking with support value 0.020
freq_item3 = apriori(book,min_support = 0.020, max_len = 4, use_colnames = True)
freq_item3.shape

freq_item3.sort_values('support', ascending = False, inplace = True)
rules3 =  association_rules(freq_item3, metric = 'lift', min_threshold = 1)
rules3.shape
# with support value 0.020 it gives 1808 rules 
plt.bar(x=list(range(1,11)), height = rules3.support[1:11], color = 'rgmyk');
plt.xlabel('item-sets'); plt.ylabel('support')

rules3.head(10)
 # changing support values, number of rules changes but the top 10 rules remained same in all senarios 