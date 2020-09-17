# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:58:57 2020

@author: shashi
"""


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules


groceries = []
# As the file is in transaction data we will be reading data directly 
with open("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment of association rules\\groceries.csv","r") as f:
    groceries = f.read()



# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    
    
all_groceries_list = []

#for i in groceries_list:
#    all_groceries_list = all_groceries_list+i
    
    
all_groceries_list = [i for item in groceries_list for i in item]
from collections import Counter

item_frequencies = Counter(all_groceries_list)
# after sorting
#item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])


# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 

import matplotlib.pyplot as plt

plt.bar(height = frequencies[:11],x = list(range(0,11)),color='rgbkymc');plt.xticks(list(range(0,11),),items[:11]);plt.xlabel("items")
plt.ylabel("Count")


# Creating Data Frame for the transactions data 
import pandas as pd
# Purpose of converting all list into Series object Coz to treat each list element as entire element not to separate 
groceries_series  = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835,:] # removing the last empty transaction

groceries_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

frequent_itemsets = apriori(X,min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.shape


# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.shape
#at support 0.005 is gives 2700 rules

rules.head(10)
rules.sort_values('lift',ascending = False,inplace=True)

# checking at support value 0.010
frequent_itemsets2 = apriori(X,min_support=0.010, max_len=3,use_colnames = True)
frequent_itemsets.shape
frequent_itemsets2.sort_values('support',ascending = False,inplace=True)

plt.bar(x = list(range(1,11)),height = frequent_itemsets2.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets2.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
rules2.shape
# as support value 0.010 it gives 598 rules
 rules2.head(10)
 rules2.sort_values('lift', ascending = False, inplace = True)

# checking at support = 0.020

frequent_itemsets3 = apriori(X,min_support=0.010, max_len=3,use_colnames = True)
frequent_itemsets3.shape
frequent_itemsets3.sort_values('support',ascending = False,inplace=True)

plt.bar(x = list(range(1,11)),height = frequent_itemsets3.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets3.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules3 = association_rules(frequent_itemsets3, metric="lift", min_threshold=1)
rules3.shape
# as support value 0.020 it gives 598 rules
 rules2.head(10)
 rules2.sort_values('lift', ascending = False, inplace = True)
# hence at 0.005 we got 2700 rules and at 0.010 and 0.020 we got 598 rules and top 10 rules remaind same for all

