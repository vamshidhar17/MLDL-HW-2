#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


from mlxtend.data  import  loadlocal_mnist
import platform
import numpy as np 
import pandas as pd
import matplotlib.pyplot as  plt 
import  seaborn as  sns
import numpy as np


# In[3]:


# 2  PCA
X,y = loadlocal_mnist(images_path="train-images.idx3-ubyte",labels_path="train-labels.idx1-ubyte")


# In[4]:


np.savetxt(fname='images.csv', X=X,  delimiter=',',  fmt='%d')
np.savetxt(fname='labels.csv',X=y,  delimiter=',',  fmt='%d')


# In[5]:


df_img =  pd.read_csv('images.csv')
df_img.head()


# In[7]:


df_label = pd.read_csv('labels.csv')
df_label.rename(columns={'5':  'label'},inplace = True)
df_label.head()


# In[8]:


label = df_label["label"]


# In[9]:


ind= np.random.randint(0,20000)
plt.figure(figsize=(20,5))
grid_data =  np.array(df_img.iloc[ind]).reshape(28,28)
plt.imshow(grid_data,  interpolation = None,  cmap =  'gray') 
plt.show()
print(label[ind])


# In[10]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
std_df=  scaler.fit_transform(df_img)
std_df.shape


# In[11]:



covar_mat=  np.matmul(std_df.T,  std_df)
covar_mat.shape


# In[12]:


from scipy.linalg import eigh
values,  vectors= eigh(covar_mat,  eigvals=(782,783))
print("Dimensions of  Eigen vector:",  vectors.shape)
vectors= vectors.T
print("Dimensions of Eigen vector:",  vectors.shape)


# In[14]:


final_df= np.matmul(vectors,  std_df.T)
print("vectros:",  vectors.shape,  "n",  "std_df:",  std_df.T.shape,  "n",  "final_df:",final_df.T.shape)


# In[15]:


final_dfT=  np.vstack((final_df,  label)).T
dataFrame=  pd.DataFrame(final_dfT,  columns=['pca_1',   'pca_2',   'label'])
dataFrame


# In[40]:


sns.FacetGrid(dataFrame,  hue=  'label').map(sns.scatterplot,   'pca_1','pca_2')
plt.show()


# In[17]:


#PROBLEM 2 PCA(A)


# In[19]:


train_df=  pd.read_csv("train.csv")
train_df.head()


# In[20]:


mean =  train_df.groupby(['label']).mean()
print((mean))


# In[21]:


std =  train_df.groupby(['label']).std()
print((std))


# In[22]:


x=  train_df.drop(['label'],  axis=1)# displaying the  images with mean of each pi
y= train_df['label']
from sklearn.decomposition  import  PCA
pca	=  PCA(n_components=2)# 10 dimensions
principalCom=  pca.fit_transform(x)
principalDf=  pd.DataFrame(data=  principalCom, columns=['pca_1',   'pca_2'])
df2=  pd.concat([principalDf,  train_df[['label']]],  axis=1)


# In[23]:


plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10,6))
c_map= plt.cm.get_cmap('jet',10)
plt.scatter(principalCom[:,0],  principalCom[:,1],  s=15,cmap= c_map, c= df2['label'])
plt.colorbar()
plt.xlabel('PC-1') , plt.ylabel('PC-2') 
plt.show()


# In[24]:


import numpy as np
import matplotlib.pyplot as  plt


# In[25]:


from  sklearn.metrics  import  r2_score
from  sklearn.metrics  import  mean_squared_error 
from  math  import  sqrt
import numpy as np


# In[26]:


X=  dataFrame.head(59999)
f=  final_dfT
r2=  r2_score(X,f)
rmse=  sqrt(mean_squared_error(X,f))


# In[27]:


pca	=  PCA()
pca.fit(x)


# In[28]:


pca.n_components_


# In[29]:


tot=  sum(pca.explained_variance_)
tot


# In[30]:


var_exp = [(i/tot)*100 for i in  sorted(pca.explained_variance_,  reverse=True)]
print(var_exp[0:5])


# In[31]:


tot=  sum(pca.explained_variance_)
tot


# In[32]:


var_exp = [(i/tot)*100 for i in sorted(pca.explained_variance_,  reverse=True)]
print(var_exp[0:5])


# In[33]:


cum_var_exp	=  np.cumsum(var_exp)


# In[34]:


plt.figure(figsize=(10, 5))
plt.step(range(1, 785),  cum_var_exp,  where='mid',label='cumulative explained var')
plt.title('Cumulative Explained Variance as  a  Function of the Number of Component')
plt.ylabel('Cumulative Explained variance')
plt.xlabel('Principal components')
plt.axhline(y=95,  color='k',  linestyle='--',  label=  '95% Explained Variance')
plt.axhline(y=90,  color='c',  linestyle='--',  label=  '90% Explained Variance')
plt.axhline(y=85,  color='r',  linestyle='--',  label=  '85% Explained Variance')
plt.legend(loc='best') 
plt.show()


# In[ ]:




