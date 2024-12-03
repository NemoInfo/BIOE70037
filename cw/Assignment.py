#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy import __version__ as scipy_version

print(f"> {"Numpy":>10}: {np.__version__}")
print(f"> {"Matplotlib":>10}: {mpl.__version__}")
print(f"> {"Pandas":>10}: {pd.__version__}")
print(f"> {"Scipy":>10}: {scipy_version}")
print(f"> {"Seaborn":>10}: {sns.__version__}")


# In[2]:


df_raw = pd.read_excel("Dataset 4845.xlsx")
df_raw.head()


# **Data cleaning** <br>
# Here we remove extra rows and empty columns

# In[3]:


print(df_raw.iloc[0,1])
groups = {1: r"$Ca$ supplements", 2:"Placebo"}
df = df_raw.iloc[:,1:4]
df.drop([0,1, 12], inplace=True)
df.columns = ["Group", "BP before", "BP after"]
df = df.astype("int32")
print(f"Group 1 -> Ca suppliment ({len(df[df["Group"]==1])} sampels)")
print(f"Group 2 -> Placebo ({len(df[df["Group"]==2])} sampels)")

df.head()


# ## Data vizualization

# In[4]:


COLOR1 = plt.get_cmap("summer")(np.linspace(.8,.2,4))
COLOR2 = plt.get_cmap("winter")(np.linspace(.8,.2,4))
COLORS = [COLOR1, COLOR2]


# In[5]:


df["diff"] = df["BP before"] - df["BP after"]

fig, axs = plt.subplots(2, 2, layout="constrained")

for r, group in enumerate([1,2]):
    axs[r,0].hist(df[df["Group"] == group].iloc[:,1:3], stacked=True, edgecolor="black", color=COLOR1[:2])
    axs[r,0].title.set_text(groups[group])
    axs[r,0].legend(["BP Before", "BP After"])

    df[df["Group"] == group]["diff"].hist(ax=axs[r,1], grid=False, edgecolor="black", color=COLOR2[r*2])
    axs[r,1].title.set_text(f"{groups[group]} difference")

    
fig.supylabel("Count")
fig.supxlabel("Blood pressure (BP)")
plt.show()


# In[6]:


sns.boxplot(x="Group", y="diff", data=df, hue="Group", dodge=False, legend=False,
            palette={1:COLOR2[0], 2:COLOR2[2]})
plt.xticks(ticks=[0, 1], labels=groups.values())
plt.ylabel("Difference")
plt.show()


# In[7]:


alpha = 0.05

for group in [1,2]:
    for col in df.columns[1:]:
        pval = stats.shapiro(df[df["Group"] == group][col]).pvalue
        print(
            f"Group {group}, {col:<9}: IS {"NOT " if pval < alpha else ""}normally distributed" + \
            f" with p-value {pval:.2f}")


# => The datasets and they're differences are both normaly distributed, we can proceed to do a t-test 

# In[8]:


stats.ttest_ind(*[df[df["Group"] == g]["diff"] for g in [1,2]])


# Data is not sufficnetly different (pvalue of 0.12), fail to reject $H_0$ that the distributions are equal

# ### Bootstrapping

# In[9]:


bdata1 = np.zeros(500)
bdata2 = np.zeros(500)
for i in range(500):
    s1b = np.random.choice(df[df["Group"] == 1]["BP before"], size=20, replace=True)
    s1a = np.random.choice(df[df["Group"] == 1]["BP after"], size=20, replace=True)
    bdata1[i] = s1b.mean() - s1a.mean()

    s2b = np.random.choice(df[df["Group"] == 2]["BP before"], size=20, replace=True)
    s2a = np.random.choice(df[df["Group"] == 2]["BP after"], size=20, replace=True)
    bdata2[i] = s2b.mean() - s2a.mean()


# In[10]:


bins = np.linspace(-6, 12, 15)
plt.hist(bdata1, bins=bins, edgecolor="black", color=COLOR1[3])
plt.hist(bdata2, alpha=0.7, bins=bins, edgecolor="black", color=COLOR1[0])

plt.show()


# In[11]:


## What alpha value should i use considering i bootstraped the data
stats.ttest_ind(bdata1, bdata2)

