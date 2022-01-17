import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv(
    'PreProcessedFinal1.csv')
#print(df)


#Data type based visualization

# plt.hist(df['ProductShortName'])
# plt.xticks(rotation=90)
# plt.show()

plt.hist(df['sentiment'])
plt.xticks(np.arange(0, 2, 1))
# plt.show()

plt.hist(df['Price'])
# plt.show()

plt.hist(df['difference'])
# plt.show()

#static Visualization

plt.hist(df['ReviewStar'])
#plt.show()

#interactive visualization
fig = px.histogram(df, x="Price", color="sentiment")
#fig.show()

#Multidimensional visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = df['Price']
ys = df['MRP']
zs = df['ReviewStar']
ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')


ax.set_xlabel('Price')
ax.set_ylabel('MRP')
ax.set_zlabel('ReviewStar')
#plt.show()