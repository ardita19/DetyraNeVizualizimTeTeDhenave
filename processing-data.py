import pandas as pd
df1 = pd.read_csv("ProductInfo.csv", index_col=None)

#atribute subset selection
data1 = df1.drop(columns=["id","ReviewURL","ProductFullName"])
#data1.to_csv("ProductInfo2.csv")
data2 = pd.read_csv("AllProductreviews.csv", index_col=None)
#print(data1)
#print(data2)


#integration
datafinal = pd.merge(data1, data2, on='ProductShortName')
#print(datafinal)
#datafinal.to_csv("MergeFile.csv")

mergeddata = pd.read_csv('MergeFile.csv', index_col=0)

#sampling
rows = mergeddata.sample(n=1000)
print(rows['ReviewBody'])

#Aggregation
countOfReviewsPerProduct = mergeddata.groupby('ProductShortName').size()
#print(countOfReviewsPerProduct)
countReviewsPerStars =  mergeddata.groupby('ReviewStar').size()
#print(countReviewsPerStars)


#missing values
nan_value = float("NaN")
mergeddata.replace("", nan_value, inplace=True)
mergeddata.replace(" ", nan_value, inplace=True)
mergeddata.dropna(inplace=True)
#print(mergeddata)

#mergeddata.to_csv("MergeFile3.csv")

#featureCreation using transformation
data = pd.read_csv('Pre-processed8.csv', index_col=0)
data['difference'] = data.apply(lambda x: x['MRP'] - x['Price'], axis=1)
#binarization
data['sentiment'] = data.apply(lambda x: x['ReviewStar'] >=3 , axis=1)
asInteger = data['sentiment'].astype(int)
data['sentiment'] = asInteger

#data.to_csv("PreProcessedFinal1.csv")
#print(data)


