import pandas as pd
from preprocessing import Preprocessing


#Load hasil crawling
pr = Preprocessing()
tokopedia_reviews = pd.read_csv("product_reviews_dirty.csv", sep=',', header='infer')
tokopedia_reviews.columns = ['id','text','rating','category','product_name', 'product_id', 'sold', 'shop_id', 'url']
print(tokopedia_reviews.iloc[:10, :3])


#Clean data if necessary
tokopedia_reviews['text'] = tokopedia_reviews['text'].apply(pr.processtext)
print(tokopedia_reviews.iloc[:10, :3])


#Pisahkan rating tinggi dengan rating rendah
positive_review = tokopedia_reviews.loc[(tokopedia_reviews['rating'] > 3)]
negative_review = tokopedia_reviews.loc[(tokopedia_reviews['rating'] <= 3)]
print("positive review:")
print(positive_review.iloc[:10, :3])
print("negative review:")
print(negative_review.iloc[:10, :3])


#save pisahan tadi dg csv baru (if necessary)
positive_review.to_csv('positive_review.csv', header=True, index=False, encoding='utf-8')
negative_review.to_csv('negative_review.csv', header=True, index=False, encoding='utf-8')


