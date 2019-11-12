from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from preprocessing import Preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB


#data punya bang farhan, ambil dr kaggle "product_review_dirty.csv"
pr = Preprocessing()
tokopedia_reviews = pd.read_csv("product_reviews_dirty.csv", sep=',', header='infer')
tokopedia_reviews.columns = ['id','text','rating','category','product_name', 'product_id', 'sold', 'shop_id', 'url']
print(tokopedia_reviews.iloc[:10, :3])


#Clean data including symbol, stopwrod and stemming a word
print("Please wait while cleaning data....")
tokopedia_reviews['text'] = tokopedia_reviews['text'].apply(pr.processtext)
tokopedia_reviews['text'] = tokopedia_reviews['text'].apply(pr.stem)
tokopedia_reviews['text'] = tokopedia_reviews['text'].apply(pr.remove_stopwords)
tokopedia_reviews = tokopedia_reviews.drop_duplicates()
print(tokopedia_reviews.iloc[:10, :3])


#ambil kolom yang penting saja
tokopedia_reviews = tokopedia_reviews[['text', 'rating']]
print(tokopedia_reviews.iloc[:10, :])

#Labelling data
#Asumsi rating 5 = Positif
#dan rating 1,2,3,4 = Negatif
#Anda bisa mengutak-atik ini, hehe
print("Please wait while creating label...")
tokopedia_reviews.loc[tokopedia_reviews['rating'] > 4, 'label'] = True
tokopedia_reviews.loc[tokopedia_reviews['rating'] <= 4, 'label'] = False
tokopedia_reviews.to_csv('tokopedia_cleaned.csv', header=True, index=False, encoding='utf-8')
print(tokopedia_reviews)


#Vectorization
print("Please wait while doing vectorization...")
vector = CountVectorizer().fit(tokopedia_reviews['text'])

#Membuat classifier
print("Please wait while creating classifier...")
X_train, X_test, y_train, y_test = train_test_split(tokopedia_reviews['text'], tokopedia_reviews['label'], train_size = 0.75)

pipeline = Pipeline([('bow', CountVectorizer(strip_accents='ascii', lowercase=True)),
                     ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB()), ])
parameters = {'bow__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'classifier__alpha': (1e-2, 1e-3), }

grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
grid.fit(X_train, y_train)

joblib.dump(grid, "model.pkl")



print("Please wait 2 or 3 mins while creating summary...")
model = joblib.load("model.pkl")

def label_to_str(x):
    if not x:
        return 'Negative'
    else:
        return 'Positive'

x = 0
text_ = [0] * len(tokopedia_reviews)
label_ = [0] * len(tokopedia_reviews)

for review in tokopedia_reviews['text']:
    predict = model.predict([review])
    text_[x] = review
    label_[x] = label_to_str(predict[0])
    x += 1


data = {"text": text_, "label": label_}
hehe = pd.DataFrame(data=data)
hehe.columns = ['text', 'predicted_label']
hehe.to_csv('predicted_model.csv', header=True, index=False, encoding='utf-8')

print("\nConfusion matrix:")
print('confusion matrix: \n', confusion_matrix(y_test, model.predict(X_test)))

print("DONE!")