from sklearn.externals import joblib
from preprocessing import Preprocessing
import numpy

model = joblib.load("model.pkl")
dp = Preprocessing()

def label_to_str(value):
    value = dp.stem(value)
    value = dp.remove_stopwords(value)
    h = model.predict([value])
    if not h[0]:
        print('->Negatif review')
    else:
        print('->Positif review')


current = True
while current:
    sentence = input("Masukkan review: ")
    if sentence == "exit":
        current = False
    else:
        label_to_str(str(sentence))

print("makasi")