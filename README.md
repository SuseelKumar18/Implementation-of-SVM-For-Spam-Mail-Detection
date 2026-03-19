# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required Python libraries such as pandas and sklearn.
2. Load the dataset spam.csv into the program.
3. Preprocess the dataset by selecting the message and label columns.
4. Convert labels ham → 0 and spam → 1.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: S Suseel Kumar
RegisterNumber: 212225240163
*/

from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('spam.csv', encoding='latin-1')

data.head()

data = data[['v1','v2']]
data.columns = ['label','message']

data.head()

data['label'] = data['label'].map({'ham':0, 'spam':1})

X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = SVC(kernel='linear')

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

sample = ["Congratulations! You have won a free lottery ticket"]

sample_vec = vectorizer.transform(sample)

result = model.predict(sample_vec)

if result[0] == 1:
    print("Spam Mail")
else:
    print("Not Spam")

```

## Output:
<img width="932" height="222" alt="image" src="https://github.com/user-attachments/assets/a0b5218d-1237-41b1-ac3b-68acf984ecae" />
<img width="343" height="102" alt="image" src="https://github.com/user-attachments/assets/fd313bc4-b053-4f29-a871-f0b5b91ec51a" />
<img width="720" height="271" alt="image" src="https://github.com/user-attachments/assets/7f715eae-1bf0-4927-b8c8-e118e2a6e2c4" />
<img width="139" height="29" alt="image" src="https://github.com/user-attachments/assets/32a41883-85b8-4f24-800c-6ff49c51263b" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
