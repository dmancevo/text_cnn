# Convolutional Neural Networks for Text Classification

Convolutional Neural Networks for text classification built on top of Gensim's well known word2vec implementation and Keras.

Dependencies:
- Numpy
- Gensim
- Keras
- H5Py (for saving the model)

Usage:

```python
X=['hello world','Μου αρέσουν οι ντολμάδες','الجو دافئ اليوم','池塘水很冷']
y=[1,2,3,4]

cnn = Text_CNN()
cnn.fit(X,y)           # Train the model.

cnn.predict(X)         # Predict most likely class.
cnn.predict_proba(X)   # Per class probabilities.

cnn.save()             # Save the model.
cnn.load()             # Load previously saved model.
```