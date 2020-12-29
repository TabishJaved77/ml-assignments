# Naïve Bayes Analysis:
# Dataset a1_d3.txt

#01. Importing the Libraries: 
import numpy as np
import pandas as pd
import re

#02. Importing the dataset
dataset = pd.read_csv("a1_d3.txt", delimiter = '\t', quoting = 3, header = None)

#03. Text cleaning
corpus = [] #Cleaned reviews in a List
for i in range(0,1000):
 review = re.sub('[^a-zA-Z]', ' ', dataset[0][i]) #Pre-processing
 review = review.lower()
 review = review.split()        #Splits string to list
 review = [word for word in review]
 review = ' '.join(review)      #  String+' '+List
 corpus.append(review)

#04. Build our vocab
word_list = []                      #Listing all words from corpus(vocab)
for review in corpus:
    words_in_review = review.split()
    word_list.extend(words_in_review)   #Add list of words to vocab
word_list = sorted(list(set(word_list)))

#05. Build Bag of Words model
bag = np.zeros((len(corpus),len(word_list)))    #Sparse matrix
review_index = 0
for review in corpus:
    for review_word in review.split():
        for i,vocab_word in enumerate(word_list):
            if vocab_word == review_word:
                bag[review_index][i] +=1
    review_index+=1
bag = pd.DataFrame(bag)

#06. Naive Bayes model:
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.priors =  np.zeros(n_classes, dtype=np.float64) #Prior Calculation
        for c in self.classes:
            self.priors[c] = np.count_nonzero(y==c)/n_samples
        
        self.likelihood = np.zeros((n_features, n_classes),dtype=np.float64)        #Likelihood Calculation
        for c in self.classes:
            denominator = np.count_nonzero(y==c)
            for f in range(0,n_features):
                numerator = np.sum(X[i][f] for i in range(0,n_samples) if y[i]==c)                  
                self.likelihood[f][c] = (numerator+ 1)/(denominator + n_features)
            
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []  #Posterior prob. for all classes
        for c in self.classes:
            prior = np.log(self.priors[c])
            posterior = 0.0
            for idx,n in enumerate(x):
                posterior += n*np.log(self.likelihood[idx][c])
            posterior = prior + posterior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)] #Maximum posterior prob.

#07. Data for the model
X = bag.iloc[:, :].values
Y = dataset.iloc[:, -1].values

#08. 5 Fold splitting:
accuracy_scores = []
f1_scores = []

from sklearn.model_selection import KFold
kf = KFold(n_splits = 5)        # 5 Fold cross validation
for train_index, test_index in kf.split(X):
    X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
    classifier = NaiveBayes()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

#09. Accuracy,  Precision, Recall & F-score Calculations for each split:
    cm = pd.crosstab(Y_test, Y_pred)
    cm = cm.to_numpy()
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    recall = cm[0][0] / (cm[0][0]+cm[1][0])
    precision = cm[0][0] / (cm[0][0]+cm[0][1])
    fscore = 2 * recall * precision / (recall + precision)
    accuracy_scores.append(accuracy)
    f1_scores.append(fscore)

#10. Mean & Standard Dev. of accuracy and Fscore calculations:   
accuracy_mean = sum(accuracy_scores) / len(accuracy_scores) 
variance = sum([((x - accuracy_mean) ** 2) for x in accuracy_scores]) / len(accuracy_scores) 
accuracy_std = variance ** 0.5
fscore_mean = sum(f1_scores) / len(f1_scores) 
variance = sum([((x - fscore_mean) ** 2) for x in f1_scores]) / len(f1_scores) 
fscore_std = variance ** 0.5

#11. Print final results:
print("Accuracy: %0.3f +/- %0.3f" % (accuracy_mean, accuracy_std))
print("F-score: %0.3f +/- %0.3f" % (fscore_mean, fscore_std))           