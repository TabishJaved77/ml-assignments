# Fisher's Linear Discriminant Analysis:
# Dataset 1

#01. Importing Libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import csv  
import seaborn as sns
from scipy.stats import norm
import scipy.stats as stats
import math

#02. Normal Distribution:
def normdist(mu, sigma, col):
	x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
	plt.plot(x, stats.norm.pdf(x, mu, sigma), color = col)

def solve(m1,m2,s1,s2):
	a = 1/(2*s1**2) - 1/(2*s2**2)
	b = m2/(s2**2) - m1/(s1**2)
	c = m1**2 /(2*s1**2) - m2**2 / (2*s2**2) - np.log(s2/s1)
	return np.roots([a,b,c])

#03. Program to read the .csv datasets and execute:
def load_dataset(data_file):
    with open(data_file) as csv_file:
        dataset = list(csv.reader(csv_file, delimiter=','))
    return dataset
dataset = load_dataset("a1_d1.csv")
class0=[]
class1=[]
for point in dataset:
	if point[-1] == '0':
		point = point[:-1]
		class0.append(point)
	else:
		point = point[:-1]
		class1.append(point)	

c0 = np.array(class0).astype(np.float)
c1 = np.array(class1).astype(np.float)
m0 = np.mean(c0, axis=0)
m1 = np.mean(c1, axis=0)
s0 = (len(class0)-1)*np.cov(c0, rowvar=0).astype(np.float)
s1 = (len(class1)-1)*np.cov(c1, rowvar=0).astype(np.float)
S = s0 + s1

S_inv = np.linalg.inv(S).astype(np.float)
v=np.matmul(S_inv, m0-m1).astype(np.float)
y0 = np.matmul(np.transpose(v), np.transpose(c0)).astype(np.float)
y1 = np.matmul(np.transpose(v), np.transpose(c1)).astype(np.float)

#04. Classification threshold line:
def plotline(poi):	
	y = np.linspace(-5,5)
	plt.plot([poi]*len(y), y, color='black')	

#05. Plot1: Scattered points plot:  
plt.scatter(c0[:,0], c0[:,1], color = "green", label = 'Class 0')
plt.scatter(c1[:,0], c1[:,1], color = "red", label = 'Class 1')
x = np.linspace(-0.05,0.05)
y = v[1] / v[0] * x
plt.plot(x, y, color = 'black')
plt.legend()
plt.show()

#06. PLot2: Visualizing the points on a line (1D) & Normal distribution plot:
normdist(np.mean(y0),np.std(y0), 'green')
normdist(np.mean(y1),np.std(y1), 'red')
result = solve(np.mean(y0),np.mean(y1),np.std(y0),np.std(y1))
plotline(result[1])
plt.scatter(y0,len(y0)*[0], color='green', label='Class0') 
plt.scatter(y1,len(y1)*[0], color='red', label='Class1')
plt.legend()
plt.show()

#07. Accuracy Calculations:
def accuracy(poi, projection0, projection1):
	count = 0
	for point in projection0:
		if point > poi:
			count += 1					
	for point in projection1:
		if point < poi:
			count += 1			
	print("Accuracy is : " , (count / len(class0 + class1) * 100),"%")		

#08. Precision, Recall & F-score Calculations:
def f_measure(poi, projection0, projection1):
    tp = 0
    fp = 0
    totalp = 0
    for i in projection1:
        if i<poi:
           tp += 1
    for i in projection0:
    	if i<poi:
    		fp += 1

    totalp = len(projection1)
    precision = tp / (fp + tp)
    recall = tp / totalp
    print("F-score is ", (2 * precision * recall) / (precision + recall))


#09. Print final results:
accuracy(result[1], y0,y1)
f_measure(result[1], y0, y1)