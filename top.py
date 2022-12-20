from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import glob
import os
import nltk
from sklearn.feature_extraction import text 
flist = glob.glob(os.path.join(os.getcwd(), "/home/useruser/Documents/krishnafiles/my ", "*.txt"))

data = []

for fpath in flist:

    with open(fpath) as f_input:
        data.append(f_input.read())

documents = data
stop_words = [unicode(x.strip(), 'utf-8') for x in open('/home/useruser/Documents/krishnafiles/stop.txt','r').read().split('\n')]

vectorizer = TfidfVectorizer(analyzer='word', stop_words = text.ENGLISH_STOP_WORDS.union(stop_words))
#vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
true_k = 4
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :10]:
        print ' %s' % terms[ind],
    print
