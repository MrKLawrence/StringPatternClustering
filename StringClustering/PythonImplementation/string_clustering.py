import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
import distance
import sys

#This resembled a code example for string clustering based on the following research on Levenshtein's distance from my previous job
#reference: https://stats.stackexchange.com/questions/123060/clustering-a-long-list-of-strings-words-into-similarity-groups

imputfilename = 'testfile.txt'

df = pd.read_csv(inputfilename, engine='python')

print("total number of jobs are ", len(df["Job Name"])

items = list(dict.fromkeys(df["Job Name"]))

out = ' '.join(items)

lengths = len(items)

print("Unique items are ",items)

print ("Number of unique items are ", lengths)

input = out.split(' ') # Replace this line

print('input = ',input)

input = np.asarray(input) #So that indexing with a list vill work

print('input array is ',input)

lev_similarity = -1*np.array([[distance.levenshtein(vl,v2) for w1 in input] for w2 in input])
affprop = AffinityPropagation(affinity="precomputed", damping=0.5)

affprop.fit(lev_similarity)
outputfilename = inputfilename.split('.')[0]+'_output.txt'
with open(outputfilename, 'a') as f:
    f.write(inputfilename)
    f.write("\n ---------- \n")
    for cluster_id in np.unique(affprop.labels_):
        exemplar = input[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(input[np.nonzero(affprop.labels_==cluster_id)])
        cluster_str = ", ".join(cluster)
        #print(" - *%s:* %s" % (exemplar, cluster_str))
        #print ("cluster " , exemplar)
        f.write(exemplar)
        f.write("\n -------------------- \n")
        print ("items are : " , cluster_str)
        f.write((cluster_str)
        f.write("\n #################### \n")
#print(exemplar)
#print("number in exemplar are ", len(exemplar))
#print(cluster_str)
#print("number in cluster_str are ", len(cluster_str))
