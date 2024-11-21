import random
import numpy as np
import matplotlib.pyplot as plt

def assign_cluster(X,centroid):
  d = 0 #Distortion
  clusters = {i: [] for i in range(len(centroid))}
  for i in range(len(X)):
    euc_dist=[]
    for j in range(len(centroid)):
      euc_dist.append(np.linalg.norm(X[i]-centroid[j]))
    min_ind = np.argmin(euc_dist)
    d += min(euc_dist)
    clusters[min_ind].append(X[i])
  '''for i in range(len(centroid)):
    clusters[i]=np.array(clusters[i])'''
  return clusters , d/len(X)

def calc_centroid(centroid,clusters,n_clusters):
  for i in range(n_clusters):
    centroid[i] = np.mean(clusters[i], axis=0)
  #print(centroid)
  return centroid

def plot_(centroid,clusters,n_clusters):
    for i in range(n_clusters):
      plt.scatterplot(clusters[i][:][0],clusters[i][:][1])
      plt.legend(str(i))

def k_means(X,n_clusters,max_iter=20,thresh=0.01):
  print('---------------No of cluster :',n_clusters)
  k = n_clusters

  # Initialize Centroids Randomly 
  centroid_ind = random.sample(range(len(X)),k)
  centroid = []  
  for i in range(k):
    centroid.append(X[centroid_ind[i]]) 
  centroid = np.array(centroid)
  #print('Initial Centroid :',centroid)

  # Assigning Clusters 
  clusters, distortion = assign_cluster(X,centroid)

  #Recalculating Centroids
  prev_centroid = centroid.copy()
  centroid = calc_centroid(centroid,clusters,n_clusters)
  #print('prev : ',prev_centroid,'new : ',centroid)
  #print('diff  :  ',np.linalg.norm(prev_centroid-centroid))
  max_iter-=1

  while (np.linalg.norm(prev_centroid-centroid)>=thresh and max_iter>=0):
    max_iter-=1
    clusters,distortion =assign_cluster(X,centroid)
    prev_centroid = centroid.copy()
    centroid = calc_centroid(centroid,clusters,n_clusters)
    #print('diff  :  ',np.linalg.norm(prev_centroid-centroid))

  return clusters,centroid,distortion