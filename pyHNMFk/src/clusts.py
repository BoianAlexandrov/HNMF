import numpy as np
import pandas as pd
import scipy,sys,os,pdb
from sklearn import metrics
from multiprocessing import current_process
from functools import partial
from numpy import linalg as LA
from sklearn import mixture
from scipy.spatial.distance import cdist
from scipy.spatial.distance import correlation as cor
from scipy.optimize import linear_sum_assignment
from scipy.optimize import nnls
from numpy.random import SeedSequence
from numpy.random import Generator, PCG64DXSM, SeedSequence
import math

import multiprocessing

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin



def cluster_converge_outerloop(Wall, Hall, totalprocess, dist="cosine",
                               gpu=False, cluster_rand_seq=None, n_cpu=-1):
    
    avgSilhouetteCoefficients = -1  # intial avgSilhouetteCoefficients 
    
    #do the parallel clustering 
    result_list = parallel_clustering(Wall, Hall, totalprocess, iterations=50,
                                      n_cpu=n_cpu,  dist=dist, gpu=gpu,
                                      cluster_rand_seq=cluster_rand_seq)
    
    for i in range(50):  # using 10 iterations to get the best clustering 
        
        temp_processAvg, temp_exposureAvg, temp_processSTE,  temp_exposureSTE, temp_avgSilhouetteCoefficients, temp_clusterSilhouetteCoefficients = result_list[i][0], result_list[i][1], result_list[i][2], result_list[i][3], result_list[i][4], result_list[i][5]
        
        if avgSilhouetteCoefficients < temp_avgSilhouetteCoefficients:
              processAvg, exposureAvg, processSTE,  exposureSTE, avgSilhouetteCoefficients, clusterSilhouetteCoefficients =   temp_processAvg, temp_exposureAvg, temp_processSTE,  temp_exposureSTE, temp_avgSilhouetteCoefficients, temp_clusterSilhouetteCoefficients
        
      
    return  processAvg, exposureAvg, processSTE,  exposureSTE, avgSilhouetteCoefficients, clusterSilhouetteCoefficients

def parallel_clustering(Wall, Hall, totalProcesses, iterations=50,  n_cpu=-1, dist= "cosine", gpu=False, cluster_rand_seq=None):
    
    if n_cpu==-1:
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(processes=n_cpu)

    # create random generators for each subprocess
    sub_rand_generator = cluster_rand_seq.spawn(iterations)
    iteration_generator_pairs = []
    # pair generator with an interation
    for i,j in zip(range(iterations), sub_rand_generator):
        iteration_generator_pairs.append([i,j])
    # import pdb; pdb.set_trace()
    pool_nmf=partial(cluster_converge_innerloop, Wall, Hall, totalProcesses, dist=dist, gpu=gpu)
    result_list = pool.map(pool_nmf, iteration_generator_pairs) 
    pool.close()
    pool.join()
    return result_list

def cluster_converge_innerloop(Wall, Hall, totalprocess, iteration_generator_pair, iteration=1, dist="cosine", gpu=False):

    
    
    rng_generator = Generator(PCG64DXSM(iteration_generator_pair[1]))
    processAvg = rng_generator.random((Wall.shape[0],totalprocess))
    exposureAvg = rng_generator.random((totalprocess, Hall.shape[1]))
    
    result = 0
    convergence_count = 0
    while True:
        processAvg, exposureAvg, processSTE,  exposureSTE, avgSilhouetteCoefficients, clusterSilhouetteCoefficients = reclustering(Wall, Hall, processAvg, exposureAvg, dist=dist, gpu=gpu)
        
        if result == avgSilhouetteCoefficients:
            break
        elif convergence_count == 10:
            break
        else:
            result = avgSilhouetteCoefficients
            convergence_count = convergence_count + 1
        
    return processAvg, exposureAvg, processSTE,  exposureSTE, avgSilhouetteCoefficients, clusterSilhouetteCoefficients

def reclustering(tempWall=0, tempHall=0, processAvg=0, exposureAvg=0, dist="cosine",gpu=False):
    # exposureAvg is not important here. It can be any matrix with the same size of a single exposure matrix
    iterations = int(tempWall.shape[1]/processAvg.shape[1])
    processes =  processAvg.shape[1]
    idxIter = list(range(0, tempWall.shape[1], processes))
   
    processes3D = np.zeros([processAvg.shape[1], processAvg.shape[0], iterations])
    exposure3D = np.zeros([exposureAvg.shape[0], iterations, exposureAvg.shape[1]])
    
    for  iteration_number in range(len(idxIter)):
        
        statidx = idxIter[iteration_number]
        loopidx = list(range(statidx, statidx+processes))
        idxPair= pairwise_cluster_raw(mat1=processAvg, mat2=tempWall[:, loopidx], dist=dist)
        # from IPython.core.debugger import Pdb; Pdb().set_trace()
        
        for cluster_items in idxPair:
            cluster_number = cluster_items[0]
            query_idx = cluster_items[1]
            processes3D[cluster_number,:,iteration_number]=tempWall[:,statidx+query_idx]
            exposure3D[cluster_number, iteration_number, :] = tempHall[statidx+query_idx,:]

    count = 0
    labels=[]
    clusters = pd.DataFrame()
    
    for cluster_id in range(processes3D.shape[0]):
        cluster_vectors = pd.DataFrame(processes3D[cluster_id,:,:])
        clusters = pd.concat([clusters,cluster_vectors.T])
        # clusters = clusters.append(cluster_vectors.T)
        for k in range(cluster_vectors.shape[1]):
            labels.append(count)
        count= count+1

    try:
        if dist=="cosine":
            SilhouetteCoefficients = metrics.silhouette_samples(clusters, labels, metric='cosine')
        if dist=="correlation":
            SilhouetteCoefficients = metrics.silhouette_samples(clusters, labels, metric='correlation')
        
    except:
        SilhouetteCoefficients = np.ones((len(labels),1))

    avgSilhouetteCoefficients = np.mean(SilhouetteCoefficients)
    
    #clusterSilhouetteCoefficients 
    splitByCluster = np.array_split(SilhouetteCoefficients, processes3D.shape[0])
    clusterSilhouetteCoefficients = np.array([])
    for i in splitByCluster:
        
        clusterSilhouetteCoefficients=np.append(clusterSilhouetteCoefficients, np.mean(i))
        
    processAvg = np.mean(processes3D, axis=2).T
    processSTE = scipy.stats.sem(processes3D, axis=2, ddof=1).T
    exposureAvg = np.mean(exposure3D, axis=1) 
    exposureSTE = scipy.stats.sem(exposure3D, axis=1, ddof=1)
    
        
    return  processAvg, exposureAvg, processSTE,  exposureSTE, avgSilhouetteCoefficients, clusterSilhouetteCoefficients

def pairwise_cluster_raw(mat1=([0]), mat2=([0]), dist="cosine"):  # the matrices (mat1 and mat2) are used to calculate the clusters and the lsts will be used to store the members of clusters

    if dist=="cosine":
        con_mat = cdist(mat1.T, mat2.T, "cosine")
    elif dist=="correlation":
        con_mat = cdist(mat1.T, mat2.T, "correlation")

    row_ind, col_ind = linear_sum_assignment(con_mat)

    idxPair=[]
    for i, j in zip(row_ind, col_ind):
        idxPair.append([i,j])
        
    return idxPair


# cluster snalysis functions for selecting best k value

def AIC(sill_avg, recon, num_resids, num_sources, sill_cutoff=0.7):
    aic_scores = []
    if sill_avg > sill_cutoff:
        aic = 2*num_sources + num_resids*math.log(recon/num_resids)
    else:
        aic = np.inf
    return aic

def result_analysis(clust_points, recon, num_resids, num_sources):
    # num_sources = res['num_sources'].iloc[0]
    # points = res['points'].sum()
    # recon = res['normF'].mean()
    if num_sources == 1:
        centers = pd.DataFrame(clust_points).mean().tolist()
        avgsill = 1
        min_Sil = 1
    else:
        Wall = np.asarray(clust_points).T
        Hall = np.random.rand (Wall.shape[1], 10)  
        seed_seq     = SeedSequence()
        cluster_generators   = seed_seq.spawn(num_sources)
        cluster_rand_seq = cluster_generators[num_sources-1]
        print("Clustering ",num_sources)
        centers, random_exposures, processSTE,  exposureSTE, avgSilhouetteCoefficients, clusterSilhouetteCoefficients=cluster_converge_outerloop(Wall, Hall, num_sources, dist="cosine" ,cluster_rand_seq=cluster_rand_seq,n_cpu=-1)
        avgsill =  avgSilhouetteCoefficients
        min_Sil= clusterSilhouetteCoefficients.min()

    aic_score = AIC(avgsill, recon, num_resids, num_sources)

    return pd.DataFrame({
        'num_source': [num_sources],
        'aic_score': [aic_score],
        'avg_sillhouette_score': [avgsill],
        'min_sillhouette_score': [min_Sil],
        'reconstruction_loss': [recon],
        'centers': [centers]
    })