import numpy as np
import random
import multiprocessing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from datetime import datetime
import time

t_start = time.time()
def select_taboo(M, n, p = 0.1, d = 1, mode = "hard", replace = False):
  if mode == "hard":
    size = int(np.floor(n*p))
    degree_seq = np.array(np.sum(M, axis = 1)[0:n])
    if size == 0:
        taboo = np.array([], dtype = int)
    else:
        taboo = np.argpartition(degree_seq, -size)[-size:]

  elif mode == "hard_fixed":
    size = d
    degree_seq = np.array(np.sum(M, axis = 1)[0:n])
    if size == 0 or size >= n:
        taboo = np.array([], dtype = int)
    else:
        taboo = np.argpartition(degree_seq, -size)[-size:]

  elif mode == "soft":
    print(p)
    size = int(np.floor(n*p))
    v = np.array(range(0,n))
    grand_prob = np.sum(M, axis = 1)
    prob_vec = grand_prob[0:n]/np.sum(grand_prob[0:n])
    taboo = np.random.choice(v, size = size, replace = replace, p = prob_vec) 

  elif mode == "soft_fixed":
    size = d
    if size >= n:
      taboo = np.array([], dtype = int)
    else:
      v = np.array(range(0,n))
      grand_prob = np.sum(M, axis = 1)
      prob_vec = grand_prob[0:n]/np.sum(grand_prob[0:n])
      taboo = np.random.choice(v, size = size, replace = replace, p = prob_vec)

  return np.array(taboo)

def simulate_graph_max(N, p = 0.1, d = 1, mode = "hard", replace = False, seed = 1):
    M = np.zeros(shape = (N,N), dtype = int)
    M[0,1] = 1; M[1,0] = 1
    #deg_t = np.array([]) ## will store D(0)_t as function of t
    np.random.seed(seed)
    
    for i in range(2,N):
        #print(i)
        if mode == "hard":
            taboo = select_taboo(M,i, p = p, mode = "hard")
        elif mode == "hard_fixed":
            taboo = select_taboo(M,i, d = d, mode = "hard_fixed")
        elif mode == "soft":
            taboo = select_taboo(M,i, p = p, mode = "soft", replace = replace)
        elif mode == "soft_fixed":
            taboo = select_taboo(M,i, d = d, mode = "soft_fixed", replace = replace)
        else:
            break
            raise ValueError("no such mode of selecting taboo")
            
        
        v = range(0, i)
        nontaboo = np.setxor1d(v, taboo, assume_unique=True)
        nontaboo_incidence = M[nontaboo]
        prob_vec = np.sum(nontaboo_incidence, axis = 1)
        edge_end = np.random.choice(nontaboo, p = prob_vec/sum(prob_vec))
        M[i, edge_end] = 1
        M[edge_end, i] = 1

        #deg_t = np.append(deg_t, np.sum(M[0,]))
    
    #deg_seq = np.sum(M, axis = 1)
    max_deg = max(np.sum(M, axis = 1))

    #deg_seq = np.sort(np.sum(M, axis = 1))
    return max_deg

# def find_deg_seq(N, p, d, S, mode = "hard", replace = False, seed = 1):
#     inputs = range(0,S)
#     mylist = []
#     mylist = Parallel(n_jobs=-1)(delayed(simulate_graph_deg)(N,p,d,mode,replace,(seed+i)) for i in inputs) ## matrix structure containing copies of deg sequences

#     min_deg = np.min(mylist)
#     max_deg = np.max(mylist)
#     deg_vec = np.zeros(shape = (S, max_deg+1-min_deg), dtype = float)
#     for i in range(0,S):
#       s = np.array([])
#       for j in range(min_deg, (max_deg+1)):
#         s = np.append(s, np.count_nonzero(mylist[i] == j))
      
#       deg_vec[i, ] = s/N

#     mean_deg_vec = np.array([]); sd_deg_vec = np.array([]); var_deg_vec = np.array([])
#     for j in range(min_deg, (max_deg+1)):
#       p_j = np.array([])
#       for i in range(0,S):
#         p_j = np.append(p_j, deg_vec[i][(j-min_deg)])
#       mean_deg_vec = np.append(mean_deg_vec, np.mean(p_j))
#       var_deg_vec = np.append(var_deg_vec, np.var(p_j))
#       sd_deg_vec = np.append(sd_deg_vec, np.sqrt(np.var(p_j)))


#     #return mylist, deg_vec, min_deg, max_deg
#     return deg_vec, min_deg, max_deg, mean_deg_vec, var_deg_vec, sd_deg_vec

def maximal_plot(N,p,d,S,mode = "hard", replace = True, seed = 1,filename):
    inputs = range(0,S)
    mylist = []
    mylist = Parallel(n_jobs = -1)(delayed(simulate_graph_max)(N,p,d,mode,replace,(seed+i)) for i in inputs)
    plt.figure(figsize = (12,6),facecolor='whitesmoke')
    plt.bar(np.arange(0,max(mylist)+1), np.bincount(mylist, minlength=max(mylist))/S)
    #titlename = f"seed = {seed} || Maximal degree distribution in Graph of size {N} and p={p} -- {mode}-coded taboo : Number of iter = {S}"
    #filename = titlename + ".png"
    #plt.title(titlename)
    plt.xlabel("maximal degree"); plt.ylabel("relative frequency")
    plt.savefig(fname = filename, dpi = 600)
    #return mylist
    return np.mean(mylist)


plist = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
arr = np.array([])
for p in plist:
    fname = "p{}.png".format(p)
    arr = np.append(maximal_plot(N, p, 1, S, mode, replace, seed, fname))

plt.plot(plist, arr, color = "blue")
plt.xlabel("p"); plt.ylabel("average maximal degree")
plt.savefig(fname = "avg max.png", dpi = 800)