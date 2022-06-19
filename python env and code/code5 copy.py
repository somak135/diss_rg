#! /usr/bin/python3

import numpy as np
import random
import multiprocessing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import itertools


def hardselect(M,n,p):
    size = int(np.floor(n*p))
    degree_seq = np.array(np.sum(M, axis = 1)[0:n])
    if size == 0:
        taboo = np.array([], dtype = int)
    else:
        taboo = np.argpartition(degree_seq, -size)[-size:]
    return np.array(taboo)
    

def softselect(M,n,p):
    size = int(np.floor(n*p))
    v = np.array(range(0,n))
    grand_prob = np.sum(M, axis = 1)
    prob_vec = grand_prob[0:n]/np.sum(grand_prob[0:n])
    taboo = np.random.choice(v, size = size, replace = False, p = prob_vec) #checked -- updates prob
    return taboo

def softselect_one(M,n):
    size = 1
    v = np.array(range(0,n))
    grand_prob = np.sum(M, axis = 1)
    prob_vec = grand_prob[0:n]/np.sum(grand_prob[0:n])
    taboo = np.random.choice(v, size = size, replace = False, p = prob_vec) #checked -- updates prob
    return taboo

def find_prod_taboo(M, taboo):
    taboo_deg = np.sum(M[taboo, ], axis = 1)
    return(np.prod(taboo_deg))

def find_sum_taboo(M, taboo):
    taboo_deg = np.sum(M[taboo, ], axis = 1)
    return(np.sum(taboo_deg))

def simulate_graph(N,p, vertex = 1, mode = "hard",seed = 1):
    M = np.zeros(shape = (N,N), dtype = int)
    M[0,1] = 1; M[1,0] = 1
    vertex_deg = np.array([]) ### stores D_i(t) for t in 1:N
    p_i_prime = np.array([]) ### stores P_i'(t) for t in 1:N
    comp_vertex_deg = np.array([]) ### stores compensated D_i(t) which is a mart for t in 1:N
    P_taboo_t = np.array([])
    grandsum_t = np.array([])
    compensation_t = np.array([])

    np.random.seed(seed)
    for i in range(2,N):
        if mode == "hard":
            taboo = hardselect(M,i,p)
        elif mode == "soft":
            taboo = softselect(M,i,p)
        elif mode == "soft_one":
            taboo = softselect_one(M,i)
        else:
            raise ValueError("no such mode of selecting taboo")
        
        v = range(0, i)
        nontaboo = np.setxor1d(v, taboo, assume_unique=True)
        nontaboo_incidence = M[nontaboo]
        prob_vec = np.sum(nontaboo_incidence, axis = 1)
        edge_end = np.random.choice(nontaboo, p = prob_vec/sum(prob_vec))
        M[i, edge_end] = 1
        M[edge_end, i] = 1

        ### value of D_vertex(i)
        degree = np.sum(M, axis = 1)[vertex]

        ### value of P_taboo(i)
        P_taboo = find_prod_taboo(M, taboo)

        ### finding the values of products of degrees for all choices of taboo
        subsets = itertools.combinations(v, int(np.floor(i*p)))
        P_subsets = Parallel(n_jobs=-1)(delayed(find_prod_taboo)(M, subset) for subset in subsets)
        grandsum = np.sum(P_subsets)

        ### sum of degrees of nontaboo vertices
        D_nontaboo = find_sum_taboo(M, nontaboo)

        ### compensation
        compensation = np.prod(1+p_i_prime[:-1])

        if i >= vertex:
          vertex_deg = np.append(vertex_deg, degree)
          p_i_prime = np.append(p_i_prime, P_taboo/(grandsum*D_nontaboo))
          P_taboo_t = np.append(P_taboo_t, P_taboo)
          grandsum_t = np.append(grandsum_t, grandsum)
          compensation_t = np.append(compensation_t, compensation)
          comp_vertex_deg = np.append(comp_vertex_deg, degree/compensation)

    #deg_seq = np.sum(M, axis = 1)
    #max_deg = max(np.sum(M, axis = 1))

    return vertex_deg, p_i_prime, P_taboo_t, grandsum_t, compensation_t, comp_vertex_deg

def find_fixed(N, p, S, vertex = 1, mode = "hard", seed = 1):
    inputs = range(0,S)
    mylist = []
    mylist = Parallel(n_jobs=-1)(delayed(simulate_graph)(N,p,vertex,mode,(seed+i)) for i in inputs)
    time = range(vertex, (N-2))
    mean_d_i_t = np.array([])
    mean_p_taboo_t = np.array([])
    mean_grandsum_t = np.array([])
    mean_compensation_t = np.array([])
    mean_d_i_star_t = np.array([])
    #var_t = np.array([])
    #sd_t = np.array([])
    for j in range(len(time)):
      d_i_t = np.array([])
      p_taboo_t = np.array([])
      grandsum_t = np.array([])
      compensation_t = np.array([])
      d_i_star_t = np.array([])

      for i in range(S):
        #print("i: ", i); print("j: ", j)
        d_i_t = np.append(d_i_t, mylist[i][0][j])
        p_taboo_t = np.append(p_taboo_t, mylist[i][2][j])
        grandsum_t = np.append(grandsum_t, mylist[i][3][j])
        compensation_t = np.append(compensation_t, mylist[i][4][j])
        d_i_star_t = np.append(d_i_star_t, mylist[i][5][j])

      mean_d_i_t = np.append(mean_d_i_t, np.mean(d_i_t))
      mean_p_taboo_t = np.append(mean_p_taboo_t, np.mean(p_taboo_t))
      mean_grandsum_t = np.append(mean_grandsum_t, np.mean(grandsum_t))
      mean_compensation_t = np.append(mean_compensation_t, np.mean(compensation_t))
      mean_d_i_star_t = np.append(mean_d_i_star_t, np.mean(d_i_star_t))
      #var_t = np.append(var_t, np.var(deg_t))
      #sd_t = np.append(sd_t, np.sqrt(np.var(deg_t)))

    return time, mean_d_i_t, mean_p_taboo_t, mean_grandsum_t, mean_compensation_t, mean_d_i_star_t

#inputs = range(0,5)
#mylist = Parallel(n_jobs=-1)(delayed(simulate_graph)(20,0.2,1,"soft",(10+i)) for i in inputs)
#print(mylist)
  
  
#### fit polynomial to D_i(t) in softcore taboo
N = 1000; p = 0.1; seed = 1; S = 50; i =1; mode = "soft"
mytime, mean_d_i_t, mean_p_taboo_t, mean_grandsum_t, mean_compensation_t, mean_d_i_star_t = find_fixed(N, p, S, vertex = i, mode = mode, seed = seed)
title = "seed = {} || {}-th vertex || Graph parameters N = {}, p = {}, mode = {} || no. of iter = {}".format(seed, i, N, p, mode, S)
plt.figure(figsize=(30,15), facecolor="whitesmoke")
plt.subplot(2,3,1)
plt.plot(mytime, mean_d_i_t, color = "orange")
plt.title(title)
plt.xlabel("t"); plt.ylabel("mean of D_i(t)")

plt.subplot(2,3,2)
plt.plot(mytime, mean_p_taboo_t, color = "red")
plt.xlabel("t"); plt.ylabel("mean of P_taboo(t)")

plt.subplot(2,3,3)
plt.plot(mytime, mean_grandsum_t, color = "blue")
plt.xlabel("t"); plt.ylabel("mean of grandsum(t)")

plt.subplot(2,3,4)
plt.plot(mytime, mean_compensation_t, color = "orange")
plt.xlabel("t"); plt.ylabel("mean of compensation_i(t)")

plt.subplot(2,3,5)
plt.plot(mytime, mean_d_i_star_t, color = "red")
plt.xlabel("t"); plt.ylabel("mean of D_i_star(t)")
#plt.savefig("image2.png")


######### image save and emailing
now = datetime.now()
string = now.strftime("%d%m%Y %H%M%S")
figname = "image" + string + ".png"
plt.savefig(figname)

########### send completion email
import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "mycolabsomak@gmail.com"  # Enter your address
receiver_email = "somaklaha365@gmail.com"  # Enter receiver address
password = "colab@isi24"
message = """\
Subject: Code complete in server

This message is sent from Python. Image saved is {}.""".format(figname)

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)