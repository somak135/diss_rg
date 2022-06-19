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

def simulate_graph_deg(N, p = 0.1, d = 1, mode = "hard", replace = False, seed = 1):
    M = np.zeros(shape = (N,N), dtype = int)
    M[0,1] = 1; M[1,0] = 1
    deg_t = np.array([]) ## will store D(0)_t as function of t
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

        deg_t = np.append(deg_t, np.sum(M[0,]))
    
    #deg_seq = np.sum(M, axis = 1)
    #max_deg = max(np.sum(M, axis = 1))

    return deg_t

def find_fixed_mean_sd(N, p, S, d, mode = "hard", replace = False, seed = 1):
    inputs = range(0,S)
    mylist = []
    mylist = Parallel(n_jobs=-1)(delayed(simulate_graph_deg)(N,p,d,mode,replace,(seed+i)) for i in inputs)
    time = range(N-2)
    mean_t = np.array([])
    var_t = np.array([])
    sd_t = np.array([])
    for j in range(len(time)):
      deg_t = np.array([])
      for i in range(S):
        deg_t = np.append(deg_t, mylist[i][j])
      mean_t = np.append(mean_t, np.mean(deg_t))
      #var_t = np.append(var_t, np.var(deg_t))
      sd_t = np.append(sd_t, np.sqrt(np.var(deg_t)))

    return mean_t, sd_t


N = 200; p = 0.1; d = 1; seed = 200; S = 20; mode = "soft"
mytime = range(N-2)
mymean_wr, mysd_wr = find_fixed_mean_sd(N, p, S, d, mode = mode, replace = True, seed = seed)
mymean_wor, mysd_wor = find_fixed_mean_sd(N, p, S, d, mode = mode, replace = False, seed = seed)

title = "Size of final graph = {} || Number of generated samples = {} || Graph parameter p = {}".format(N,S,p)
plt.figure(figsize=(30,6), facecolor="whitesmoke")
plt.subplot(1,2,1)
plt.plot(mytime, mymean_wr, color = "orange")
plt.plot(mytime, mymean_wor, color = "blue")
plt.xlabel("t"); plt.ylabel("mean of D_i(t)")
plt.subplot(1,2,2)
plt.plot(mytime, mysd_wr, color = "orange")
plt.plot(mytime, mysd_wor, color = "blue")
plt.xlabel("t"); plt.ylabel("sd of D_i(t)")
plt.savefig(fname = "1.png", dpi = 800)


import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "mycolabsomak@gmail.com"  # Enter your address
receiver_email = "somaklaha365@gmail.com"  # Enter receiver address
password = "colab@isi24"
message = """\
Subject: Code complete in server

This message is sent from Python. """

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
  server.login(sender_email, password)
  server.sendmail(sender_email, receiver_email, message)
