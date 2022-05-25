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
    #print(p)
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

  return np.array(taboo), (len(np.array(taboo)) - len(np.unique(np.array(taboo))))

# def hardselect(M,n,p):
#     size = int(np.floor(n*p))
#     degree_seq = np.array(np.sum(M, axis = 1)[0:n])
#     if size == 0:
#         taboo = np.array([], dtype = int)
#     else:
#         taboo = np.argpartition(degree_seq, -size)[-size:]
#     return np.array(taboo)

# def hardselect_fixed(M,n,d):
#     size = d
#     degree_seq = np.array(np.sum(M, axis = 1)[0:n])
#     if size == 0:
#         taboo = np.array([], dtype = int)
#     else:
#         taboo = np.argpartition(degree_seq, -size)[-size:]
#     return np.array(taboo)
    

# def softselect(M,n,p, replace = False):
#     size = int(np.floor(n*p))
#     v = np.array(range(0,n))
#     grand_prob = np.sum(M, axis = 1)
#     prob_vec = grand_prob[0:n]/np.sum(grand_prob[0:n])
#     taboo = np.random.choice(v, size = size, replace = replace, p = prob_vec) 
#     return taboo

# def softselect_fixed(M,n,d, replace = False):
#     size = d
#     v = np.array(range(0,n))
#     grand_prob = np.sum(M, axis = 1)
#     prob_vec = grand_prob[0:n]/np.sum(grand_prob[0:n])
#     taboo = np.random.choice(v, size = size, replace = replace, p = prob_vec) 
#     return taboo

def simulate_graph(N, p = 0.1, d = 1, mode = "hard", replace = False, seed = 1):
    M = np.zeros(shape = (N,N), dtype = int)
    M[0,1] = 1; M[1,0] = 1
    np.random.seed(seed)
    repetition = np.array([])
    for i in range(2,N):
        #print(i)
        if mode == "hard":
            taboo, rep = select_taboo(M,i, p = p, mode = "hard")
        elif mode == "hard_fixed":
            taboo, rep = select_taboo(M,i, d = d, mode = "hard_fixed")
        elif mode == "soft":
            taboo, rep = select_taboo(M,i, p = p, mode = "soft", replace = replace)
        elif mode == "soft_fixed":
            taboo, rep = select_taboo(M,i, d = d, mode = "soft_fixed", replace = replace)
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

        repetition = np.append(repetition, rep)
    
    #deg_seq = np.sum(M, axis = 1)
    #max_deg = max(np.sum(M, axis = 1))

    return M, repetition

def find_maximal(N,p = 0.1, d = 1, mode = "hard", replace = False, seed = 1):
    M, repetition = simulate_graph(N,p = p, d = d, mode = mode, replace = replace, seed = seed)
    v = np.sum(M, axis = 1)
    return max(v), sum(v == max(v)), repetition

def maximal_plot(S, N, p = 0.1, d = 1, mode = "hard", replace = False, seed = 1):
    inputs = range(0,S)
    mylist = []
    mylist = Parallel(n_jobs = -1)(delayed(find_maximal)(N,p = p, d = d, mode = mode, replace = replace, seed = (seed+i)) for i in inputs)

    plt.figure(figsize = (30,6),facecolor='whitesmoke')
    
    ##### plot maximal histo
    plt.subplot(1,3,1)
    maximal_list = [item[0] for item in mylist]
    plt.bar(np.arange(0,max(maximal_list)+1), np.bincount(maximal_list, minlength=max(maximal_list))/S)
    #titlename = f"seed = {seed} || Maximal degree distribution in Graph with N={N} and p={p} and d={d} -- {mode}-coded taboo with replace={replace} : Number of iter = {S}"
    #filename = titlename + ".png"
    #plt.title(titlename)
    plt.xlabel("maximal degree"); plt.ylabel("relative frequency")

    ##### plot copy of maximal histo
    plt.subplot(1,3,2)
    copyof_maximal_list = [item[1] for item in mylist]
    plt.bar(np.arange(0,max(copyof_maximal_list)+1), np.bincount(copyof_maximal_list, minlength=max(copyof_maximal_list))/S)
    #titlename = f"seed = {seed} || Distribution of number of Maximal degrees in Graph with N={N} and p={p} and d={d} -- {mode}-coded taboo with replace={replace} : Number of iter = {S}"
    #filename = titlename + ".png"
    #plt.title(titlename)
    plt.xlabel("copy of maximal degree"); plt.ylabel("relative frequency")

    ##### plot copy of maximal histo
    plt.subplot(1,3,3)
    repetitions = [item[2] for item in mylist]
    avg_repetitions = (np.sum(repetitions, axis = 0)/range(3, N+1))/S
    plt.plot(avg_repetitions)
    #titlename = f"seed = {seed} || Average proportion of repeated taboo in Graph with N={N} and p={p} and d={d} -- {mode}-coded taboo with replace={replace} : Number of iter = {S}"
    #filename = titlename + ".png"
    #plt.title(titlename)
    plt.xlabel("avg. repetitions in taboo-ing"); plt.ylabel("relative frequency")

    plt.savefig(fname = filename, dpi = 600)
    #return mylist

######################################################################################################
######################################################################################################

def diffp(S, N, p, seed):
  t1 = time.time()
  #S = 10
  #N = 100
  #seed = 1
  #p = 0.1
  #d = 3
  mode = "soft"
  
  
  inputs = range(0, S)
  mylist1 = []; mylist2 = []
  mylist1 = Parallel(n_jobs = -1)(delayed(find_maximal)(N, p = p, mode = "soft", replace = False, seed = (seed+i+1)) for i in inputs)
  mylist2 = Parallel(n_jobs = -1)(delayed(find_maximal)(N, p = p, mode = "soft", replace = True, seed = (seed+i)) for i in inputs)
  
  
  plt.figure(figsize = (30,6),facecolor='whitesmoke')
  ### plot two maximal histo
  plt.subplot(1,3,1)
  maximal_list1 = [item[0] for item in mylist1]
  maximal_list2 = [item[0] for item in mylist2]
  xlim = max(max(maximal_list1), max(maximal_list2)) + 1
  plt.bar(np.arange(0,xlim), 
          np.bincount(maximal_list1, minlength=xlim)/S, color = "red", alpha = 0.5)
  plt.bar(np.arange(0,xlim), 
          np.bincount(maximal_list2, minlength=xlim)/S, color = "blue", alpha = 0.5)
  #titlename = f"seed = {seed} || Maximal degree distribution in Graph with N={N} and p={p} and d={d} -- {mode}-coded taboo : Number of iter = {S}"
  #filename = titlename + ".png"
  #plt.title(titlename)
  plt.xlabel("maximal degree"); plt.ylabel("rel. freq.")
  
  ### plot two copy of maximal histo
  plt.subplot(1,3,2)
  copyof_maximal_list1 = [item[1] for item in mylist1]
  copyof_maximal_list2 = [item[1] for item in mylist2]
  xlim = max(max(copyof_maximal_list1), max(copyof_maximal_list2))+1
  plt.bar(np.arange(0,xlim),
          np.bincount(copyof_maximal_list1, minlength=xlim)/S, color = "red", alpha = 0.5, label = "Without replacement")
  plt.bar(np.arange(0,xlim),
          np.bincount(copyof_maximal_list2, minlength=xlim)/S, color = "blue", alpha = 0.5, label = "With replacement")
  # titlename = f"seed = {seed} || Maximal degree distribution in Graph with N={N} and p={p} and d={d} -- {mode}-coded taboo : Number of iter = {S}"
  # filename = titlename + ".png"
  # plt.title(titlename)
  plt.legend(loc = "best")
  plt.xlabel("copy of maximal degree"); plt.ylabel("rel. freq.")
  
  ### plot 
  plt.subplot(1,3,3)
  repetitions = [item[2] for item in mylist2]
  avg_repetitions = (np.sum(repetitions, axis = 0)/range(3, N+1))/S
  plt.plot(avg_repetitions)
  #titlename = f"seed = {seed} || Average proportion of repeated taboo in Graph with N={N} and p={p} and d={d} -- {mode}-coded taboo with replace={replace} : Number of iter = {S}"
  #filename = titlename + ".png"
  #plt.title(titlename)
  plt.xlabel("time"); plt.ylabel("avg. repetitions in taboo-ing")
  
  #plt.savefig(fname = filename, dpi = 600)
  #return mylist
  
  ######### image save and emailing
  now = datetime.now()
  string = now.strftime("%d%m%Y %H%M%S")
  figname = "image" + string + " p {}".format(p) + ".png"
  plt.savefig(fname = figname, dpi = 800)
  t2 = time.time()
  return figname
  
  ########### send completion email
  # import smtplib, ssl
  
  # port = 465  # For SSL
  # smtp_server = "smtp.gmail.com"
  # sender_email = "mycolabsomak@gmail.com"  # Enter your address
  # receiver_email = "somaklaha365@gmail.com"  # Enter receiver address
  # password = "colab@isi24"
  # message = """\
  # Subject: Code complete in server
  
  # This message is sent from Python. Image saved is {}. Time taken is {} seconds.""".format(figname, round(t2 - t1))
  
  # context = ssl.create_default_context()
  # with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
  #     server.login(sender_email, password)
  #     server.sendmail(sender_email, receiver_email, message)
      
      
listp = np.array([0])
fignames = np.array([])
for i in listp:
  fignames = np.append(fignames, diffp(S = 1000, N = 4000, p = i, seed = np.int(1000*i)))

########### send completion email
t_end = time.time()
import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "mycolabsomak@gmail.com"  # Enter your address
receiver_email = "somaklaha365@gmail.com"  # Enter receiver address
password = "colab@isi24"
message = """\
Subject: Code complete in server

This message is sent from Python. Time taken is {} seconds starting from {} to {}. Figures are:
  {}""".format(round(t_end - t_start), t_start, t_end, fignames)

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
  server.login(sender_email, password)
  server.sendmail(sender_email, receiver_email, message)
