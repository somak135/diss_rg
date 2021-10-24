N = 1000; p = 0.1
M = matrix(0, nrow = N, ncol = N)
M[1,2] = 1; M[2,1] = 1


hardselect = function(n,p) {
  size = floor(n*p)
  v = c(1:n)
  if(size == 0) {
    return(integer(0))
  }
  else {
    taboo = v[1:size]
    return(taboo)
  }
}

softselect = function(M,n,p) {
  size = floor(n*p)
  v = c(1:n)
  grand_prob = rowSums(M)
  prob_vec = grand_prob[1:n]
  taboo = sample(v, size = size, replace = F, prob = prob_vec)
  return(taboo)
}

for(i in 3:N) {
  #set.seed(i)
  taboo = hardselect((i-1), p)
  v = c(1:(i-1))
  nontaboo = v[!v %in% taboo]
  nontaboo_incidence = M[nontaboo, ]
  prob_vec = rowSums(M[nontaboo, ])
  edge_end = sample(nontaboo, 1, prob = prob_vec)
  M[i, edge_end] = 1
  M[edge_end, i] = 1
}
#hist(rowSums(M), freq = F)
print(max(rowSums(M)))