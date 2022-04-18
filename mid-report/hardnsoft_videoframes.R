#install.packages("igraph")
library(igraph)

mat = matrix(c(0,1,1,1,0,1,1,1,0), nrow = 3)
g1 = graph_from_adjacency_matrix(mat, mode = "max", weighted = NULL, diag = FALSE, add.colnames = NA, 
                            add.rownames = NA, vertex.color = c("red", "green", "blue"))
plot(g1)

adjm <- matrix(sample(0:1, 100, replace=TRUE, prob=c(0.9,0.1)), nc=10)
g1 <- graph_from_adjacency_matrix( adjm )
plot(g1)

append_zero = function(mat) {
  d = dim(mat)[1]
  v1 = rep(0, d)
  v2 = rep(0, d+1)
  mat1 = rbind(mat, v1)
  mat1 = cbind(mat1, v2)
  return(mat1)
}

mat1 = matrix(c(0,1,1,0), nrow = 2)
g1 = graph_from_adjacency_matrix(mat1, mode = "undirected", add.colnames = NA, 
                                 add.rownames = NA)
E(g1)$color = "black"
V(g1)$color = "orange"
plot(g1, layout = layout_in_circle)

mat2 = append_zero(mat1)
g2 = graph_from_adjacency_matrix(mat2, mode = "undirected", add.colnames = NA, 
                                 add.rownames = NA)
E(g2)$color = "black"
V(g2)$color = "orange"
V(g2)$color[length(V(g2)$color)] = "green"
plot(g2, layout = layout_in_circle)


hardcore = function(M, n, p) {
  taboo_num = floor(n*p)
  degree_seq = rowSums(M)[1:n]
  taboo = tail(sort(degree_seq, index.return = TRUE)$ix, taboo_num)
  return(taboo)
}

softcore = function(M, n, p) {
  taboo_num = floor(n*p)
  degree_seq = rowSums(M)[1:n]
  prob_vec = degree_seq/sum(degree_seq)
  taboo = sample(c(1:n), size = taboo_num, replace = FALSE, prob = prob_vec)
  #taboo = match(taboo, degree_seq)
  return(taboo)
}

fun = function(N, p, mode, seed) {
  set.seed(seed)
  adjmat = matrix(0, nrow = 2, ncol = 2)
  adjmat[1,2] = 1
  adjmat[2,1] = 1
  g1 = graph_from_adjacency_matrix(adjmat, mode = "undirected", add.colnames = NA,
                                   add.rownames = NA)
  E(g1)$color = "black"
  V(g1)$color = "orange"
  #dev.new()
  png(filename = "1.png", width = 1200, height = 1200)
  plot(g1, layout = layout_in_circle)
  dev.off()
  
  for(i in 3:N) {
    adjmat = append_zero(adjmat)
    if(mode == "hard") {
      taboo = hardcore(adjmat, (i-1), p)
    }
    if(mode == "soft") {
      taboo = softcore(adjmat, (i-1), p)
    }
    g = graph_from_adjacency_matrix(adjmat, mode = "undirected", add.colnames = NA,
                                    add.rownames = NA)
    E(g)$color = "black"
    V(g)$color = "orange"
    V(g)$color[taboo] = "red"
    V(g)$color[i] = "green"
    #dev.new()
    png(filename = paste0(i,"1", ".png"), width = 1200, height = 1200)
    plot(g, layout = layout_in_circle)
    dev.off()
    
    ########################################## then join edge
    
    nontaboo = setdiff(c(1:(i-1)), taboo)
    prob_vec = rowSums(adjmat[nontaboo, nontaboo])/sum(rowSums(adjmat[nontaboo, nontaboo]))
    jointo = sample(nontaboo, size = 1, prob = prob_vec)
    adjmat[i, jointo] = 1
    adjmat[jointo, i] = 1
    g = graph_from_adjacency_matrix(adjmat, mode = "undirected", add.colnames = NA,
                                    add.rownames = NA)
    E(g)$color = "black"
    V(g)$color = "orange"
    V(g)$color[taboo] = "red"
    V(g)$color[i] = "green"
    png(filename = paste0(i,"2", ".png"), width = 1200, height = 1200)
    plot(g, layout = layout_in_circle)
    dev.off()
  }
}

fun(15, p = 0.2, mode = "hard", seed = 1)
fun(15, p = 0.2, mode = "soft", seed = 1)

