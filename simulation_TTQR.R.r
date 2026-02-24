######################
### Generate X, y
######################
DGP = function(n, coef, sigma, tau, dim = c(10,10,10)) {
  ###model: Y_i=<A,X_i>+ep_i
  ###sigma: sd of error term epsilon (ep)
  ###tau: tau th quantile
  ###dim: dimension of tensor A
  ###rank: rank of tensor A
  signal = vector(length = n)
  X = list()
  for (i in 1:n) {
    X_i = rand_tensor(modes = dim)
    X[[i]] = X_i
    signal[i] = vec(coef)%*%vec(X_i)
    #print(i)
  }
  q_tau = qnorm(tau,0,1)
  ep = rnorm(n,mean = -q_tau, sd = sigma)
  y = signal + ep
  rtn = list(X = X, y = y)
  return(rtn)
}


X_unfold = function(tnsrls, mode){
  #vectorize tensor X_i by vectorize mode-k matricization of X_i---vec(X_(k))
  #and storage in a n-row matrix
  ###tnsrls: a list of tensor
  n = length(tnsrls)
  dim = tnsrls[[1]]@modes
  xvec = k_unfold(tnsrls[[1]],mode)
  X_vec = matrix(xvec@data,nrow = 1)
  for (i in 2:n) {
    xvec = k_unfold(tnsrls[[i]],mode)
    X_vec = rbind(X_vec,matrix(xvec@data,nrow = 1))
  }
  return(X_vec)
}

########################################
### sparse alternating QR, estimate A
########################################

Highdim_AQR = function(y, X, X_vec, A0, rank, tau, lambda, penalty = c(TRUE,TRUE,FALSE), tol=5e-2, iter_max){
  ###y: scalar response,  X: list of tensor covariates
  ###X_vec: list of vectorized X,  X_unfold(X,mode = k)
  ###A0: initial
  ###rank: predetermined multilinear rank
  ###tau: quantile
  ###lambda: tuning
  ###tol: convergence threshold, iter_max: maximum number of iterations
  
  
  order = length(rank)
  dim = A0@modes
  n = length(y)
  ######HOSVD for A0#######
  h = hosvd(A0,ranks = rank)
  G = h$Z
  U = h$U
  
  #######iteration#########
  tolerance = 1
  iter = 0
  obj = vector(length = iter_max) #store objective function value 
  while(tolerance>=tol & iter<iter_max){
    
    ####compute U_k####
    lambdas =  sapply(U,function(x) norm(matrix(x,ncol = 1),"1"))
    lambdas = lambdas[penalty]
    #k=1
    Kr_U = U[[order]]
    if(order > 2){
      for (k in (order-1):2) {
        Kr_U = kronecker(Kr_U,U[[k]])
      }
    }
    mat = Kr_U%*% t(k_unfold(G,1)@data)
    mat = kronecker(mat,diag(dim[1]))
    X_Uk = X_vec[[1]] %*% mat  
    lambdak =  prod(lambdas[-1])*lambda
    est = Uk_ADMM(y, X = X_Uk, B0 = U[[1]], tau = tau, lambda = lambdak, gamma = 1, tol = 2e-4, iter_max = 200) 
    U[[1]] = est$B
    lambdas[1] = norm(matrix(U[[1]],ncol = 1),"1")
    #K>=2
    for (k in 2:order) {
      Kr_U = U[[1]]
      for (i  in (1:order)[-c(1,k)]) {
        Kr_U = kronecker(U[[i]],Kr_U)
      }
      mat = Kr_U%*%t(k_unfold(G,k)@data)
      mat = kronecker(mat,diag(dim[k]))
      X_Uk = X_vec[[k]] %*% mat
      
      if(penalty[k]){
        lambdak = prod(lambdas[-k])*lambda
        est = Uk_ADMM(y, X = X_Uk, B0 = U[[k]], tau = tau, lambda = lambdak, gamma = 1, tol = 2e-4, iter_max = 200)
        U[[k]] = est$B
      }else{
        Uk = rq(y~X_Uk,tau = tau)$coefficients[-1]
        Uk = matrix(Uk,nrow = dim[k],ncol = rank[k])
        U[[k]] = apply(Uk,2,FUN = function(x) x/sqrt(sum(x^2)))
      }
    }
    
    ####compute G#####
    Kr_U = U[[order]]
    if(order > 2){
      for (k in (order-1):2) {
        Kr_U = kronecker(Kr_U,U[[k]])
      }
    }
    X_G = X_vec[[1]] %*% kronecker(Kr_U,U[[1]]) 
    G = rq(y~ 0+X_G,tau = tau)$coefficients
    G = array(G, dim = rank)
    G = as.tensor(G)
    
    A = ttl(G, U, ms = 1:order)
    tolerance = fnorm(A-A0)/max(1,fnorm(A0))
    A0 = A
    iter = iter+1
    
    A_vec = vec(A)
    res = y-X_vec[[1]] %*%A_vec
    loss = mean(res*(tau-(res<0)))
    obj[iter] = loss+lambda*prod( sapply(U, function(x) sum(abs(x))) )
    
    #print(iter)
  }
  obj = obj[1:iter]
  
  rtn = list(A = A, U = U, G = G, obj = obj, iter = iter, tol = tolerance)
  return(rtn)
}

###### ADMM to compute U_k ######
Uk_ADMM = function(y, X, B0, tau, lambda,  gamma=1, tol = 1e-4, iter_max = 1000){
  ###input response y and covariate X
  ###B0: initial
  ###gamma: ADMM-gamma
  
  ###initials###
  B = B0
  P = B0
  dim1 = nrow(B)
  rank1 = ncol(B)
  beta_b = as.vector(B0)
  beta = as.vector(B0)
  E = matrix(0,nrow = dim1,ncol = rank1)
  eta = matrix(0,nrow = dim1*rank1, ncol = 1)
  u = matrix(0, nrow = length(y), ncol = 1)
  r = matrix(0,nrow = length(y), ncol = 1)
  
  tolerance1 = 1
  iter1 = 0
  while(tolerance1>=tol & iter1<=iter_max){
    beta = beta_b+eta/gamma
    beta = pmax(beta-lambda/gamma,0)-pmax(-beta-lambda/gamma,0)
    
    r = u/gamma+y-X%*%beta_b
    r = pmax(r-tau/gamma,0)-pmax(-r+(tau-1)/gamma,0)
    
    ###beta_b###
    tolerance2 = 1; iter2 = 0; 
    beta_b0 = beta_b
    while (tolerance2>=1e-3 & iter2<=50) {
      beta_bb = t(X)%*%(u/gamma+y-r)-eta/gamma+beta+as.vector(P)-as.vector(E)/gamma
      inv = solve(t(X)%*%X+2*diag(dim1*rank1))
      beta_bb = inv%*%beta_bb
      Bb = matrix(beta_bb,nrow = dim1, ncol = rank1)
      svd = svd(Bb+E/gamma,nu = dim1,nv = rank1)
      V1 = svd$u
      V2 = svd$v
      P = V1%*%diag(1,nrow = dim1,ncol = rank1)%*%t(V2)
      E = E+gamma*(Bb-P) 
      tolerance2 =  norm(beta_bb-beta_b0,type = "F")
      beta_b0 = beta_bb
      iter2 = iter2+1
    }
    beta_b = beta_bb
    
    u = u+gamma*(y-X%*%beta_b-r)
    eta = eta+gamma*(beta_b-beta)
    
    B = matrix(beta,nrow = dim1,ncol = rank1)
    tolerance1 = norm(B-B0,type = "F")
    B0 = B
    iter1 = iter1+1
  }
  #print(paste("iter1", iter1))
  #print(paste("tole1",tolerance1))
  rtn = list(B = B, tolerance = tolerance1)
  return(rtn)
}

###########################
### Example 2 sparse
###########################
library(rTensor)
library(quantreg)

tau = 0.5
dim = c(5,5,5)
rank = c(3,3,3)
n = 500;
sigma_g = 1
sigma = 1
order = 3

real = 1;p1 = dim[1];p2 = dim[2];p3 = dim[3];
A <- array(0.0, dim = c( p1, p2, p3))
for (i3 in 1:p3) {
  A[1:p1, 1, i3] <- real
}
A <- as.tensor(A);

num <- 200;#重复次数

EE_100 <- c();

for (nm in 1:num) {
  cat("正在运行第",nm,"次重复实验。\n");
  dat = DGP(n, coef = A, sigma = sigma, tau = tau, dim = dim);
  
  y = dat$y;
  X = dat$X;
  
  X_vec = list()
  for(k in 1:order){
    X_vec[[k]] = X_unfold(X,mode = k)
  }
  
  dat_val = DGP(n, coef = A, sigma = sigma, tau = tau, dim = dim)
  y_val = dat_val$y
  X_val = dat_val$X
  X_val_vec = X_unfold(X_val,mode = 1)
  
  ##########  initial  ###############
  A0 = rq.fit.lasso(X_vec[[1]],y,lambda = 0.0001,eps = 0.1);
  a0 = array(A0$coefficients[-1],dim=dim)
  a0 = as.tensor(a0)
  
  
  ########estimate  ################
  Ahat = Highdim_AQR(y, X, X_vec, a0, rank = rank, tau = tau, lambda = 0.02,
                     penalty = c(TRUE,TRUE,TRUE),tol = 4e-03, iter_max = 60)
  
  A_es <- Ahat$A;
  
  EE <- sqrt(sum((A_es@data - A@data)^2));
  EE_100[nm] <- EE;
}




