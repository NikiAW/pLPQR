# # ===========================================================================================
# # R code for the simulations of
# # ``Unravelling Determinants of Global Life Expectancy: A Joint Quantile Approach for Heterogeneous Longitudinal Studies"
# # ===================================================================

rm(list=ls())
# install.packages("copula")
library("copula")
library(pLPQR)

copula <- "gauss_wi"; betatype <- "large-hetero"
bf <- function(xx){return(c(1,qnorm(xx)))}
bderivf <- function(xx){return(c(0,1/dnorm(xx)))}
K <- 2

if(betatype=="large"){
  theta0 <- theta <- matrix(c(0,2,1,0,3,
                              1,rep(0,4)),ncol=K)
}else if(betatype=="large-hetero"){
  theta0 <- theta <- matrix(c(0,2,1,0,3,
                              1,0.5,0,1,0),ncol=K)
}

p <- dim(theta0)[1]
nsub <- 200   # number of subjects  800
m <- 3        # number of time periods
nk <- m*rep(1,nsub)   # vector, number of time periods for the nsub subjects
n <- sum(nk)          # number of total observations
index <- rep(1:m,nsub)

seed <- 123
set.seed(seed)
Rho <- 0.8
library(MASS)
x <- matrix(1,nrow=n,ncol=1)
for(j in 1:(p-1)){
  x <- cbind(x,runif(n,min=(j-1)/5,max=1+(j-1)/5))
}

qe <- y <- NULL
set.seed(seed)
# generate from Gaussian copula
if(copula=="gauss_cs"){
  if(length(unique(nk))==1){
    m <- unique(nk)
    R <- (1-Rho)*diag(m)+Rho*matrix(1,m,m)
    qe <- c(t(rCopula(nsub, normalCopula(param = P2p(R), dim = m, dispstr = "un"))))
  }else{
    for(ii in 1:nsub){
      R <- (1-Rho)*diag(nk[ii])+Rho*matrix(1,nk[ii],nk[ii])
      qe <- c(qe, pnorm(mvrnorm(1, rep(0,nk[ii]), R)))
    }
  }
}else if(copula=="gauss_ar"){
  if(length(unique(nk))==1){
    m <- unique(nk)
    R <- matrix(NA,m,m)
    for(i in 1:m){
      for(j in 1:m){
        R[i,j] <- Rho^abs(i-j)
      }
    }
    qe <- c(t(pnorm(mvrnorm(nsub, rep(0,m), R))))
  }else{
    for(ii in 1:nsub){
      R <- matrix(NA,nk[ii],nk[ii])
      for(i in 1:nk[ii]){
        for(j in 1:nk[ii]){
          R[i,j] <- Rho^abs(i-j)
        }
      }
      qe <- c(qe, pnorm(mvrnorm(1, rep(0,nk[ii]), R)))
    }
  }
}else if(copula=="gauss_wi"){
  if(length(unique(nk))==1){
    m <- unique(nk)
    R <- diag(m)
    qe <- c(t(pnorm(mvrnorm(nsub, rep(0,m), R))))
  }else{
    for(ii in 1:nsub){
      qe <- c(qe, pnorm(mvrnorm(1, rep(0,nk[ii]), diag(nk[ii]))))
    }
  }
}else if(copula=="gumbel"){
  theta_tau <- 1 / (1 - 0.75)
  for(ii in 1:nsub){
    gumbel_cop <- gumbelCopula(theta_tau, dim = nk[ii])
    qe <- c(qe, rCopula(1, gumbel_cop))
  }
}
beta0 <- lapply(qe,function(xx){theta %*% bf(xx)})  # #c(2,-1,1)

for(ii in 1:n){
  y <- c(y,x[ii,]%*%beta0[[ii]])
}

### CQR initials
library(qrcm)
set.seed(seed)
IQR <- iqr(y~ -1 + x, formula.p = ~ I(qnorm(p)))
initials <- c(IQR$coefficients)

# # weights
tau0 <- seq(0.01,0.99,0.01)
wt <- rep(1,length(tau0))
nt <- length(tau0)
btau=sapply(tau0,bf)
bderivtau <- sapply(tau0,bderivf)

# # # # CQR using qif
CS.CQR <- cqr(type="cs", x,y, nk,rowindex=c(0,cumsum(nk)), tau0, nsub,nt, gammaint=initials, btau=sapply(tau0,bf), bderiv=sapply(tau0,bderivf), K=length(bf(0)), wt=rep(1,length(tau0)))
AR.CQR <- cqr(type="ar", x,y, nk,rowindex=c(0,cumsum(nk)), tau0, nsub,nt, gammaint=initials, btau=sapply(tau0,bf), bderiv=sapply(tau0,bderivf), K=length(bf(0)), wt=rep(1,length(tau0)))
WI.CQR <- cqr(type="wi", x,y, nk,rowindex=c(0,cumsum(nk)), tau0, nsub,nt, gammaint=initials, btau=sapply(tau0,bf), bderiv=sapply(tau0,bderivf), K=length(bf(0)), wt=rep(1,length(tau0)))

# # QIF with adlasso
CS.adlasso <- CV_cqr_adlasso(type="cs", lambda =seq(1,0.1,-0.1), x,y, nsub,nk, bf,bderivf, gammaint=c(CS.CQR$gamma), v=1/abs(c(CS.CQR$gamma)), tau = seq(0.01, 0.99, 0.01),rho=0.01)
AR.adlasso <- CV_cqr_adlasso(type="ar", lambda =seq(1,0.1,-0.1), x,y, nsub,nk, bf,bderivf, gammaint=c(AR.CQR$gamma), v=1/abs(c(AR.CQR$gamma)), tau = seq(0.01, 0.99, 0.01),rho=0.01)
WI.adlasso <- CV_cqr_adlasso(type="wi", lambda =seq(1,0.1,-0.1), x,y, nsub,nk, bf,bderivf, gammaint=c(WI.CQR$gamma), v=1/abs(c(WI.CQR$gamma)), tau = seq(0.01, 0.99, 0.01),rho=0.01)

# LASSO
CS.lasso <- CV_cqr_adlasso(type="cs", lambda =seq(1,0.1,-0.1), x,y, nsub,nk, bf,bderivf, gammaint=c(CS.CQR$gamma), v=rep(1,length(initials)), tau = seq(0.01, 0.99, 0.01),rho=0.01)
AR.lasso <- CV_cqr_adlasso(type="ar", lambda =seq(1,0.1,-0.1), x,y, nsub,nk, bf,bderivf, gammaint=c(AR.CQR$gamma), v=rep(1,length(initials)), tau = seq(0.01, 0.99, 0.01),rho=0.01)
WI.lasso <- CV_cqr_adlasso(type="wi", lambda =seq(1,0.1,-0.1), x,y, nsub,nk, bf,bderivf, gammaint=c(WI.CQR$gamma), v=rep(1,length(initials)), tau = seq(0.01, 0.99, 0.01),rho=0.01)

