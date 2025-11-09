#' Cross-validation for Penalized Longitudinal Parametric Quantile Regression with Adaptive LASSO
#'
#' Performs cross-validation to select the optimal tuning parameter for penalized longitudinal
#' parametric quantile regression with adaptive LASSO penalty. This function evaluates
#' a sequence of lambda values and selects the one that minimizes the quadratic inference function
#' Bayesian Information Criterion (QIFBIC).
#'
#' @param x Numeric matrix of covariates (n x p), with rows sorted by subject and time.
#' @param y Numeric vector of responses (length n).
#' @param nsub Integer, number of subjects.
#' @param nk Integer vector, number of observations for each subject.
#' @param bf Function that computes basis functions, takes tau and returns numeric vector.
#' @param bderivf Function that computes derivatives of basis functions.
#' @param gammaint Numeric vector of initial coefficient estimates.
#' @param v Vector of adaptive weights.
#' @param lambda Numeric vector of tuning parameters. Default is seq(1, 0.1, -0.1).
#' @param type Character, working correlation structure: "cs" (compound symmetry),
#'   "ar" (autoregressive), or "wi" (working independence). Default is "cs".
#' @param tau Numeric vector of quantile levels. Default is seq(0.01, 0.99, 0.01).
#' @param nt Integer, number of tau levels. Default is length(tau).
#' @param rho Numeric, penalty parameter. Default is 0.01.
#'
#' @return A list containing:
#' \item{r}{Optimal model results}
#' \item{gamma}{Estimated coefficients}
#' \item{QIFBIC}{Optimal BIC value}
#' \item{lambda}{Optimal lambda value}
#' \item{rlist}{List of results for all lambda values}
#'
#' @references Yu, Z., Yu, K.M., Ni, Y.X., & Tian, M.Z. (2025). Unravelling Determinants of Global Life Expectancy: A Joint Quantile Approach for Heterogeneous Longitudinal Studies.
#'
#' @examples
#' \donttest{
#' # library(qrcm)
#' nsub <- 200
#' m <- 3
#' nk <- rep(m, nsub)
#' n <- sum(nk)
#' Rho <- 0.8
#' R <- (1-Rho)*diag(m)+Rho*matrix(1,m,m)
#' tau0 <- seq(0.01,0.99,0.01)
#'
#' # Define basis functions
#' bf <- function(xx){return(c(1,qnorm(xx)))}
#' bderivf <- function(xx){return(c(0,1/dnorm(xx)))}
#' K <- 2
#'
#' # Generate example data
#' theta0 <- theta <- matrix(c(0,2,1,0,3,1,rep(0,4)),ncol=K)
#' p <- dim(theta0)[1]
#' x <- matrix(1,nrow=n,ncol=1)
#' for(j in 1:(p-1)){
#'   x <- cbind(x,runif(n,min=(j-1)/5,max=1+(j-1)/5))
#' }
#' generate_correlated_unif_base <- function(n, dim, rho) {
#'   result <- matrix(0, n, dim)
#'   result[,1] <- rnorm(n)
#'   for(j in 2:dim) {
#'     result[,j] <- rho * result[,1] + sqrt(1-rho^2) * rnorm(n)
#'   }
#'   pnorm(result)
#' }
#' qe <- c(t(generate_correlated_unif_base(nsub, m, Rho)))
#' beta0 <- lapply(qe,function(xx){theta %*% bf(xx)})
#' y <- NULL
#' for(ii in 1:n){
#'   y <- c(y,x[ii,]%*%beta0[[ii]])
#' }
#'
#' ## CQR initials
#' # IQR <- iqr(y~ -1 + x, formula.p = ~ I(qnorm(p)))
#' # initials <- c(IQR$coefficients)
#' # CS.CQR <- cqr(type="cs", x,y, nk,rowindex=c(0,cumsum(nk)), tau0, nsub,nt=length(tau0),
#' # gammaint=initials, btau=sapply(tau0,bf), bderiv=sapply(tau0,bderivf),
#' # K=length(bf(0)), wt=rep(1,length(tau0)))
#'
#' ## Perform cross-validation
#' # CS.adlasso <- CV_cqr_adlasso(type="cs", lambda =seq(1,0.1,-0.1), x,y, nsub,nk, bf,bderivf,
#' # gammaint=c(CS.CQR$gamma), v=1/abs(c(CS.CQR$gamma)), tau0,rho=0.01)
#' }
#'
#' @useDynLib pLPQR, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @export
CV_cqr_adlasso <- function(x,  # the covariate matrix
                           y,  # the corresponding response vector
                           nsub, # the number of subjects
                           nk,   # the number of observations for each subject
                           bf,   # the known functions b(tau)
                           bderivf,  # the derivative of the known functions b(tau)
                           gammaint,  # initial estimates of the unknown coefficient matrix
                           v=1/abs(c(gammaint)),  # Vector of adaptive weights
                           lambda = seq(1, 0.1, -0.1),  # the tuning parameter in pLPQR
                           type = "cs",  # working correlation matrix type
                           tau = seq(0.01, 0.99, 0.01), # quantile levels
                           nt = length(tau),  # the number of tau's
                           rho = 0.01  # a constant penalty parameter
) {
  K <- length(bf(0.5))

  if (length(gammaint) != ncol(x) * K) {
    stop(paste("gammaint should have length", ncol(x) * K, "but has length", length(gammaint)))
  }

  # Calculate rowindex
  rowindex <- c(0, cumsum(nk))  # the starting row index minus 1 for each country

  # Compute btau and bderiv matrices
  btau <- sapply(tau, bf)
  bderiv <- sapply(tau, bderivf)
  K <- length(bf(0.5))  # Use 0.5 as a sample tau value
  wt <- rep(1, length(tau))

  # Call the C++ function
  result <- cv_cqr_adlasso0(type, lambda, x, y, nk, rowindex, tau, nsub, nt,
                           gammaint, v=1.0 /abs(gammaint), rho, btau, bderiv, K, wt)
  return(result)
}
