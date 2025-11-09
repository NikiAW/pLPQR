// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::plugins(cpp11)]]

#define ARMA_WARN_LEVEL 1

#include <RcppArmadillo.h>
#include <stdio.h>
#include <cmath>
#include <Rmath.h>
#include <omp.h>
#include <unordered_map>
#include <chrono>

using namespace Rcpp;
using namespace arma;

arma::mat robust_inv(const arma::mat& A, int max_reg_attempts = 3) {
  arma::mat result;
  arma::mat I = arma::eye(A.n_rows, A.n_cols);

  bool solve_success = arma::solve(result, A, I);

  if(solve_success) {
    return result;
  }

  double lambda = 1e-8 * arma::trace(A) / A.n_rows;

  for(int attempt = 0; attempt < max_reg_attempts; ++attempt) {
    arma::mat A_reg = A;
    A_reg.diag() += lambda;

    solve_success = arma::solve(result, A_reg, I);

    if(solve_success) {
      return result;
    }

    lambda *= 10;
  }

  arma::mat A_final = A;
  A_final.diag() += lambda;

  try {
    result = arma::inv(A_final);
  } catch(...) {
    result = I;
  }

  return result;
}


inline arma::vec cpp_pnorm_faster(const arma::vec& x) {
  return arma::normcdf(x);
}

inline arma::vec cpp_dnorm_faster(const arma::vec& x) {
  return arma::normpdf(x);
}


arma::mat compute_btau_cpp(const Function& bf, int K) {
  // Optimized implementation: minimize R function calls
  arma::vec p10 = arma::linspace(1.0/1024.0, 1023.0/1024.0, 1023);
  arma::mat btau(K, 1023);

  // Pre-allocate memory to avoid repeated allocation
  NumericVector bf_result(K);

  for(int i = 0; i < 1023; i++) {
    bf_result = bf(p10(i)); // Reuse the same vector
    for(int j = 0; j < K; j++) {
      btau(j, i) = bf_result(j);
    }
  }

  return btau;
}


//' Bisection Method for Probability Calculation
//'
//' This function implements an efficient bisection algorithm to compute probabilities
//' for quantile regression models. It performs a two-phase search:
//' a coarse search for rapid convergence followed by a fine search for precision.
//'
//' @param theta Numeric matrix of estimated parameters (p x K), where p is the number
//'   of covariates and K is the number of basis functions.
//' @param y Numeric vector of response values (length n).
//' @param X Numeric design matrix (n x p) containing covariates.
//' @param bf R function that computes the K-dimensional basis functions of tau.
//'   Should take a single numeric argument (tau) and return a numeric vector.
//' @param n_it Integer specifying the maximum number of iterations for the bisection
//'   algorithm. Default is 20.
//'
//' @return Numeric vector of probabilities (length n) in the range [0, 1] representing
//'   the estimated quantile levels for each observation.
//'
//' @details
//' The algorithm works as follows:
//' \itemize{
//'   \item \strong{Phase 1 (Coarse search, iterations 2-10)}: Uses large steps for
//'         rapid convergence to the approximate solution.
//'   \item \strong{Phase 2 (Fine search, iterations 11-n_it)}: Uses progressively
//'         smaller steps for high precision, with caching of basis function evaluations.
//' }
//'
//' The function employs caching to avoid repeated evaluations of the basis functions
//' at the same tau values, improving computational efficiency.
//'
//' @examples
//' \donttest{
//' # Example usage
//' set.seed(123)
//' n <- 100
//' p <- 3
//' K <- 2
//'
//' # Generate sample data
//' theta <- matrix(rnorm(p * K), nrow = p, ncol = K)
//' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
//' y <- rnorm(n)
//'
//' # Define basis function
//' bf <- function(tau) {
//'   return(c(1, qnorm(tau)))
//' }
//'
//' # Compute probabilities
//' probabilities <- p_bisec_cpp(theta, y, X, bf, n_it = 20)
//' head(probabilities)
//' }
//'
//' @references
//' \itemize{
//'   \item Sottile, G., and Frumento, P. (2023). Parametric estimation of non-crossing quantile functions. Statistical Modelling, 23 (2), 173-195.
//' }
//'
//' @export
// [[Rcpp::export]]
arma::vec p_bisec_cpp(const arma::mat& theta,     // the estimated parameter matrix
                      const arma::vec& y,          // the response
                      const arma::mat& X,          // the design matrix
                      const Function& bf,          // the K-dimensional vector composed of known functions of tau
                      int n_it = 20) {             // the maximum number of iterations, default 20

  const int n = y.n_elem;
  const int K = theta.n_cols;
  arma::vec m = arma::ones<arma::vec>(n) * 512; // Start at middle position (1-based)
  arma::vec p10 = arma::linspace(1.0/1024.0, 1023.0/1024.0, 1023);

  // Efficient computation of btau using the dedicated function
  arma::mat btau = compute_btau_cpp(bf, K);
  arma::mat betatau = theta * btau;
  betatau = betatau.t();

  // Phase 1: Coarse search (iterations 2-10) - use integer indexing for speed
  for (int i = 2; i <= 10; i++) {
    arma::vec delta_m(n);
    const int step_size = 1 << (10 - i); // Pre-compute step size

    for(int j = 0; j < n; j++) {
      const int idx = static_cast<int>(m(j)) - 1;
      delta_m(j) = y(j) - arma::sum(X.row(j) % betatau.row(idx));
    }

    m = m + arma::sign(delta_m) * step_size;
  }

  for(int j = 0; j < n; j++) {
    int idx = static_cast<int>(m(j)) - 1;
    m(j) = p10(idx);
  }

  // Phase 2: Fine search (iterations 11-n_it) - use smaller steps
  std::unordered_map<double, arma::vec> bf_cache;

  for (int i = 11; i <= n_it; i++) {
    arma::vec delta_m(n);
    const double step_size = 1.0 / (1 << i); // Pre-compute step size

    for(int j = 0; j < n; j++) {
      const double m_val = m(j);

      auto cache_it = bf_cache.find(m_val);
      arma::vec bf_vec;

      if(cache_it != bf_cache.end()) {
        bf_vec = cache_it->second;
      } else {
        NumericVector bf_result = bf(m_val);
        bf_vec = arma::vec(bf_result.begin(), bf_result.length());
        bf_cache[m_val] = bf_vec;
      }

      delta_m(j) = y(j) - arma::as_scalar(X.row(j) * theta * bf_vec);
    }

    m = m + arma::sign(delta_m) * step_size;
  }

  // Handle boundary conditions efficiently using vectorized operations
  const double step_factor = 1.0 / (1 << n_it);
  const double lower_bound = step_factor;
  const double upper_bound = 1.0 - step_factor;

  // Set exact boundary values where needed using manual loop for compatibility
  for(int j = 0; j < n; j++) {
    if(m(j) == lower_bound) {
      m(j) = 0.0;
    }else if(m(j) == upper_bound) {
      m(j) = 1.0;
    }
  }

  return m;
}


//' Simulate longitudinal data by a specified correlation matrix
//'
//' This function generates simulated longitudinal data using a copula-based approach
//' for composite quantile regression models. It creates correlated response data
//' with specified correlation structure across time points within subjects.
//'
//' @param thetahat Numeric matrix of estimated parameters (p x K), where p is the number
//'   of covariates and K is the number of basis functions.
//' @param X Numeric design matrix (n x p) containing covariates for all observations.
//' @param nsub Integer specifying the number of subjects in the longitudinal data.
//' @param bf R function that computes the K-dimensional basis functions of tau.
//'   Should take a single numeric argument (tau) and return a numeric vector.
//' @param index Integer vector indicating the time ordering within subjects.
//' @param id Integer vector identifying subject membership for each observation.
//' @param spcor Numeric matrix (m x m) specifying the spatial/temporal correlation
//'   structure between time points, where m is the number of time points per subject.
//'
//' @return A list containing the following components:
//' \describe{
//'   \item{y}{Numeric vector of simulated response values (length n).}
//'   \item{x}{Numeric design matrix, same as input X.}
//'   \item{tau_ij}{Numeric vector of quantile levels (length n) used for data generation.}
//'   \item{nsub}{Integer, number of subjects.}
//'   \item{index}{Integer vector, time ordering within subjects.}
//'   \item{id}{Integer vector, subject identifiers.}
//' }
//'
//' @details
//' The function implements the following data generation process:
//' \enumerate{
//'   \item \strong{Copula Generation}: Uses Cholesky decomposition of the correlation
//'         matrix to generate correlated normal samples, which are transformed to
//'         uniform [0,1] variables via the normal CDF.
//'   \item \strong{Quantile Levels}: The uniform variables serve as quantile levels
//'         (tau_ij) that introduce within-subject correlation.
//'   \item \strong{Response Generation}: For each observation, computes the basis
//'         functions at the generated quantile level, applies the parameter matrix
//'         to get time-varying coefficients, and generates the response as a linear
//'         combination of covariates and coefficients.
//' }
//'
//' This approach ensures that the simulated data maintains the specified correlation
//' structure across time points within each subject, making it suitable for
//' longitudinal data analysis.
//'
//' @examples
//' \donttest{
//' # Example: Simulate longitudinal data with 50 subjects and 3 time points
//' set.seed(123)
//' nsub <- 50
//' m <- 3  # time points per subject
//' n <- nsub * m
//' p <- 2  # number of covariates
//' K <- 2  # number of basis functions
//'
//' # Parameter matrix
//' thetahat <- matrix(c(1.0, 0.5, 0.3, 0.8), nrow = p, ncol = K)
//'
//' # Design matrix
//' X <- cbind(1, matrix(rnorm(n * (p-1)), nrow = n, ncol = p-1))
//'
//' # Basis function
//' bf <- function(tau) {
//'   return(c(1, qnorm(tau)))
//' }
//'
//' # Correlation structure (compound symmetry) using base R
//' rho <- 0.6
//' spcor <- (1 - rho) * diag(m) + rho * matrix(1, m, m)
//'
//' # Subject identifiers and time indices
//' id <- rep(1:nsub, each = m)
//' index <- rep(1:m, nsub)
//'
//' # Generate data
//' sim_data <- simdata_cpp(thetahat, X, nsub, bf, index, id, spcor)
//'
//' # Check the structure
//' str(sim_data)
//' head(sim_data$y)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List simdata_cpp(const arma::mat& thetahat, // the estimated parameter matrix
                       const arma::mat& X, // the design matrix
                       int nsub,  // the number of subjects
                       const Function& bf, // the K-dimensional vector composed of known functions of tau
                       const arma::vec& index,
                       const arma::vec& id,
                       const arma::mat& spcor
) {
  int n = X.n_rows;
  int p = thetahat.n_rows;
  int K = thetahat.n_cols;
  int m = spcor.n_cols;

  // 1. Copula
  arma::mat chol_spcor = arma::chol(spcor);
  arma::mat norm_samples = arma::randn(nsub, m) * chol_spcor.t();
  arma::mat tau_ij_mat = arma::normcdf(norm_samples);
  arma::vec tau_ij = arma::vectorise(tau_ij_mat.t());

  arma::vec ysim(n, arma::fill::zeros);
  arma::mat beta(p, n, arma::fill::zeros);
  NumericVector bf_result;
  arma::vec beta_col, bf_vec;

  for (int ii = 0; ii < n; ii++){
    bf_result = bf(tau_ij(ii));
    bf_vec = arma::vec(bf_result.begin(), K);
    beta_col = thetahat * bf_vec;
    beta.col(ii) = beta_col;
    ysim(ii) = arma::dot(X.row(ii), beta_col);
  }

  return Rcpp::List::create(
    Rcpp::Named("y") = ysim,
    Rcpp::Named("x") = X,
    Rcpp::Named("tau_ij") = tau_ij,
    Rcpp::Named("nsub") = nsub,
    Rcpp::Named("index") = index,
    Rcpp::Named("id") = id
  );
}

arma::vec soft(arma::vec x, arma::vec a){
  return (abs(x)-a)%sign(x)%(abs(x)>a);
}




//' Composite Quantile Regression with Parametric Coefficient Modeling
//'
//' Fits a composite quantile regression model for longitudinal data using parametric
//' modeling of quantile coefficients with different working correlation structures.
//'
//' @param x Numeric matrix of covariates (n x p), with rows sorted by subject and time.
//' @param y Numeric vector of responses (length n).
//' @param nk Integer vector, number of observations for each subject.
//' @param rowindex Integer vector indicating the starting row index (minus 1) for each subject.
//'   Should have length (nsub + 1) with rowindex[1] = 0 and rowindex[nsub+1] = n.
//' @param tau0 Numeric vector of quantile levels used for composite quantile regression.
//' @param nsub Integer, number of subjects.
//' @param nt Integer, number of tau levels.
//' @param gammaint Numeric vector of initial coefficient estimates (length p*K).
//' @param btau Numeric matrix of basis function values at tau0 (K x nt).
//' @param bderiv Numeric matrix of basis function derivative values at tau0 (K x nt).
//' @param K Integer, number of basis functions.
//' @param type Character, working correlation structure: "cs" (compound symmetry),
//'   "ar" (autoregressive), or "wi" (working independence).
//' @param wt Numeric vector of weights for different quantile levels (length nt).
//'
//' @return A list containing:
//' \item{gamma}{Estimated coefficient vector (length p*K)}
//' \item{cov}{Estimated covariance matrix of coefficients (p*K x p*K)}
//' \item{Dev}{Deviance value}
//' \item{AIC}{Akaike Information Criterion}
//' \item{BIC}{Bayesian Information Criterion}
//'
//' @details
//' This function implements composite quantile regression for longitudinal data
//' using parametric modeling of quantile coefficients. The method allows for
//' different working correlation structures to account for within-subject dependence:
//' \itemize{
//'  \item \code{"cs"}: Compound symmetry (exchangeable) correlation structure
//'   \item \code{"ar"}: Autoregressive structure of order 1
//'  \item \code{"wi"}: Working independence assumption
//' }
//'
//' The algorithm uses a quadratic inference function (QIF) approach and iteratively
//' updates coefficient estimates until convergence.
//'
//' @references Zhao, W., Lian, H., and Song, X. (2017). Composite quantile regression for correlated data.
//' Computational Statistics & Data Analysis, 109:15–33.
//' Fu, L., & Wang, Y. G. (2012). Quantile regression for longitudinal data
//' with a working correlation model. Computational Statistics & Data Analysis, 56(8), 2526-2538.
//' @export
// [[Rcpp::export]]
List cqr(arma::mat x,  // the covariate matrix, with rows sorted primarily by country and secondarily by year
         arma::vec y,  // the corresponding response vector
         arma::vec nk, // a vector indicating the number of observations for each subject
         arma::vec rowindex, // the starting row index minus 1 for each country
         arma::vec tau0,     // a vector of quantile levels that used for the composite quantile regression
         int nsub,  // number of subjects
         int nt,  // number of tau's
         arma::vec gammaint, // initial estimates of the unknown coefficient matrix
         arma::mat btau,      // the value of the known functions b(tau) at tau0
         arma::mat bderiv,  // the value of the derivative of the known functions b(tau)
         int K,        // the length of the known functions b(tau)
         String type,    // working correlation matrix with "cs" indicating the normal copula with an exchangeable correlation matrix, "ar" indicating the normal copula with an autoregressive structure of order 1, and "wi" indicating the working independence
         arma::vec wt){

  int p = x.n_cols, maxit=500, iteration;
  double Q0=1e6, Q = 1e6, betadiff = 1.0, w = 1.0, tau=0.1, tol=1e-5;

  arma::mat Oma(p*K,p*K, arma::fill::eye), Oma0(p*K,p*K, arma::fill::eye), gam(p,K,arma::fill::zeros);
  arma::vec ganew = gammaint, ga = gammaint, v = 1 / abs(gammaint), b(K), xi =ganew;
  arma::mat xx(nk(1),p,arma::fill::zeros), dotmu=xx.t(), D(nk(1),nk(1)), Gam(nk(1),nk(1), arma::fill::eye);
  arma::vec diff = ganew - xi, xnorm(nk(1)), r1(nk(1)), r2(nk(1));
  arma::mat matrix1(nk(1),nk(1),arma::fill::ones), m1(nk(1),nk(1)), m2(nk(1),nk(1)), eyenk(nk(1),nk(1),arma::fill::eye), ma(nk(1),nk(1));
  arma::vec r(nk(0)), mu, yy(nk(0)), arqif1dev(p*K,arma::fill::zeros);
  arma::mat eyepk(p*K,p*K, arma::fill::eye), arqif2dev(p*K,p*K, arma::fill::zeros), invarqif2dev(p*K,p*K, arma::fill::zeros);
  bool done;
  int diter=0;
  if(type == "cs"){
    arma::vec arsumg(p*K*2,arma::fill::zeros);
    arma::mat arsumc(p*K*2,p*K*2,arma::fill::zeros), arcinv(p*K*2,p*K*2,arma::fill::zeros), rr(p*K,p*K*2,arma::fill::zeros), rrt(p*K*2,p*K,arma::fill::zeros), arsumgfirstdev(p*K*2,p*K,arma::fill::zeros), di(p*K*2,p*K,arma::fill::zeros); //, arsumc(p*K,p*K,arma::fill::zeros), arsumgfirstdev2(p*K,p*K,arma::fill::zeros);
    arma::vec gi(p*K*2,arma::fill::zeros);

    iteration = 0;
    while((iteration==0)||((betadiff>tol)&&(iteration<maxit))){
      iteration=iteration+1;
      ga = ganew;
      Q0 = Q;
      Oma0 = Oma;
      arsumg.fill(0.0);
      arsumc.fill(0);
      arsumgfirstdev.fill(0);

      for(int i=0; i<nsub; i++){
        for(int j=0; j<nt; j++){
          tau = tau0(j);
          xx = x.submat(rowindex(i),0,rowindex(i+1)-1,p-1);
          yy = y.subvec(rowindex(i),rowindex(i+1)-1);
          b = btau.col(j);
          gam  = reshape(ga, p, K);
          mu = xx * gam * b;
          dotmu = xx.t();

          r = sqrt(diagvec(kron(b.t(),xx)*Oma*kron(b,xx.t())));
          xnorm = sqrt(nsub) * (yy-mu) / r;
          r1 = cpp_pnorm_faster(xnorm) + tau - 1.0;
          r2 = cpp_dnorm_faster(xnorm) / r;
          D = sqrt(nsub) * diagmat(vectorise(r2));

          Gam = diagmat(vectorise(1.0/(xx * gam * bderiv.col(j))));

          m1 = matrix1 - eyenk;
          gi = kron(b, wt(j) * join_cols(dotmu * Gam * r1, dotmu * Gam * m1 * r1))/nsub;
          di = - kron(b*b.t(), wt(j)*join_cols(dotmu * Gam * D * dotmu.t(), dotmu * Gam * m1 * D * dotmu.t()))/nsub;

          arsumg = arsumg + gi;
          arsumc = arsumc + gi * gi.t();
          arsumgfirstdev = arsumgfirstdev + di;
        }
      }

      arsumc = arsumc * nsub;
      done = arma::solve(rrt, arsumc, arsumgfirstdev);
      diter =0;
      while(!done){
        arsumc.diag() += 100 * arma::datum::eps;
        done = arma::solve(rrt, arsumc, arsumgfirstdev);
        if(diter>500) break;
        diter++;
      }
      rr = rrt.t();
      arqif1dev = rr * arsumg;

      arqif2dev = rr * arsumgfirstdev;

      ganew = ganew - w * arma::solve(arqif2dev, arqif1dev);

      Oma = pinv(arqif2dev);

      betadiff = accu(abs(ganew-ga));
      Q = nsub * as_scalar(arsumg.t() * arma::solve(arsumc, arsumg));

      if(abs(Q-Q0)<tol) break;
      w = w/2;
    }

  }else if(type == "ar"){
    arma::vec arsumg(p*K*3,arma::fill::zeros);
    arma::mat arsumc(p*K*3,p*K*3,arma::fill::zeros), arcinv(p*K*3,p*K*3,arma::fill::zeros), rr(p*K,p*K*3,arma::fill::zeros), rrt(p*K*3,p*K,arma::fill::zeros), arsumgfirstdev(p*K*3,p*K,arma::fill::zeros), di(p*K*3,p*K,arma::fill::zeros);
    arma::vec gi(p*K*3,arma::fill::zeros);


    iteration = 0;
    while((iteration==0)||((betadiff>tol)&&(iteration<maxit))){
      iteration=iteration+1;
      ga = ganew;
      Q0 = Q;
      Oma0 = Oma;
      arsumg.fill(0.0);
      arsumc.fill(0);
      arsumgfirstdev.fill(0);

      for(int i=0; i<nsub; i++){
        for(int j=0; j<nt; j++){
          tau = tau0(j);
          xx = x.submat(rowindex(i),0,rowindex(i+1)-1,p-1);
          yy = y.subvec(rowindex(i),rowindex(i+1)-1);
          b = btau.col(j);
          gam  = reshape(ga, p, K);
          mu = xx * gam * b;
          dotmu = xx.t();

          r = sqrt(diagvec(kron(b.t(),xx)*Oma*kron(b,xx.t())));
          xnorm = sqrt(nsub) * (yy-mu) / r;
          r1 = cpp_pnorm_faster(xnorm) + tau - 1.0;
          r2 = cpp_dnorm_faster(xnorm) / r;
          D = sqrt(nsub) * diagmat(vectorise(r2));

          Gam = diagmat(vectorise(1.0/(xx * gam * bderiv.col(j))));

          m1.zeros(); m2.zeros();
          m1(0,0)=1; m1(nk(i)-1,nk(i)-1)=1;
          m2(0,1)=1; m2(nk(i)-1,nk(i)-2)=1;
          for(int jj=1;jj<nk(i)-1;jj++){
            m2(jj,jj-1) = 1;
            m2(jj,jj+1) = 1;
          }
          gi = kron(b,wt(j) * join_cols(dotmu * Gam * r1,   dotmu * Gam * m1 * r1,   dotmu * Gam * m2 * r1))/nsub;
          di = - kron(b*b.t(), wt(j)*join_cols(dotmu * Gam * D * dotmu.t(), dotmu * Gam * m1 * D * dotmu.t(), dotmu * Gam * m2 * D * dotmu.t()))/nsub;

          arsumg = arsumg + gi;
          arsumc = arsumc + gi * gi.t();
          arsumgfirstdev = arsumgfirstdev + di;
        }
      }

      arsumc = arsumc * nsub;
      done = arma::solve(rrt, arsumc, arsumgfirstdev);
      diter =0;
      while(!done){
        arsumc.diag() += 100 * arma::datum::eps;
        done = arma::solve(rrt, arsumc, arsumgfirstdev);
        if(diter>500) break;
        diter++;
      }
      rr = rrt.t();

      arqif1dev = rr * arsumg;

      arqif2dev = rr * arsumgfirstdev;
      ganew = ganew - w * arma::solve(arqif2dev, arqif1dev);

      Oma = pinv(arqif2dev);
      betadiff = accu(abs(reshape(ganew-ga,p,K)*btau.col((nt-1)/2)))/(accu(abs(reshape(ga,p,K)*btau.col((nt-1)/2))));
      Q = nsub * as_scalar(arsumg.t() * arma::solve(arsumc, arsumg));

      if(abs(Q-Q0)<tol) break;
      w = w/2;
    }
  }else if(type == "wi"){
    arma::vec arsumg(p*K,arma::fill::zeros);
    arma::mat arsumc(p*K,p*K,arma::fill::zeros), arcinv(p*K,p*K,arma::fill::zeros), rr(p*K,p*K,arma::fill::zeros), rrt(p*K,p*K,arma::fill::zeros), arsumgfirstdev(p*K,p*K,arma::fill::zeros), di(p*K,p*K,arma::fill::zeros);
    arma::vec gi(p*K,arma::fill::zeros);

    iteration = 0;
    while((iteration==0)||((betadiff>tol)&&(iteration<maxit))){
      iteration=iteration+1;
      ga = ganew;
      Q0 = Q;
      Oma0 = Oma;
      arsumg.fill(0.0);
      arsumc.fill(0);
      arsumgfirstdev.fill(0);

      for(int i=0; i<nsub; i++){
        for(int j=0; j<nt; j++){
          tau = tau0(j);
          xx = x.submat(rowindex(i),0,rowindex(i+1)-1,p-1);
          yy = y.subvec(rowindex(i),rowindex(i+1)-1);
          b = btau.col(j);
          gam  = reshape(ga, p, K);
          mu = xx * gam * b;
          dotmu = xx.t();

          r = sqrt(diagvec(kron(b.t(),xx)*Oma*kron(b,xx.t())));
          xnorm = sqrt(nsub) * (yy-mu) / r;
          r1 = cpp_pnorm_faster(xnorm) + tau - 1.0;
          r2 = cpp_dnorm_faster(xnorm) / r;
          D = sqrt(nsub) * diagmat(vectorise(r2));

          Gam = diagmat(vectorise(1.0/(xx * gam * bderiv.col(j))));

          gi = kron(b, wt(j) * dotmu * Gam * r1)/nsub;
          di = - kron(b*b.t(), wt(j)*dotmu * Gam * D * dotmu.t())/nsub;

          arsumg = arsumg + gi;
          arsumc = arsumc + gi * gi.t();
          arsumgfirstdev = arsumgfirstdev + di;
        }
      }

      arsumc = arsumc * nsub;
      done = arma::solve(rrt, arsumc, arsumgfirstdev);
      diter =0;
      while(!done){
        arsumc.diag() += 100 * arma::datum::eps;
        done = arma::solve(rrt, arsumc, arsumgfirstdev);
        if(diter>500) break;
        diter++;
      }
      rr = rrt.t();
      arqif1dev = rr * arsumg;

      arqif2dev = rr * arsumgfirstdev;
      ganew = ganew - w * arma::solve(arqif2dev, arqif1dev);

      Oma = pinv(arqif2dev);
      betadiff = accu(abs(reshape(ganew-ga,p,K)*btau.col((nt-1)/2)))/(accu(abs(reshape(ga,p,K)*btau.col((nt-1)/2))));
      Q = nsub * as_scalar(arsumg.t() * arma::solve(arsumc, arsumg));

      if(abs(Q-Q0)<tol) break;
      w = w/2;
    }
  }

  List out;
  out["gamma"] = ganew;
  out["cov"] = Oma;
  out["Dev"] = log(Q);
  if(type == "cs"){
    out["AIC"] = Q + 2 * (p*K);
    out["BIC"] = Q + log(nsub) * (p*K);
  }else if(type == "ar"){
    out["AIC"] = Q + 2 * (p*K*2);
    out["BIC"] = Q + log(nsub) * (p*K*2);
  }else if(type == "wi"){
    out["AIC"] = Q + 2 * (p*K);
    out["BIC"] = Q + log(nsub) * (p*K);;
  }
  return out;

}


arma::vec compute_r_opt(const arma::vec& b, const arma::mat& xx, const arma::mat& Oma) {
  int p = xx.n_cols, k = b.n_elem;

  arma::mat M(p, p, arma::fill::zeros);

  for (int j = 0; j < k; j++) {
    for (int l = 0; l < k; l++) {
      M += b(j) * b(l) * Oma.submat(j*p, l*p, (j+1)*p-1, (l+1)*p-1);
    }
  }

  arma::mat xxM = xx * M;
  return arma::sqrt(arma::sum(xx % xxM, 1));
}





//' Penalized Longitudinal Parametric Quantile Regression with Adaptive LASSO
//'
//' Fits a penalized longitudinal parametric quantile regression model with adaptive LASSO
//' penalty for a specific tuning parameter value. This function implements composite quantile
//' regression for longitudinal data using parametric modeling of quantile coefficients with
//' adaptive LASSO regularization to achieve variable selection.
//'
//' @param x Numeric matrix of covariates (n x p), with rows sorted by subject and time.
//' @param y Numeric vector of responses (length n).
//' @param nk Integer vector, number of observations for each subject.
//' @param rowindex Integer vector indicating the starting row index (minus 1) for each subject.
//'   Should have length (nsub + 1) with rowindex[1] = 0 and rowindex[nsub+1] = n.
//' @param tau0 Numeric vector of quantile levels used for composite quantile regression.
//' @param nsub Integer, number of subjects.
//' @param nt Integer, number of tau levels.
//' @param gammaint Numeric vector of initial coefficient estimates (length p*K).
//' @param v Numeric vector of adaptive weights for the LASSO penalty (length p*K).
//' @param rho Numeric, penalty parameter for the ADMM algorithm.
//' @param lambda Numeric, tuning parameter controlling the strength of LASSO regularization.
//' @param btau Numeric matrix of basis function values at tau0 (K x nt).
//' @param bderiv Numeric matrix of basis function derivative values at tau0 (K x nt).
//' @param K Integer, number of basis functions.
//' @param type Character, working correlation structure: "cs" (compound symmetry),
//'   "ar" (autoregressive), or "wi" (working independence).
//' @param wt Numeric vector of weights for different quantile levels (length nt).
//'
//' @return A list containing:
//' \item{gamma}{Estimated coefficient vector after adaptive LASSO regularization (length p*K)}
//' \item{cov}{Estimated covariance matrix of coefficients (p*K x p*K)}
//' \item{Q}{Quadratic inference function value}
//' \item{Dev}{Deviance value including penalty term}
//' \item{AIC}{Akaike Information Criterion with degrees of freedom adjustment}
//' \item{BIC}{Bayesian Information Criterion with degrees of freedom adjustment}
//'
//' @details
//' This function implements penalized composite quantile regression for longitudinal data
//' using the adaptive LASSO penalty. The method combines parametric modeling of quantile coefficients
//' and generalized unbiased estimating equations method with a adaptive LASSO penalty for variable selection.
//'
//' The algorithm uses the Alternating Direction Method of Multipliers (ADMM) to solve
//' the optimization problem with adaptive LASSO penalty. Three correlation structures
//' are supported:
//' \itemize{
//'   \item \code{"cs"}: Compound symmetry (exchangeable) correlation structure
//'   \item \code{"ar"}: Autoregressive structure of order 1
//'   \item \code{"wi"}: Working independence assumption
//' }
//'
//' The adaptive weights (\code{v}) are typically set to the inverse of initial coefficient
//' estimates to achieve the oracle properties of adaptive LASSO.
//'
//' @references Yu, Z., Yu, K.M., Ni, Y.X., & Tian, M.Z. (2025). Unravelling Determinants of Global Life Expectancy: A Joint Quantile Approach for Heterogeneous Longitudinal Studies.
//' @export
// [[Rcpp::export]]
Rcpp::List cqr_adlasso(arma::mat x,  // the covariate matrix, with rows sorted primarily by country and secondarily by year
                       arma::vec y,  // the corresponding response vector
                       arma::vec nk, // a vector indicating the number of observations for each subject
                       arma::vec rowindex, // the starting row index minus 1 for each country
                       arma::vec tau0,     // a vector of quantile levels
                       int nsub,  // number of subjects
                       int nt,  // number of tau's
                       arma::vec gammaint, // initial estimates of the unknown coefficient matrix
                       arma::vec v, // Initial vector of the weights for the adaptive lasso
                       double rho, // a constant penalty parameter
                       double lambda,  // the tuning parameter in pLPQR
                       arma::mat btau,      // the value of the known functions b(tau) at tau0
                       arma::mat bderiv,  // the value of the derivative of the known functions b(tau)
                       int K,    // the length of the known functions b(tau)
                       String type,   // working correlation matrix with "cs" indicating the normal copula with an exchangeable correlation matrix, "ar" indicating the normal copula with an autoregressive structure of order 1, and "wi" indicating the working independence
                       arma::vec wt  // a vector of nonconstant weight measuring the varying amounts of information across quantiles
){
  int p = x.n_cols;
  int maxit = 500;
  double tol = 1e-5;
  double e_abs =1e-5, e_rel=1e-5;
  double epsilon = arma::datum::eps;

  arma::mat Oma(p*K, p*K, arma::fill::eye);
  arma::mat gam(p, K, arma::fill::zeros);
  arma::vec ganew = gammaint;
  arma::vec ga = gammaint;
  arma::vec xi = ganew;
  arma::vec xiold = ganew;
  arma::vec s = ganew;
  arma::vec delta = rho * (ganew - xi);
  arma::vec diff(ganew.n_elem);
  arma::vec b(K);
  arma::mat eyepk(p*K, p*K, arma::fill::eye);
  arma::mat arqif2dev(p*K, p*K, arma::fill::zeros);
  arma::vec arqif1dev(p*K, arma::fill::zeros);

  double Q = 1e6, Q0 = 1e6, betadiff = 1.0, w = 1.0;
  double primal_res = 0, dual_res = 0;
  bool done;
  int diter = 0;
  v = abs(v);

  if(type=="cs"){
    arma::mat rr(p*K, p*K*2, arma::fill::zeros);
    arma::mat rrt(p*K*2, p*K, arma::fill::zeros);
    arma::vec arsumg(p*K*2, arma::fill::zeros), gi(p*K*2, arma::fill::zeros);
    arma::mat arsumc(p*K*2, p*K*2, arma::fill::zeros), di(p*K*2, p*K, arma::fill::zeros);
    arma::mat arsumgfirstdev(p*K*2, p*K, arma::fill::zeros);

    std::unordered_map<int, arma::mat> m1_cache;
    for(int i = 0; i < nsub; i++){
      int current_nk = nk(i);
      if (m1_cache.find(current_nk) == m1_cache.end()){
        m1_cache[current_nk] = arma::ones(current_nk, current_nk) - arma::eye(current_nk, current_nk);
      }
    }

    for(int ii=0; ii < maxit; ++ii){

      w = 1.0;
      for(int iteration=0; iteration < maxit; ++iteration){
        ga = ganew;
        gam = reshape(ga, p, K);
        Q0 = Q;
        arsumg.zeros();
        arsumc.zeros();
        arsumgfirstdev.zeros();

        for(int i=0; i<nsub; i++){
          arma::mat xx = x.rows(arma::span(rowindex(i), rowindex(i+1)-1));
          arma::vec yy = y.rows(arma::span(rowindex(i), rowindex(i+1)-1));
          arma::mat dotmu = xx.t(), D(nk(i),nk(i)), Gam(nk(i),nk(i), arma::fill::eye), dotmuGam(p,nk(i)), Ddotmut(nk(i),p);
          arma::mat& m1 = m1_cache[nk(i)];
          arma::vec r(nk(i)), mu(nk(i)), xnorm(nk(i)), r1(nk(i)), r2(nk(i));

          for(int j=0; j<nt; j++){
            b = btau.col(j);
            mu = xx * gam * b;

            r = compute_r_opt(b, xx, Oma);

            xnorm = sqrt(nsub) * (yy - mu) / r;
            r1 = cpp_pnorm_faster(xnorm) + tau0(j) - 1.0;
            r2 = cpp_dnorm_faster(xnorm) / r;

            D = diagmat(r2);

            // Gam = diagmat(1.0 / (xx * gam * bderiv.col(j))); // Yang 2017
            dotmuGam = dotmu * Gam;
            Ddotmut = D * dotmu.t();

            //gi = kron(b, wt(j) * join_cols(dotmuGam * r1, dotmuGam * m1 * r1 )) / nsub;
            gi = vectorise(wt(j) * join_cols( dotmuGam * r1, dotmuGam * m1 * r1 ) * b.t()) / nsub;
            di = -kron(b*b.t(), wt(j) * join_cols( dotmuGam * Ddotmut, dotmuGam * m1 * Ddotmut )) / sqrt(nsub);

            arsumg += gi;
            arsumc += gi * gi.t();
            arsumgfirstdev += di;
          }
        }

        arsumc = arsumc * nsub;
        done = solve(rrt, arsumc, arsumgfirstdev);
        diter = 0;
        while(!done && diter < 500){
          arsumc.diag() += 100 * epsilon;
          done = solve(rrt, arsumc, arsumgfirstdev);
          diter++;
        }

        rr = rrt.t();
        arqif1dev = 2 * rr * arsumg + rho * (ganew - xi) + delta;
        arqif2dev = 2 * rr * arsumgfirstdev;

        ganew = ganew - w * solve(arqif2dev + rho * eyepk, arqif1dev);
        Oma = robust_inv(arqif2dev);

        betadiff = arma::norm(ganew - ga, 2);

        if(betadiff < tol) break;

        Q = nsub * as_scalar(arsumg.t() * solve(arsumc, arsumg));

        // Armijo line search
        while(Q > Q0 + 0.1 * dot(arqif1dev, ganew - ga)){
          w *= 0.5;
          if(w < 1e-10) break;
        }

      }

      // update xi
      xiold = xi;
      xi = soft(ganew + delta / rho, lambda * v / rho);

      // update delta
      diff = ganew - xi;
      delta = delta + rho * diff;

      primal_res = arma::norm(diff, 2);
      dual_res = - rho * arma::norm(xi - xiold, 2);
      // // adaptive rho
      if(primal_res > 10 * dual_res) rho = rho * 2;
      if(dual_res > 10 * primal_res) rho = rho / 2;

      // stopping criterion
      if(primal_res <= sqrt(p*K)*e_abs + e_rel*std::max(arma::norm(ganew),arma::norm(xi))  &&
         dual_res <= sqrt(2*p*K)*e_abs + e_rel*rho*arma::norm(delta)) break;
    }
  }else if(type=="ar"){
    arma::mat rr(p*K, p*K*3, arma::fill::zeros);
    arma::mat rrt(p*K*3, p*K, arma::fill::zeros);
    arma::vec arsumg(p*K*3, arma::fill::zeros), gi(p*K*3,arma::fill::zeros);
    arma::mat arsumc(p*K*3, p*K*3, arma::fill::zeros), di(p*K*3,p*K,arma::fill::zeros);
    arma::mat arsumgfirstdev(p*K*3, p*K, arma::fill::zeros);

    std::unordered_map<int, std::pair<arma::mat, arma::mat>> matrix_templates;
    for (int i = 0; i < nsub; i++) {
      int current_nk = nk(i);
      if (matrix_templates.find(current_nk) == matrix_templates.end()) {
        arma::mat m1 = arma::zeros<arma::mat>(current_nk, current_nk);
        arma::mat m2 = arma::zeros<arma::mat>(current_nk, current_nk);

        m1(0, 0) = m1(current_nk-1, current_nk-1) = 1.0;
        m2(0, 1) = m2(current_nk-1, current_nk-2) = 1.0;
        for (int jj = 1; jj < current_nk-1; jj++) {
          m2(jj, jj-1) = m2(jj, jj+1) = 1.0;
        }

        matrix_templates[current_nk] = {m1, m2};
      }
    }


    for(int ii=0; ii < maxit; ++ii){
      w = 1.0;
      for(int iteration=0; iteration < maxit; ++iteration){
        ga = ganew;
        gam = reshape(ga, p, K);
        Q0 = Q;
        arsumg.zeros();
        arsumc.zeros();
        arsumgfirstdev.zeros();

        for(int i=0; i<nsub; i++){
          arma::mat xx = x.rows(arma::span(rowindex(i), rowindex(i+1)-1));
          arma::vec yy = y.rows(arma::span(rowindex(i), rowindex(i+1)-1));
          arma::mat dotmu = xx.t(), D(nk(i),nk(i)), Gam(nk(i),nk(i), arma::fill::eye), dotmuGam(p,nk(i)), dotmuGam1(p,nk(i)), dotmuGam2(p,nk(i)), Ddotmut(nk(i),p);
          arma::vec r(nk(i)), mu(nk(i)), xnorm(nk(i)), r1(nk(i)), r2(nk(i));
          arma::mat& m1 = matrix_templates[nk(i)].first;
          arma::mat& m2 = matrix_templates[nk(i)].second;

          for(int j=0; j<nt; j++){

            b = btau.col(j);
            mu = xx * gam * b;

            r = compute_r_opt(b, xx, Oma);
            xnorm = sqrt(nsub) * (yy - mu) / r;
            r1 = cpp_pnorm_faster(xnorm) + tau0(j) - 1.0;
            r2 = cpp_dnorm_faster(xnorm) / r;
            D = diagmat(r2);

            // Gam = diagmat(1.0 / (xx * gam * bderiv.col(j)));
            dotmuGam = dotmu * Gam;
            dotmuGam1 = dotmuGam * m1;
            dotmuGam2 = dotmuGam * m2;
            Ddotmut = D * dotmu.t();

            //gi = kron(b, wt(j) * join_cols( dotmuGam * r1, dotmuGam1 * r1, dotmuGam2 * r1 )) / nsub;
            gi = vectorise(wt(j) * join_cols( dotmuGam * r1, dotmuGam * m1 * r1, dotmuGam * m2 * r1 ) * b.t()) / nsub;
            di = -kron(b*b.t(), wt(j) * join_cols(
              dotmuGam * Ddotmut,
              dotmuGam1 * Ddotmut,
              dotmuGam2 * Ddotmut
            )) / sqrt(nsub);

            arsumg += gi;
            arsumc += gi * gi.t();
            arsumgfirstdev += di;
          }
        }
        arsumc = arsumc * nsub;
        done = solve(rrt, arsumc, arsumgfirstdev);
        diter = 0;
        while(!done && diter < 500){
          arsumc.diag() += 100 * epsilon;
          done = solve(rrt, arsumc, arsumgfirstdev);
          diter++;
        }

        rr = rrt.t();
        arqif1dev = 2 * rr * arsumg + rho * (ganew - xi) + delta;
        arqif2dev = 2 * rr * arsumgfirstdev;

        ganew = ganew - w * solve(arqif2dev + rho * eyepk, arqif1dev);
        Oma = robust_inv(arqif2dev);

        betadiff = arma::norm(ganew - ga, 2);

        if(betadiff < tol) break;

        Q = nsub * as_scalar(arsumg.t() * solve(arsumc, arsumg));

        // Armijo line search
        while(Q > Q0 + 0.1 * dot(arqif1dev, ganew - ga)){
          w *= 0.5;
          if(w < 1e-10) break;
        }
      }

      // update xi
      xiold = xi;
      xi = soft(ganew + delta / rho, lambda * v / rho);

      // update delta
      diff = ganew - xi;
      delta = delta + rho * diff;

      // adaptive rho
      primal_res = arma::norm(diff, 2);
      dual_res = -rho * arma::norm(xi - xiold, 2);
      if(primal_res > 10 * dual_res) rho = rho * 2;
      if(dual_res > 10 * primal_res) rho = rho / 2;

      // stopping criterion
      if(primal_res <= sqrt(p*K)*e_abs + e_rel*std::max(arma::norm(ganew),arma::norm(xi))  &&
         dual_res <= sqrt(2*p*K)*e_abs + e_rel*rho*arma::norm(delta)) break;
    }
  }else if(type=="wi"){
    arma::mat rr(p*K, p*K, arma::fill::zeros);
    arma::mat rrt(p*K, p*K, arma::fill::zeros);
    arma::vec arsumg(p*K, arma::fill::zeros), gi(p*K,arma::fill::zeros);
    arma::mat arsumc(p*K, p*K, arma::fill::zeros), di(p*K,p*K,arma::fill::zeros);
    arma::mat arsumgfirstdev(p*K, p*K, arma::fill::zeros);

    for(int ii=0; ii < maxit; ++ii){
      w = 1.0;
      for(int iteration=0; iteration < maxit; ++iteration){
        ga = ganew;
        gam = reshape(ga, p, K);
        Q0 = Q;
        arsumg.zeros();
        arsumc.zeros();
        arsumgfirstdev.zeros();

        for(int i=0; i<nsub; i++){
          arma::mat xx = x.rows(arma::span(rowindex(i), rowindex(i+1)-1));
          arma::vec yy = y.rows(arma::span(rowindex(i), rowindex(i+1)-1));
          arma::mat dotmu = xx.t(), D(nk(i),nk(i)), Gam(nk(i),nk(i), arma::fill::eye), dotmuGam(p,nk(i));
          arma::vec r(nk(i)), mu(nk(i)), xnorm(nk(i)), r1(nk(i)), r2(nk(i));

          for(int j=0; j<nt; j++){

            b = btau.col(j);
            mu = xx * gam * b;

            r = compute_r_opt(b, xx, Oma);
            xnorm = sqrt(nsub) * (yy - mu) / r;
            r1 = cpp_pnorm_faster(xnorm) + tau0(j) - 1.0;
            r2 = cpp_dnorm_faster(xnorm) / r;
            D = diagmat(r2);

            // Gam = diagmat(1.0 / (xx * gam * bderiv.col(j)));
            dotmuGam = dotmu * Gam;
            gi = vectorise(wt(j) * dotmuGam * r1 * b.t()) / nsub;
            di = -kron(b*b.t(), wt(j) * dotmuGam * D * dotmu.t()) / sqrt(nsub);

            arsumg += gi;
            arsumc += gi * gi.t();
            arsumgfirstdev += di;
          }
        }
        arsumc = arsumc * nsub;
        done = solve(rrt, arsumc, arsumgfirstdev);
        diter = 0;
        while(!done && diter < 500){
          arsumc.diag() += 100 * epsilon;
          done = solve(rrt, arsumc, arsumgfirstdev);
          diter++;
        }

        rr = rrt.t();
        arqif1dev = 2 * rr * arsumg + rho * (ganew - xi) + delta;
        arqif2dev = 2 * rr * arsumgfirstdev;

        ganew = ganew - w * solve(arqif2dev + rho * eyepk, arqif1dev);
        Oma = robust_inv(arqif2dev);

        betadiff = arma::norm(ganew - ga, 2);

        if(betadiff < tol) break;

        Q = nsub * as_scalar(arsumg.t() * solve(arsumc, arsumg));

        // Armijo line search
        while(Q > Q0 + 0.1 * dot(arqif1dev, ganew - ga)){
          w *= 0.5;
          if(w < 1e-10) break;
        }
      }

      // update xi
      xiold = xi;
      xi = soft(ganew + delta / rho, lambda * v / rho);

      // update delta
      diff = ganew - xi;
      delta = delta + rho * diff;

      // adaptive rho
      primal_res = arma::norm(diff, 2);
      dual_res = rho * arma::norm(xi - xiold, 2);
      if(primal_res > 10 * dual_res) rho = rho * 2;
      if(dual_res > 10 * primal_res) rho = rho / 2;

      // stopping criterion
      if(primal_res <= sqrt(p*K)*e_abs + e_rel*std::max(arma::norm(ganew),arma::norm(xi))  &&
         dual_res <= sqrt(2*p*K)*e_abs + e_rel*rho*arma::norm(delta)) break;
    }
  }


  Rcpp::List out;
  out["gamma"] = xi;
  out["cov"] = Oma;
  out["Q"] = Q;
  double Dev = log(Q + lambda * accu(v % xi));

  int df = 0;
  arma::vec dd(p, arma::fill::zeros);
  for(int jj=0; jj<K; jj++){
    for(int ii=0; ii<p; ii++){
      dd(ii) += xi(ii+K*jj);
    }
  }
  for(int ii=0; ii<p; ii++){
    if(abs(dd(ii)) > 1e-3) df++;
  }

  out["Dev"] = Dev;
  if(type=="cs"){
    out["AIC"] = Q + 2 * df;
    out["BIC"] = Q + log(nsub) * df;
  }else if(type == "ar"){
    out["AIC"] = Q + 2 * df;
    out["BIC"] = Q + log(nsub) * df;
  }else if(type == "wi"){
    out["AIC"] = Q + 2 * df;
    out["BIC"] = Q + log(nsub) * df;
  }

  return out;
}



//' Basic Function of Cross-validation for Penalized Longitudinal Parametric Quantile Regression with Adaptive LASSO
//'
//' Performs cross-validation to select the optimal tuning parameter for penalized longitudinal
//' parametric quantile regression with adaptive LASSO penalty. This function evaluates
//' a sequence of lambda values and selects the one that minimizes the quadratic inference function
//' Bayesian Information Criterion (QIFBIC).
//'
//' @param type Character, working correlation structure: "cs" (compound symmetry),
//'   "ar" (autoregressive), or "wi" (working independence).
//' @param lambda Numeric vector of tuning parameters to evaluate.
//' @param x Numeric matrix of covariates (n x p), with rows sorted by subject and time.
//' @param y Numeric vector of responses (length n).
//' @param nk Integer vector, number of observations for each subject.
//' @param rowindex Integer vector indicating the starting row index (minus 1) for each subject.
//'   Should have length (nsub + 1) with rowindex[1] = 0 and rowindex[nsub+1] = n.
//' @param tau0 Numeric vector of quantile levels used for composite quantile regression.
//' @param nsub Integer, number of subjects.
//' @param nt Integer, number of tau levels.
//' @param gammaint Numeric vector of initial coefficient estimates (length p*K).
//' @param v Numeric vector of adaptive weights for the LASSO penalty (length p*K).
//' @param rho Numeric, penalty parameter for the ADMM algorithm.
//' @param btau Numeric matrix of basis function values at tau0 (K x nt).
//' @param bderiv Numeric matrix of basis function derivative values at tau0 (K x nt).
//' @param K Integer, number of basis functions.
//' @param wt Numeric vector of weights for different quantile levels (length nt).
//'
//' @return A list containing:
//' \item{r}{Optimal model results corresponding to the best lambda value}
//' \item{gamma}{Estimated coefficients from the optimal model}
//' \item{QIFBIC}{Optimal BIC value}
//' \item{lambda}{Optimal lambda value that minimizes BIC}
//' \item{rlist}{List of results for all lambda values evaluated}
//'
//' @details
//' This function performs cross-validation by evaluating a sequence of tuning parameters
//' (lambda values) for the adaptive LASSO penalty in penalized longitudinal parametric
//' quantile regression. The selection criterion is based on the Bayesian Information
//' Criterion (BIC), which balances model fit and complexity.
//'
//' Warm starting significantly improves computational efficiency by using the solution
//' from the previous lambda as the initial value for the next lambda.
//'
//' @seealso
//' \code{\link{cqr_adlasso}} for fitting the penalized model for a specific lambda value.
//' \code{\link{CV_cqr_adlasso}} for the R wrapper function that provides a more user-friendly interface.
//' @references Yu, Z., Yu, K.M., Ni, Y.X., & Tian, M.Z. (2025). Unravelling Determinants of Global Life Expectancy: A Joint Quantile Approach for Heterogeneous Longitudinal Studies.
//' @useDynLib pLPQR, .registration = TRUE
//' @importFrom Rcpp evalCpp
//' @export
// [[Rcpp::export]]
Rcpp::List cv_cqr_adlasso0(String type,   // working correlation matrix with "cs" indicating the normal copula with an exchangeable correlation matrix, "ar" indicating the normal copula with an autoregressive structure of order 1, and "wi" indicating the working independence
                          arma::vec lambda, // the tuning parameter in pLPQR
                          arma::mat x,  // the covariate matrix, with rows sorted primarily by country and secondarily by year
                          arma::vec y,  // the corresponding response vector
                          arma::vec nk, // a vector indicating the number of observations for each subject
                          arma::vec rowindex, // the starting row index minus 1 for each country
                          arma::vec tau0,     // a vector of quantile levels
                          int nsub,  // number of subjects
                          int nt,  // number of tau's
                          arma::vec gammaint, // initial estimates of the unknown coefficient matrix
                          arma::vec v, // vector indicating the adaptive weights
                          double rho, // a constant penalty parameter
                          arma::mat btau,      // the value of the known functions b(tau) at tau0
                          arma::mat bderiv,  // the value of the derivative of the known functions b(tau)
                          int K,    // the length of the known functions b(tau)
                          arma::vec wt  // a vector of nonconstant weight measuring the varying amounts of information across quantiles
){
  double s = 100;
  double Lambda = lambda(0), lam = lambda(0);;
  int I = lambda.n_elem;
  arma::vec yhat = y;

  Rcpp::List r(1), rr(I);
  // Progress prog(I, true);

  r = cqr_adlasso(x,y, nk, rowindex, tau0, nsub,nt, gammaint, v, rho, Lambda, btau, bderiv, K, type, wt);
  s = sum(as<NumericVector>(r["BIC"]));
  rr[0] = r;
  // prog.increment();
  arma::vec local_gammaint = as<arma::vec>(r["gamma"]);
  double local_s;
  Rcpp::List local_rrr;

  for(int i=1; i<I; i++){
    lam = lambda(i);
    local_rrr = cqr_adlasso(x,y, nk, rowindex, tau0, nsub,nt, local_gammaint, v, rho, lam, btau, bderiv, K, type, wt);
    local_s = sum(as<NumericVector>(local_rrr["BIC"]));
    local_gammaint = as<arma::vec>(local_rrr["gamma"]);
    rr[i] = local_rrr;
    if(local_s < s){
      s = local_s;
      r = local_rrr;
      Lambda = lam;
    }
  }

  Rcpp::List out;
  out["r"] = r;
  out["gamma"] = r["gamma"];
  out["QIFBIC"] = s;
  out["lambda"] = Lambda;
  out["rlist"] = rr;

  return out;
}

//' Goodness-of-Fit Tests for Penalized Longitudinal Parametric Quantile Regression
//'
//' Performs stratified analysis and goodness-of-fit testing for parametric quantile regression models.
//' This function computes probability integral transform (PIT) values and conducts
//' Cramér-von Mises and Kolmogorov-Smirnov tests to assess model adequacy across strata.
//'
//' @param data A list containing the dataset with the following components:
//'   \itemize{
//'     \item \code{y}: Numeric vector of responses
//'     \item \code{x}: Numeric matrix of covariates
//'     \item \code{id}: Numeric vector of subject identifiers
//'     \item \code{nsub}: Integer, number of subjects
//'     \item \code{index}: Numeric vector indicating stratum membership for each observation
//'   }
//' @param bf Function that computes basis functions, takes a quantile level tau and returns a numeric vector.
//' @param bderivf Function that computes derivatives of basis functions, takes a quantile level tau and returns a numeric vector.
//'
//' @return A list containing:
//' \item{cvm}{Global weighted Cramér-von Mises statistic across all strata}
//' \item{weights}{Numeric vector of weights for each stratum based on sample size}
//' \item{stats}{List containing stratum-specific test statistics:
//'   \itemize{
//'     \item \code{cvm}: Cramér-von Mises statistics for each stratum
//'     \item \code{ks}: Kolmogorov-Smirnov statistics for each stratum
//'   }}
//' \item{cvms}{Numeric vector of Cramér-von Mises statistics for each stratum}
//' \item{kss}{Numeric vector of Kolmogorov-Smirnov statistics for each stratum}
//' \item{PIT.S}{List of probability integral transform values for each stratum}
//'
//' @references Yu, Z., Yu, K.M., Ni, Y.X., & Tian, M.Z. (2025). Unravelling Determinants of Global Life Expectancy: A Joint Quantile Approach for Heterogeneous Longitudinal Studies.
//'
//' @seealso
//' \code{\link{cv_cqr_adlasso0}} for the cross-validation function used internally.
//' \code{\link{p_bisec_cpp}} for the probability integral transform computation.
//' @references Yu, Z., Yu, K.M., Ni, Y.X., & Tian, M.Z. (2025). Unravelling Determinants of Global Life Expectancy: A Joint Quantile Approach for Heterogeneous Longitudinal Studies.
//' Yang, C.-C., Chen, Y.-H., and Chang, H.-Y. (2017). Composite marginal quantile regression analysis for longitudinal adolescent body mass index data: Quantile regression for longitudinal adolescent BMI. Statistics in Medicine, 36(21):3380–3397.
//' @export
// [[Rcpp::export]]
Rcpp::List stratumCvM_para_test_cpp(const Rcpp::List& data,
                                    const Rcpp::Function& bf, // the K-dimensional vector composed of known functions of tau
                                    const Rcpp::Function& bderivf // the K-dimensional vector composed of derivative of bf
) {
  arma::vec y = Rcpp::as<arma::vec>(data["y"]);
  arma::mat x = Rcpp::as<arma::mat>(data["x"]);
  arma::vec id = Rcpp::as<arma::vec>(data["id"]);
  int nsub = Rcpp::as<int>(data["nsub"]);
  arma::vec index = Rcpp::as<arma::vec>(data["index"]);

  arma::vec tau0 = arma::linspace(0.01, 0.99, 99);
  int nt = tau0.n_elem;
  double rho = 1.0;

  NumericVector bf_sample = bf(0.0);
  int K = bf_sample.length();

  arma::vec lam = arma::linspace(0.1, 0.9, 9); // seq(0.1,0.9,0.1)

  arma::mat btau(K, nt);
  arma::mat bderiv(K, nt);
  arma::vec wt(nt, arma::fill::ones);

  for(int i = 0; i < nt; i++) {
    NumericVector bf_result = bf(tau0(i));
    NumericVector bderiv_result = bderivf(tau0(i));
    for(int j = 0; j < K; j++) {
      btau(j, i) = bf_result(j);
      bderiv(j, i) = bderiv_result(j);
    }
  }

  arma::vec rowindex(nsub + 1);
  rowindex(0) = 0;
  for(int i = 1; i <= nsub; i++) {
    rowindex(i) = i;
  }

  arma::vec unique_stratum = arma::unique(index);
  int n_strata = unique_stratum.n_elem;

  Rcpp::List PIT_S(n_strata);
  Rcpp::CharacterVector stratum_names(n_strata);

  for(int s = 0; s < n_strata; s++) {
    double stratum_val = unique_stratum(s);

    arma::uvec stratum_indices = arma::find(index == stratum_val);
    int n_stratum = stratum_indices.n_elem;

    if(n_stratum > 0) {
      arma::vec ys(n_stratum);
      arma::mat xs(n_stratum, x.n_cols);

      for(int i = 0; i < n_stratum; i++) {
        int idx = stratum_indices(i);
        ys(i) = y(idx);
        xs.row(i) = x.row(idx);
      }

      Rcpp::Environment qrcm_env("package:qrcm");
      Rcpp::Function iqr_func = qrcm_env["iqr"];

      Rcpp::DataFrame ys_df = Rcpp::DataFrame::create(
        Rcpp::Named("ys") = ys,
        Rcpp::Named("xs") = xs
      );

      Rcpp::List IQR_result = iqr_func(Rcpp::Named("formula") = Rcpp::Formula("ys ~ -1 + xs"),
                                       Rcpp::Named("formula.p") = Rcpp::Formula("~ I(qnorm(p))"),
                                       Rcpp::Named("data") = ys_df);

      arma::vec gammaint = Rcpp::as<arma::vec>(IQR_result["coefficients"]);

      Rcpp::List WI_result = cv_cqr_adlasso0("wi", lam, xs, ys, arma::ones<arma::vec>(nsub), rowindex,
                                            tau0, nsub, nt, gammaint, 1.0 / arma::abs(gammaint), rho, btau, bderiv, K, wt);

      arma::vec gamma_vec = Rcpp::as<arma::vec>(WI_result["gamma"]);
      arma::mat gamma_mat = arma::reshape(gamma_vec, K, gamma_vec.n_elem / K);
      gamma_mat = gamma_mat.t();

      arma::vec pit_result = p_bisec_cpp(gamma_mat, ys, xs, bf);

      PIT_S[s] = pit_result;
      stratum_names[s] = std::to_string(static_cast<int>(stratum_val));
    }
  }

  arma::vec cvm_values(n_strata);
  arma::vec ks_values(n_strata);

  for(int s = 0; s < n_strata; s++) {
    arma::vec pit = Rcpp::as<arma::vec>(PIT_S[s]);
    int n = pit.n_elem;

    if(n > 0) {
      arma::vec p_sorted = arma::sort(pit);

      arma::vec expected = arma::linspace(0.5, n - 0.5, n) / n;
      double cvm = arma::sum(arma::square(p_sorted - expected)) + 1.0 / (12.0 * n);

      arma::vec empirical_cdf = arma::linspace(1.0, n, n) / n;
      double ks = 0.0;
      for(int i = 0; i < n; i++) {
        double diff1 = std::abs(p_sorted(i) - (i + 1.0) / n);
        double diff2 = std::abs(p_sorted(i) - i / n);
        ks = std::max(ks, std::max(diff1, diff2));
      }

      cvm_values(s) = cvm;
      ks_values(s) = ks;
    } else {
      cvm_values(s) = 0.0;
      ks_values(s) = 0.0;
    }
  }

  arma::vec stratum_counts(n_strata);
  for(int s = 0; s < n_strata; s++) {
    double stratum_val = unique_stratum(s);
    stratum_counts(s) = arma::sum(index == stratum_val);
  }

  arma::vec weights = stratum_counts / y.n_elem;
  double cvm_global = arma::sum(weights % cvm_values);

  Rcpp::List stat_stratum = Rcpp::List::create(
    Rcpp::Named("cvm") = cvm_values,
    Rcpp::Named("ks") = ks_values
  );

  return Rcpp::List::create(
    Rcpp::Named("cvm") = cvm_global,
    Rcpp::Named("weights") = weights,
    Rcpp::Named("stats") = stat_stratum,
    Rcpp::Named("cvms") = cvm_values,
    Rcpp::Named("kss") = ks_values,
    Rcpp::Named("PIT.S") = PIT_S
  );
}

