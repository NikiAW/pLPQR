# # ===========================================================================================
# # R code for the case study of
# # ``Unravelling Determinants of Global Life Expectancy: A Joint Quantile Approach for Heterogeneous Longitudinal Studies"
# # ===================================================================
rm(list=ls())
library(qrcm)
library(Rcpp);library(RcppProgress)
library(pLPQR)
load("data2019.RData")
data <- variables

# ====================== Figure 1. The dynamic of life expectancy ======================================
library(ggrepel) # geom_text_repel
library(ggtext) # element_markdown
library(tidyverse)
library(systemfonts)
highlights <- c("Japan","Switzerland", "Iceland", "Italy", "Spain",
                "United Kingdom","United States","China",
                "Lesotho","Eswatini","Central African Republic")
n <- length(highlights)
data$group <- data$country
levels(data$group) <- c(levels(data$group),"other")
data$group[!(data$country %in% highlights)] <- "other"
data$name_lab <- as.character(data$country)
data$name_lab[!(data$year == 2019)] <- NA_character_

mycolor <- c("#3969AC","#11A579","#008695","#CF1C90",
             "#E73F74","#F2B701","grey50","#F97B72",
             "#E68310","#7F3C8D","#80BA5A")

ggplot(data %>% filter(group != "other"),
       aes(year, life, group = country)) +
  geom_vline(
    xintercept = c(seq(2000, 2019, by = 5),2019),
    color = "grey91",
    size = .8
  ) +
  geom_segment(
    data = tibble(y = seq(40, 85, by = 10), x1 = 2000, x2 = 2019),
    aes(x = x1, xend = x2, y = y, yend = y),
    inherit.aes = FALSE,
    color = "grey91",
    size = .8
  ) +
  geom_line(
    data = data %>% filter(group == "other"),
    color = "grey75",
    size = 1,
    alpha = .5
  ) +
  ## colored lines
  geom_line(aes(color = group), size = 1.5) +
  geom_point(aes(color = group), data = data[data$group != "other",], size=1.7) +
  geom_text_repel(
    aes(color = group, label = name_lab),
    data = data[data$group != "other",],
    family = "Avenir Next Condensed",
    fontface = "bold",
    size = 9,
    direction = "y",
    xlim = c(2024, NA),
    hjust = 0,
    segment.size = 1.9,  # size of the dotted line
    segment.alpha = .8,
    segment.linetype = "dotted",
    box.padding = .4,
    segment.curvature = -0.1,
    segment.ncp = 3,
    segment.angle = 20
  ) +
  coord_cartesian(
    clip = "off",
    ylim = c(40, 85)
  ) +
  scale_x_continuous(
    expand = c(0, 0),
    limits = c(2000, 2026),  # adjust for the window width
    breaks = c(seq(2000, 2019, by = 5),2019)
  ) +
  scale_y_continuous(
    expand = c(0, 0),
    breaks = seq(40, 85, by = 5)
  ) +
  scale_color_manual(
    values = c("Central African Republic"="#3969AC",
               "China"="#11A579",
               "Eswatini"="#008695",
               "Italy"="#CF1C90",
               "Iceland"="#E73F74",
               "Japan"="#F2B701",
               "Lesotho"="grey50",
               "Spain"="#F97B72",
               "Switzerland"="#E68310",
               "United Kingdom"="#7F3C8D",
               "United States"="#80BA5A")
  ) + labs(x = 'Year                        ',
           y = 'Life Expectancy (years)', col = 'Country') +
  theme(plot.title = element_markdown(size = 10),
        legend.position = "none",
        text = element_text(size=25))



# ----- Figure E1. sequential box plots for outlier detection -------
library(ggplot2)
library(dplyr)

identify_outliers <- function(data) {
  outliers <- data %>%
    group_by(year) %>%
    mutate(
      Q1 = quantile(life, 0.25, na.rm = TRUE),
      Q3 = quantile(life, 0.75, na.rm = TRUE),
      IQR = Q3 - Q1,
      lower_bound = Q1 - 1.5 * IQR,
      upper_bound = Q3 + 1.5 * IQR,
      is_outlier = life < lower_bound | life > upper_bound
    ) %>%
    filter(is_outlier) %>%
    select(year, life, country, lower_bound, upper_bound)

  return(outliers)
}
outliers_df <- identify_outliers(data)

selected_palette <- c("#FC757B","#F97F5F","#FAA26F","#FDCD94","#FEE199","#B0D6A9","#65BDBA","#3C9BC9","#2686b5","#057dcd")
n_years <- length(unique(data$year))
extended_palette <- colorRampPalette(selected_palette)(n_years)

ggplot(data, aes(x = factor(year), y = life)) +
  geom_boxplot(aes(fill = factor(year)), alpha = 0.8, outlier.color = "red") +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "darkred") +
  geom_point(data = outliers_df,
             color = "red", size = 3, shape = 1, stroke = 1.5) +
  geom_text(data = outliers_df,
            aes(label = sprintf("%s\n(%.2f)", country, life)),
            vjust = 1.5, size = 2.8, color = "red", lineheight = 0.8) +
  labs(x = "Year",
       y = "Life Expectancy",
       fill = "Year") +
  scale_fill_manual(values = extended_palette) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# ====================== pLPQR modeling ======================================

index <- subset(variables, select=c(country, countrycode, year))
y <- scale(log(variables$life), scale=T, center=T)
reg <- model.matrix(~ region -1, variables)[,-1]
x0 <- subset(variables, select = -c(country,countrycode,year,life,region))
xst <- apply(x0,2,scale, center=T, scale=T)
x <- cbind(1, reg, xst)
p <- dim(x)[2]

## model fitting --------
check <- function(u,tau){
  return(u*(tau-ifelse(u<0,1,0)))
}
eva0 <- function(theta,x,y){
  Mc <- Mc2 <- Mc3 <- Mc4 <- Mc5 <- Mc6 <- NULL
  tau <- seq(0.1,0.9,0.1)
  for(kk in 1:9){
    taui <- tau[kk]
    Mc[kk] <- mean(check(y-x%*%theta%*%bf(taui),taui))
    Mc2[kk] <- mean((y-x%*%theta%*%bf(taui))^2)
    Mc3[kk] <- mean(abs(y-x%*%theta%*%bf(taui)))
    Mc4[kk] <- median(check(y-x%*%theta%*%bf(taui),taui))
    Mc5[kk] <- median((y-x%*%theta%*%bf(taui))^2)
    Mc6[kk] <- median(abs(y-x%*%theta%*%bf(taui)))
  }
  return(c(mean(Mc),mean(Mc2),mean(Mc3),mean(Mc4),mean(Mc5),mean(Mc6)))
}
eva1 <- function(x,y,theta0,theta1,theta2,theta3){
  rr <- rbind(eva0(theta0,x,y),eva0(theta1,x,y),eva0(theta2,x,y),eva0(theta3,x,y))
  colnames(rr) <- c("MeanC","MSE","MeanAbs","MedC","MedSE","MedAbs")
  rownames(rr) <- c("wi","IQR","cs","ar")
  return(t(rr))
}

bftype="sqrt-log1-p"
nsub <- length(unique(index$country))  # 170 countries
nk <- c(table(index$country))  # 20
m <- length(unique(index$year)) # 20 years
tau0 <- seq(0.01,0.99,0.01)
nt <- length(tau0)
wt <- rep(1,length(tau0))
rho <- 0.1
lambda <- 0.1

seed <- 123
bf <- function(xx){return(c(1,sqrt(-log(1-xx))))}
bderivf <- function(xx){return(c(0, 1/(2*(1-xx)*sqrt(-log(1-xx)))))}
K <- 2
set.seed(seed)
IQR <- iqr(y~ -1 + x, formula.p = ~ I(sqrt(-log((1-p)))),remove.qc = TRUE)
initials <- c(IQR$coefficients)

# # CQR using qif
CS.CQR <- AR.CQR <- WI.CQR <- CS.lasso <- AR.lasso <- WI.lasso <- CS.adlasso <- AR.adlasso <- WI.adlasso <- NULL

set.seed(seed)
CS.CQR <- cqr(type="cs", x,y, nk,rowindex=c(0,cumsum(nk)), tau0, nsub,nt, gammaint=initials, btau=sapply(tau0,bf), bderiv=sapply(tau0,bderivf), K=length(bf(0)), wt=rep(1,length(tau0)))
set.seed(seed)
AR.CQR <- cqr(type="ar", x,y, nk,rowindex=c(0,cumsum(nk)), tau0, nsub,nt, gammaint=initials, btau=sapply(tau0,bf), bderiv=sapply(tau0,bderivf), K=length(bf(0)), wt=rep(1,length(tau0)))

# # # LASSO
set.seed(seed)
CS.lasso <- CV_cqr_adlasso(type="cs", lambda =seq(1,0.1,-0.1), x,y, nsub,nk, bf,bderivf, gammaint=c(CS.CQR$gamma), v=rep(1,length(initials)), tau = seq(0.01, 0.99, 0.01),rho=0.01)
set.seed(seed)
AR.lasso <- CV_cqr_adlasso(type="ar", lambda =seq(1,0.1,-0.1), x,y, nsub,nk, bf,bderivf, gammaint=c(AR.CQR$gamma), v=rep(1,length(initials)), tau = seq(0.01, 0.99, 0.01),rho=0.01)

# # # adaptive lasso
set.seed(seed)
CS.adlasso <- CV_cqr_adlasso(type="cs", lambda =seq(1,0.1,-0.1), x,y, nsub,nk, bf,bderivf, gammaint=c(CS.CQR$gamma), v=1/abs(c(CS.CQR$gamma)), tau = seq(0.01, 0.99, 0.01),rho=0.01)
set.seed(seed)
AR.adlasso <- CV_cqr_adlasso(type="ar", lambda =seq(1,0.1,-0.1), x,y, nsub,nk, bf,bderivf, gammaint=c(AR.CQR$gamma), v=1/abs(c(AR.CQR$gamma)), tau = seq(0.01, 0.99, 0.01),rho=0.01)


# ------- Sensitivity analysis -------------
outlier.index <- (variables$country != "Lesotho")
x.rm <- x[outlier.index,]
y.rm <- y[outlier.index]
index.rm <- index[index$country!="Lesotho",]

bftype="sqrt-log1-p"
nsub <- length(unique(index.rm$country))  # 170 countries
nk <- c(table(index.rm$country))  # 20
m <- length(unique(index.rm$year)) # 20 years

seed <- 123
bf <- function(xx){return(c(1,sqrt(-log(1-xx))))}
bderivf <- function(xx){return(c(0, 1/(2*(1-xx)*sqrt(-log(1-xx)))))}
K <- 2
set.seed(seed)
IQR.rm <- iqr(y.rm~ -1 + x.rm, formula.p = ~ I(sqrt(-log((1-p)))),remove.qc = TRUE)
initials.rm <- c(IQR.rm$coefficients)

CS.CQR.rm <- CS.lasso.rm <- CS.adlasso.rm <- NULL

# # CQR using qif
set.seed(seed)
CS.CQR.rm <- cqr(type="cs", x.rm, y.rm, nk,rowindex=c(0,cumsum(nk)), tau0, nsub,nt, gammaint=initials.rm, btau=sapply(tau0,bf), bderiv=sapply(tau0,bderivf), K=length(bf(0)), wt=rep(1,length(tau0)))

# # # adaptive lasso
set.seed(seed)
CS.adlasso.rm <- CV_cqr_adlasso(type="cs", lambda =seq(1,0.1,-0.1), x.rm,y.rm, nsub,nk, bf,bderivf, gammaint=c(CS.CQR.rm$gamma), v=1/abs(c(CS.CQR.rm$gamma)), tau = seq(0.01, 0.99, 0.01),rho=0.01)


# ============= Figure 3. The comparison of the quantile-specific covariate effects beta(tau)======================================
library(ggplot2)
colnames(x)[colnames(x)=="regionAmericas"]="Americas"
colnames(x)[colnames(x)=="regionEastern Mediterranean"]="Eastern_Mediterranean"
colnames(x)[colnames(x)=="regionSouth-East Asia"]="SouthEast_Asia"
colnames(x)[colnames(x)=="regionWestern Pacific"]="Western_Pacific"
colnames(x)[colnames(x)=="regionEurope"]="Europe"
colnames(x)[colnames(x)=="temp"]="temperature"

X <- x[,-1]
Y <- y
nk <- c(table(variables$country))
TAU <- seq(0.1,0.9,0.05)
BETA <- SE <- NULL

# CQR estimates
cqrMM <- function(x,     # n*d
                  y,     # n*1
                  tau=seq(0.1,0.9,0.1),   # 1*q vector
                  w=rep(1,dim(x)[1]),     # n*1 vector
                  bb=6*tau-mean(6*tau)+coef[1],    # 1*q vector
                  beta=NULL,  # initial value for beta
                  maxit=200, toler=1e-6){
  np <- dim(x)
  n <- np[1]
  d <- np[2]
  # w <- rep(n,1)
  nX <- cbind(1,x)*sqrt(w)
  ny <- y * sqrt(w)
  coef <- ginv(t(nX)%*%nX)%*%t(nX)%*%ny
  beta <- coef[-1]
  # bb <- 6*tau-mean(6*tau)+coef[1]

  q <- length(tau)
  c <- 2*tau-1
  delta=1;it=0
  tn <- toler/n
  e0 <- -tn/log(tn)
  eps <- (e0-tn)/(1+log(e0))

  sumw <- sum(w)
  p5 <- t(w)%*%x
  p1 <- matrix(0,d,q)
  p6 <- array(0,dim=c(d,d,q))

  while(delta>toler && it<maxit){
    it <- it+1
    beta0 <- beta
    bb0 <- bb
    ar <- abs(matrix(y-x%*%beta0,n,q)-matrix(bb0,n,q,byrow=T))
    A <- 1/(ar+eps)
    p2 <- t(y*w)%*%A
    p3 <- t(x*w)%*%A
    p4 <- t(w)%*%A
    denor <- 0;numer <- 0
    for(k in 1:q){
      for(j in 1:d){
        temp1 <- t(A[,k]*w*x[,j])
        p1[j,k] <- temp1%*%y
        p6[j,,k] <- temp1%*%x
      }
      denor <- denor+p6[,,k]-p3[,k]%*%t(p3[,k])/p4[k]
      numer <- numer+t(p1[,k]-p2[k]%*%p3[,k]/p4[k]-sumw*c[k]*p3[,k]/p4[k]+c[k]*p5)
    }
    beta <- ginv(denor)%*%numer
    bb <- (t((y-x%*%beta)*w)%*%A+sumw*c)/p4
    delta <- abs(mean(bb-bb0))+sum(abs(beta-beta0))
  }
  return(list(b=bb,beta=beta))
}

# CQR for longitudinal data in Zhao, Lian, and Song (2017)
cqrqif_Zhao <- function(x,y,
                        nk,      # number of observations for each subjects
                        betaint, # initial estimates
                        af,      # quantile estimates of error density
                        tau0,     # composite quantiles
                        type,    # working correlation matrix with type 1 for CS and 2 for AR(1)
                        wt=NULL  # weights for WCQR
){

  nt <- length(tau0);      # nt: number of quantile levels
  if(length(wt)==0) wt=ones(nt,1)

  nsub <- length(nk)
  p <- dim(x)[2]

  # for setting a pointer to the start and end of data index for a subject
  cn <- c(0,cumsum(nk))
  # x,y for each subject
  xa <- ya <- list()
  for(i in 1:nsub){
    xa[[i]] <- x[(cn[i]+1):(cn[i+1]),]  # nk*p matrix
    ya[[i]] <- y[(cn[i]+1):(cn[i+1])]
  }

  Q <- 1e6
  betadiff <- 1
  iteration <- 0
  maxit <- 500
  w <- 1    # the step size in Newton-Raphson algorithm
  Oma <- diag(p+nt)
  ga <- c(af, betaint)
  ganew <- ga

  mu <- dotmu <- vmu <- r1 <- r2 <- D <- list()

  # Newton-Raphson algorithm
  while((betadiff>1e-6)&(iteration<maxit)){
    iteration <- iteration + 1
    ga <- ganew
    arsumg <- arsumc <- arsumgfirstdev <- 0
    Q0 <- Q
    Oma0 <- Oma

    for(i in 1:nsub){
      for(jj in 1:nt){
        tau <- tau0[jj]
        xx <- cbind(matrix(0,nk[i],nt), xa[[i]])
        xx[,jj] <- rep(1,nk[i])
        mu[[i]] <- xx %*% ga
        dotmu[[i]] <- t(xx)
        # vmu[[i]] <- diag(nk[i])

        r <- sqrt(diag(xx%*% Oma %*%t(xx)))  # r below (8)
        r[sapply(r,function(xr){xr<1e-4})] <- 1e-4
        r1[[i]] <- pnorm(sqrt(nsub) * (ya[[i]]-mu[[i]])/r) - (1-tau)
        r2[[i]] <- sqrt(nsub) * dnorm(sqrt(nsub) * (ya[[i]]-mu[[i]])/r)/r
        D[[i]] <- diag(c(r2[[i]])) # nk * nk diagonal matrix

        if(type == "cs"){
          m1 <- matrix(1,nk[i],nk[i]) - diag(nk[i])
          gi <- wt[jj] * rbind(dotmu[[i]]%*%r1[[i]], dotmu[[i]]%*%m1%*%r1[[i]]) / (nsub*nt)
          di <- - wt[jj] * rbind(dotmu[[i]]%*%D[[i]]%*%t(dotmu[[i]]), dotmu[[i]]%*%m1%*%D[[i]]%*%t(dotmu[[i]])) / (nsub*nt)
        }else if(type == "ar"){
          m1 <- matrix(0,nk[i],nk[i])
          m1[1,2] <- m1[nk[i],nk[i]-1] <- 1
          for(j in 2:(nk[i]-1)){
            m1[j,j-1] <- m1[j,j+1] <- 1
          }
          gi <- wt[jj] * rbind(dotmu[[i]]%*%r1[[i]], dotmu[[i]]%*%m1%*%r1[[i]]) / nsub
          di <- - wt[jj] * rbind(dotmu[[i]]%*%D[[i]]%*%t(dotmu[[i]]), dotmu[[i]]%*%m1%*%D[[i]]%*%t(dotmu[[i]])) / nsub
        }else{
          gi <- wt[jj] * dotmu[[i]]%*%r1[[i]] / nsub
          di <- - wt[jj] * dotmu[[i]]%*%D[[i]]%*%t(dotmu[[i]]) / nsub
        }

        arsumg <- arsumg + gi
        arsumc <- arsumc + gi %*% t(gi)
        arsumgfirstdev <- arsumgfirstdev + di
      }
    }

    arsumc <- arsumc * nsub
    arcinv <- ginv(arsumc)

    arqif1dev <- t(arsumgfirstdev) %*% arcinv %*% arsumg
    arqif2dev <- t(arsumgfirstdev) %*% arcinv %*% arsumgfirstdev
    invarqif2dev <- ginv(arqif2dev)
    # update
    ganew <- ganew - w* invarqif2dev %*% arqif1dev

    Oma <- ginv(t(arsumgfirstdev)%*%arcinv%*%arsumgfirstdev)
    betadiff <- sum(abs(ganew-ga))
    if(max(abs(betadiff))<1e-6) break

    Q <- t(arsumg) %*% arcinv %*% arsumg
    if(abs(Q-Q0)<1e-6) break
    w <- w/2
  }

  beta <- ganew[(nt+1):(nt+p)]
  af <- ganew[1:nt]

  est <- diag(ginv(t(arsumgfirstdev)%*%arcinv%*%arsumgfirstdev))
  est <- est[(nt+1):(nt+p)]

  ret <- list(beta=beta, af=af, est=est)  # est: standard error estimates for coefficients
  return(ret)
}



ll=1
library(MASS)
for(taui in TAU){
  tau0 <- taui
  nt <- length(tau0)

  # initials
  initials <- cqrMM(X,Y,tau0)
  equalwt <- rep(1,nt)

  # cqrqif_Zhao for a single quantile
  QIF.CS.CQR <- cqrqif_Zhao(X,Y,nk,betaint=initials$beta,af=initials$b,tau0,type="cs",wt=equalwt)

  BETA <- rbind(BETA, c(QIF.CS.CQR$af, QIF.CS.CQR$beta))
  rownames(BETA)[ll] <- taui
  ll <- ll+1
}
colnames(BETA) <- colnames(x) # colnames(SE) <- colnames(x)[-1]
BETA0 <- BETA <- data.frame(tau=as.factor(TAU),BETA)

# beta
cv.bic.cs <- CS.adlasso
TAU <- seq(0.1,0.9,0.01)
thetahat <- matrix(cv.bic.cs$r$gamma,ncol=K)
rownames(thetahat) <- colnames(x)
thetahat
betahat <- thetahat %*% sapply(TAU,bf)
rownames(betahat)[1] <- colnames(x)[1] <- "Africa"
colnames(betahat) <- TAU

judge <- matrix(abs(cv.bic.cs$r$gamma),ncol=K) !=0
VS <- rowSums((judge)*1)!=0
vary <- rowSums(matrix(judge[,-1],ncol=K-1)*1) != 0

# constant effects
cbind(colnames(x)[(VS==T)&(vary==F)],round((matrix(cv.bic.cs$r$gamma,ncol=K))[(VS==T)&(vary==F),1],3))
# varying effects
cbind(colnames(x),judge,round(matrix(cv.bic.cs$r$gamma,ncol=K),3))[VS,]

# VS <- vary
betahat <- t(betahat[VS,])

# variance
p <- dim(x)[2]
covfunc <- function(taui, estcov){
  betacov <- NULL
  for(kk in 1:p){
    betacov <- c(betacov, drop(matrix(bf(taui),nrow=1) %*% estcov[(0:(K-1))*p+kk,(0:(K-1))*p+kk] %*% bf(taui)))
  }
  return(betacov)
}
betacov <- (sapply(TAU, covfunc, cv.bic.cs$r$cov))[VS,]
n <- dim(x)[1]
CI <- t(qnorm(0.975) * sqrt(betacov))
colnames(CI) <- paste0("CI_",colnames(x))

plotdata <- data.frame(cbind(TAU, betahat, CI))
colnames(plotdata)[1] <- "tau"

library(tidyr) # pivot_longer
library(dplyr) # mutate
longbeta <- plotdata %>%
  pivot_longer(
    cols = Africa:smoking,
    names_to = "betai",
    values_to = "betahat"
  ) %>% mutate(
    ymin = case_when(
      betai == "Africa" ~ betahat - CI_Africa,
      betai == "Americas" ~ betahat - CI_Americas,
      betai == "Eastern_Mediterranean" ~ betahat - CI_Eastern_Mediterranean,
      betai == "Europe" ~ betahat - CI_Europe,
      betai == "SouthEast_Asia" ~ betahat - CI_SouthEast_Asia,
      betai == "Western_Pacific" ~ betahat - CI_Western_Pacific,
      betai == "water" ~ betahat - CI_water,
      betai == "HIV_rate" ~ betahat - CI_HIV_rate,
      betai == "overweight" ~ betahat - CI_overweight,
      betai == "internet" ~ betahat - CI_internet,
      betai == "DPT" ~ betahat - CI_DPT,
      betai == "GDP" ~ betahat - CI_GDP,
      betai == "health" ~ betahat - CI_health,
      betai == "alcohol" ~ betahat - CI_alcohol,
      betai == "CO2" ~ betahat - CI_CO2,
      betai == "school" ~ betahat - CI_school,
      betai == "forest" ~ betahat - CI_forest,
      betai == "temperature" ~ betahat - CI_temperature,
      betai == "smoking" ~ betahat - CI_smoking,
      betai == "energy" ~ betahat - CI_energy
    ),
    ymax = case_when(
      betai == "Africa" ~ betahat + CI_Africa,
      betai == "Americas" ~ betahat + CI_Americas,
      betai == "Eastern_Mediterranean" ~ betahat + CI_Eastern_Mediterranean,
      betai == "Europe" ~ betahat + CI_Europe,
      betai == "SouthEast_Asia" ~ betahat + CI_SouthEast_Asia,
      betai == "Western_Pacific" ~ betahat + CI_Western_Pacific,
      betai == "water" ~ betahat + CI_water,
      betai == "HIV_rate" ~ betahat + CI_HIV_rate,
      betai == "overweight" ~ betahat + CI_overweight,
      betai == "internet" ~ betahat + CI_internet,
      betai == "DPT" ~ betahat + CI_DPT,
      betai == "GDP" ~ betahat + CI_GDP,
      betai == "health" ~ betahat + CI_health,
      betai == "alcohol" ~ betahat + CI_alcohol,
      betai == "CO2" ~ betahat + CI_CO2,
      betai == "school" ~ betahat + CI_school,
      betai == "forest" ~ betahat + CI_forest,
      betai == "temperature" ~ betahat + CI_temperature,
      betai == "smoking" ~ betahat + CI_smoking,
      betai == "energy" ~ betahat + CI_energy
    )
  ) %>%
  dplyr::select(betai, tau, betahat, ymin, ymax)

longbeta <- as.data.frame(longbeta)
longbeta$betahat <- as.numeric(longbeta$betahat)
longbeta$ymax <- as.numeric(longbeta$ymax)
longbeta$ymin <- as.numeric(longbeta$ymin)
longbeta$tau <- as.numeric(longbeta$tau)

# plots for cqr beta
library(reshape2)
long_df_reshape2 <- melt(BETA, id.vars = "tau")
colnames(long_df_reshape2) <- c("tau","betai","betahat")
long_df_reshape2 <- long_df_reshape2[,c(2,1,3)]
long_df_reshape2 <- cbind(long_df_reshape2,ymax=NA,ymin=NA);
long_df_reshape2$ymax <- long_df_reshape2$ymin <- NA
long_df_reshape2$Method <- "LQR"

longbeta$Method <- "pLPQR"
longdata0 <- data <- rbind(longbeta,long_df_reshape2)
data <- data %>% filter(!(betai %in% c("Africa", "SouthEast_Asia", "Americas", "Europe", "Western_Pacific", "Eastern_Mediterranean")))
longbeta1 <- longbeta %>% filter(!(betai %in% c("Africa", "SouthEast_Asia", "Americas", "Europe", "Western_Pacific", "Eastern_Mediterranean")))
data$betai <- as.factor(data$betai)

betahat <- betahat[,order(abs(betahat[81,]),decreasing=T)] # tau=0.9
varorder <- colnames(betahat)[!(colnames(betahat) %in% c("Africa", "SouthEast_Asia", "Americas", "Europe", "Western_Pacific", "Eastern_Mediterranean"))]
longbeta1$betai <- factor(longbeta1$betai,levels=varorder)
data$betai <- factor(data$betai,levels=varorder)

library(ggplot2)
class(data$tau) <- "numeric"
mycolors <- c("#f29fcb","#031b7c")
base <- ggplot(data=data, aes(x=tau, y=betahat,
                              ymin=ymin, ymax=ymax,
                              group=Method,color=Method,linetype=Method,linewidth=Method)) +
  geom_line() +
  geom_ribbon(data=longbeta1,alpha=0.5, fill="#BEE2FF", linetype = "longdash",size=0.5,color="#BEE2FF") +
  facet_wrap(~betai,ncol=3, scales="free_y", labeller = label_parsed) +
  scale_color_manual(values=mycolors, labels = c("LQR" = "LQR", "pLPQR" = "pLPQR")) +
  scale_linetype_manual(values = c("LQR" = 1, "pLPQR" = 1), labels = c("LQR" = "LQR", "pLPQR" = "pLPQR")) +
  scale_linewidth_manual(values = c("LQR" = 1.1, "pLPQR" = 1.5), labels = c("LQR" = "LQR", "pLPQR" = "pLPQR")) +
  xlab(as.expression(expression(tau))) +
  ylab(as.expression(expression( paste(beta[1],"(", tau, ")") ))) +
  scale_y_continuous(n.breaks=4, minor_breaks = NULL
  ) +
  scale_x_continuous(n.breaks=5, minor_breaks = NULL) + theme_minimal() +
  labs(x=expression(tau),y=expression(bold(beta(tau)))) +
  theme(text = element_text(size=15),
        strip.text = element_text(size=15, margin = margin(),face="bold"),
        axis.title=element_text(size=15,face="bold.italic"),
        axis.text=element_text(size=15),
        legend.title = element_text(size=15),legend.text = element_text(size=15),legend.key.size = unit(1.3, 'cm'),
        panel.grid = element_line(color = "grey", size = 0.35, linetype = 1))
base

design <- c("
ABC
DEF
GHI
JKL
MNO
"
)
library("ggh4x")
base + ggh4x::facet_manual(~betai, scales = "free", design = design)



# ===== Figure 5. A three-dimensional plot displays the estimated life expectancy against the two most influential determinants ===========
tau <- 0.9

data3d <- data.frame(
  x = variables$water,
  y = variables$HIV_rate
)
ymean <- mean(log(variables$life))
ysd <- sd(log(variables$life))
data3d$z <- drop(exp( ( x %*% thetahat %*% bf(tau) )*ysd + ymean))

data3d <- data3d[!duplicated(data3d[, c("x", "y")]), ]
data3d_sorted <- data3d[order(data3d[,1]), ]

# ## linear interpolation
library(akima)
interp_data <- with(data3d, {
  akima::interp(x = data3d$x,  # x1
         y = data3d$y,  # x2
         z = data3d$z,  # yhat
         linear = T,
         duplicate = "mean",
         xo = seq(min(data3d$x), max(data3d$x), length = 50),
         yo = seq(max(data3d$y), min(data3d$y), length = 50))
})

min_z <- min(data3d$z, na.rm = TRUE)
max_z <- max(data3d$z, na.rm = TRUE)
tick_positions <- seq(0, 1, 0.25)
tick_values <- min_z + tick_positions * (max_z - min_z)

morandi_palettes <- list(
  "夏日海滩2" = c("#FC757B","#F97F5F","#FAA26F","#FDCD94","#FEE199","#B0D6A9","#65BDBA","#3C9BC9","#2686b5","#057dcd"),# "#71b8ed",,"#1e3d58"
)


selected_palette <- morandi_palettes[["夏日海滩2"]]
morandi_colors <- colorRampPalette(selected_palette)(100)

library(dplyr) # %>%
library(httr) # config()
library(plotly)
plot_ly() %>%
  add_surface(
    x = interp_data$x,
    y = interp_data$y,
    z = t(interp_data$z),
    colorscale = list(seq(0, 1, length.out = 100), morandi_colors),
    colorbar = list(
      title = "",
      tickfont = list(size = 14, weight = "bold"),
      y = 0.5,
      yanchor = "middle",
      x = 1.05,
      xanchor = "left",
      len = 0.4,
      thickness = 18,
      outlinewidth = 0
    ),
    contours = list(
      z = list(
        show = TRUE,
        width = 40,
        usecolormap = TRUE,
        project = list(
          z = TRUE,
          opacity = 0.95),
        highlight = TRUE,
        highlightcolor = "white",
        highlightwidth = 15
      )
    )
  ) %>%
  layout(
    scene = list(
      xaxis = list(
        title = list(
          text = "Water",
          font = list(size = 18, weight = "bold")
        ),
        showspikes = FALSE,
        showline = FALSE,
        zeroline = FALSE,
        nticks = 5,
        tickfont = list(size = 13, weight = "bold")
      ),
      yaxis = list(
        title = list(
          text = "HIV rate",
          font = list(size = 18, weight = "bold")
        ),
        showspikes = FALSE,
        showline = FALSE,
        zeroline = FALSE,
        tickfont = list(size = 13, weight = "bold")
      ),
      zaxis = list(
        title = "",
        range = c(min_z, max_z),
        showspikes = FALSE,
        showline = FALSE,
        zeroline = FALSE,
        tickfont = list(size = 13, weight = "bold")
      ),
      camera = list(eye = list(x = 1, y = 2, z = 1))
    ),
    annotations = list(
      list(
        x = 0.1,
        y = 0.67,
        xref = "paper",
        yref = "paper",
        text = "$\\hat{Q}_{0.9}(y)$",
        showarrow = FALSE,
        font = list(size = 24, color = "#333333", weight = "bold"),
        textangle = 0,
        xanchor = "right",
        yanchor = "middle"
      ),
      list(
        x = 1.05, y = 0.73,
        xref = "paper",
        yref = "paper",
        text = "$\\hat{Q}_{0.9}(y)$",
        showarrow = FALSE,
        font = list(size = 20, color = "#333333", weight = "bold"),
        textangle = 0,
        xanchor = "left",
        yanchor = "middle"
      )
    ),
    margin = list(l = 80, r = 80, b = 50, t = 50)
  ) %>%
  config(mathjax = "cdn")



# ======== Figure 2. The regional variation in life expectancy ==========
library(tidyverse)
library(lubridate)
library(showtext)
library(patchwork)
library(ragg)
library(maps)

# region <- read.csv("alcohol2.csv")
# region <- subset(region,(Dim1=="All types")&(Period==2019))[,c(3,5,7)]
# colnames(region) <- c("region","iso_a3","year")
# region$iso_a3 <- as.character(region$iso_a3)
region <- data.frame(region=variables$region, iso_a3=variables$countrycode, year=variables$year)
# obtain the global map
library(rnaturalearth)
world <- ne_countries(scale = "medium", returnclass = "sf")
world_who <- world %>%
  left_join(region, by = "iso_a3") %>%
  mutate(region = ifelse(is.na(region), "Not classified", region))
world_who <- world_who[!(world_who$iso_a3 %in% "ATA"),]

world_who[world_who$formal_en%in%c("Hong Kong Special Administrative Region, PRC","Macao Special Administrative Region, PRC","Territory of Ashmore and Cartier Islands","Territory of Norfolk Island","Territory of Heard Island and McDonald Islands","New Caledonia","French Polynesia","Wallis and Futuna Islands","Pitcairn, Henderson, Ducie and Oeno Islands","Republic of the Marshall Islands","Commonwealth of the Northern Mariana Islands","Republic of the Marshall Islands","Territory of Guam","Republic of Palau"),
          "region"] <- "Western Pacific"
world_who[world_who$formal_en%in%c("French Republic","Turkish Republic of Northern Cyprus","Føroyar Is. (Faeroe Is.)","Territory of the French Southern and Antarctic Lands","Saint-Martin (French part)","Saint Pierre and Miquelon","Republic of Kosovo","Greenland","Principality of Liechtenstein","Principality of Monaco","State of the Vatican City","Bailiwick of Guernsey","Republic of San Marino","Kingdom of Norway"),
          "region"] <- "Europe"
world_who[world_who$formal_en%in%c("Sint Maarten (Dutch part)","Saint-Barthélemy","Curaçao","Turks and Caicos Islands","British Virgin Islands","The Bermudas or Somers Isles","Cayman Islands","American Samoa","Virgin Islands of the United States","Commonwealth of Puerto Rico","Aruba"),
          "region"] <- "Americas"
world_who[world_who$formal_en%in%c("Republic of South Sudan","Sahrawi Arab Democratic Republic"),
          "region"] <- "Africa"
world_who[world_who$formal_en%in%c("Republic of the Sudan","West Bank and Gaza"),
          "region"] <- "Eastern Mediterranean"
world_who[world_who$formal_en%in%c("Åland Islands","Falkland Islands","Bailiwick of Jersey","South Georgia and the Islands","Republic of Somaliland"),
          "region"] <- "Not classified"
world_who$formal_en[(world_who$region%in%"Not classified")&(!is.na(world_who$formal_en))]
world_who[(world_who$region%in%"Not classified"),]

world_who[world_who$formal_en%in%"French Republic","geometry"]
world_who$iso_a3 %in% "GUF"
world_who$iso_a3 %in% "ATA" # antarctica

sci_colors <- c(
  "Eastern Mediterranean" = "#5b9ad5",
  "Western Pacific" = "#65BDBA",
  "Europe" = "#B0D6A9",
  "Americas" = "#FEE199",
  "South-East Asia" = "#f29f52",
  "Africa" = "#FC757B",
  "Not classified" = "#CCCCCC"
)

countries_lines <- tibble(x =    c(43, 100,110,79, 20, -60),
                          xend = c(0,  120,185,120,0,  -120),
                          y =    c(26, 60, 35, 22, -10,-10),
                          yend = c(104,104,30, -53,-53,-15)
)

# static map
static_map <- ggplot(world_who) +
  geom_sf(aes(fill = region), color = "white", size = 0.1) +
  scale_fill_manual(values = sci_colors, name = "WHO Region",
                    breaks = c("Africa", "South-East Asia", "Americas", "Europe", "Western Pacific", "Eastern Mediterranean",
                               "Not classified"),
                    labels = c("Africa", "South-East Asia", "Americas", "Europe", "Western Pacific", "Eastern Mediterranean",
                               "Not classified")) +
  geom_segment(data = countries_lines, aes(x = x, xend = xend, y = y , yend = yend), color = "grey50", inherit.aes = FALSE) +
  scale_x_continuous(limits = c(-240, 250), expand = c(0,0)) +
  scale_y_continuous(limits = c(-130, 180)) +
  theme_minimal() + theme_void() +
  theme(
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background = element_rect(fill = "white", colour = NA),
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 10),
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    plot.caption = element_text(hjust = 0.5, size = 10),
    plot.margin = margin(10, 10, 60, 10)
  ) +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE))
static_map

create_method_legend <- function(){
  ggplot() +
    annotate("text", x = 1.8, y = 1, label = "Method", size = 4, fontface = "bold", hjust = 0) +
    annotate("rect", xmin = 2.5, xmax = 2.8, ymin = 0.85, ymax = 1.15,
             fill = "#BEE2FF", alpha = 0.5, color = "#BEE2FF", linetype = "longdash", size = 0.3) +
    annotate("segment", x = 2.5, xend = 2.8, y = 1, yend = 1,
             color = "#031b7c", size = 0.7, linetype = "solid") +
    annotate("text", x = 3.1, y = 1, label = "pLPQR", size = 4) +
    annotate("segment", x = 3.5, xend = 3.8, y = 1, yend = 1,
             color = "#f29fcb", size = 0.7, linetype = "solid") +
    annotate("text", x = 4.1, y = 1, label = "LQR", size = 4) +
    xlim(0, 6) + ylim(0.5, 1.8) +
    theme_void() +
    theme(plot.background = element_blank(),
          plot.margin = margin(1, 1, 1, 1))
}

method_legend <- create_method_legend()

static_map +
  inset_element(method_legend, left = 0.1, bottom = -0.04, right = 0.9, top = -0.14)

chart <- function(rn, longdata0){
  longbetai <- longdata0 %>% filter(betai == rn)
  longbetai$tau <- as.numeric(longbetai$tau)
  longbetai %>% ggplot(aes(x=tau, y=betahat,
                       ymin=ymin, ymax=ymax,
                       group=Method, color=Method, linetype=Method, linewidth=Method)) +
    geom_line() +
    geom_ribbon(alpha=0.3, fill="#BEE2FF", linetype = "longdash", size=0.5, color="#BEE2FF") +
    scale_color_manual(values=mycolors) +
    scale_linetype_manual(values = c("LQR" = 1, "pLPQR" = 1)) +
    scale_linewidth_manual(values = c("LQR" = 0.7, "pLPQR" = 0.7)) +
    xlab(expression(tau)) +
    ylab(expression(paste(beta[1],"(", tau, ")"))) +
    scale_y_continuous(n.breaks=4, minor_breaks = NULL) +
    scale_x_continuous(n.breaks=5, minor_breaks = NULL) +
    theme_minimal() +
    labs(x=expression(tau), y=expression(bold(beta(tau))), title = rn) +
    theme(
      text = element_text(size=10, family = "sans"),
      axis.title = element_text(size=10, face="bold"),
      axis.text = element_text(size=10, color = "black"),
      legend.position = "none",
      panel.grid.major = element_line(color = "grey90", size = 0.2),
      panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = NA, colour = NA),
      panel.background = element_rect(fill = NA, colour = NA),
      panel.border = element_blank(),
      plot.margin = margin(10, 10, 10, 10),
      axis.line = element_line(color = "black", size = 0.3),
      axis.ticks = element_line(color = "black", size = 0.3)
    )
}

final <- static_map +
  inset_element(chart("Eastern_Mediterranean", longdata0),
    left = 0.3, bottom = 0.72, right = 0.6, top = 1.02) +
  inset_element(chart("Europe", longdata0),
    left = 0.63, bottom = 0.72, right = 0.93, top = 1.02) +
  inset_element(chart("Western_Pacific", longdata0),
                left = 0.85, bottom = 0.38, right = 1.15, top = 0.68) +
  inset_element(chart("SouthEast_Asia", longdata0),
                left = 0.63, bottom = -0.02, right = 0.93, top = 0.28) +
  inset_element(chart("Africa", longdata0),
                left = 0.3, bottom = -0.02, right = 0.6, top = 0.28) +
  inset_element(chart("Americas", longdata0),
                left = -0.03, bottom = 0.3, right = 0.27, top = 0.6) +
  inset_element(method_legend,
                left = 0.1, bottom = -0.04, right = 0.9, top = -0.14)
final
# ggsave(paste0("map-", format(Sys.time(), "%Y%m%d_%H%M%S"), ".png"),
#        plot = final, device = agg_png(width = 25, height = 15, units = "in", res = 800))



# ======= Figure 4. Quantile dynamics of life expectancy determinants’ effects and category-level contributions =============
# devtools::install_github("davidsjoberg/ggsankey")
library(ggsankey)
library(tidyverse)
library(shadowtext)

betahat0 <- betahat

betahat <- t(betahat)
betahat <- as.data.frame(betahat)
mat <- data.frame(variable=c("Africa","Western_Pacific","Eastern_Mediterranean","SouthEast_Asia","Europe","Americas",
                             "water","HIV_rate","overweight","internet","DPT",
                             "GDP","health","alcohol","CO2","school","forest",
                             "temperature","smoking","energy"),
                  category=c("R","R","R","R","R","R",
                             "S","H","L","S","H",
                             "S","H","L","E","S","E",
                             "E","L","S"))

betahat_long <- betahat %>%
  rownames_to_column("tau") %>%
  pivot_longer(
    cols = -tau,
    names_to = "variable",
    values_to = "value"
  ) %>%
  mutate(
    value = value,
    tau = as.numeric(tau),
    tau_factor = factor(tau),
    abs_value = abs(value) * 2  # rescale
  )

rank_data <- betahat_long %>%
  left_join(mat, by = c("variable" = names(mat)[1]))

names(rank_data)[names(rank_data) == names(mat)[2]] <- "category"

rank_data_filtered <- rank_data %>%
  filter(category != "R") %>%
  mutate(
    original_value = value,
    value_sign = ifelse(original_value < 0, "negative", "positive")
  )

rank_data_filtered$category <- factor(rank_data_filtered$category,
                                      levels=c("E","L","H","S"),
                                      labels=c("Environment","Lifestyle","Healthcare","Socio-economics"))

p <- ggplot(rank_data_filtered,
            aes(x = tau_factor,
                node = variable,
                fill = category,
                value = abs_value,
                label = variable)) +
  geom_sankey_bump(
    space = 0.1,
    type = "alluvial",
    smooth = 5,
    alpha = 0.7,
    color = NULL,
    linewidth = 0.01,
    force = 0.1,
    tension = 0.8
  ) +
  scale_fill_manual(
    values = c(
      "Healthcare" = "#FDCD94",
      "Socio-economics" = "#FC757B",
      "Lifestyle" = "#65BDBA",
      "Environment" = "#057dcd"
    ),
    na.value = "grey50"
  ) +
  scale_x_discrete(breaks = seq(0.1,0.9,0.1)) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.1))) +
  labs(x = as.expression(expression( tau )), y = NULL, fill = "Category") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    plot.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    axis.text.y = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5,size = 17),
    axis.title.x = element_text(size = 18),
    axis.title.y = element_blank(),
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(color = "grey40", size = 18),
    legend.title = element_text(face = "bold")
  )
p

p_built <- ggplot_build(p)
bump_data <- p_built$data[[1]]

get_actual_positions <- function(bump_data, target_x) {
  bump_data %>%
    filter(near(x, target_x, tol = 0.01)) %>%
    group_by(label) %>%
    summarise(
      y_actual = mean(y, na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    rename(variable = label)
}

x_range <- range(bump_data$x)
first_x <- x_range[1]
last_x <- x_range[2]

start_actual <- get_actual_positions(bump_data, first_x)
end_actual <- get_actual_positions(bump_data, last_x)

g_labs_start_actual <- start_actual %>%
  mutate(x_display = first_x - 0.6)

g_labs_end_actual <- end_actual %>%
  mutate(x_display = last_x + 0.6)

plot_with_labels <- p +
  geom_shadowtext(data = g_labs_start_actual,
    aes(x = x_display, y = y_actual, label = variable),
    inherit.aes = FALSE, hjust = 1, bg.color = "white", color = "black", size = 6) +
  geom_shadowtext(
    data = g_labs_end_actual,
    aes(x = x_display, y = y_actual, label = variable),
    inherit.aes = FALSE,hjust = 0, bg.color = "white", color = "black", size = 6) +
  coord_cartesian(clip = "off") +
  theme(plot.margin = margin(0, 92, 0, 93)) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 18),
    legend.title = element_text(size = 18, face = "bold"),
    plot.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 0),
    axis.text.y = element_blank(),
    axis.title.y = element_blank()
  )

print(plot_with_labels)



stacked_ribbon_data <- rank_data_filtered %>%
  group_by(tau_factor, category) %>%
  summarise(
    total_abs = sum(abs_value, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  group_by(tau_factor) %>%
  mutate(
    total_sum = sum(total_abs, na.rm = TRUE),
    proportion = total_abs / total_sum,
    cumulative_prop = cumsum(proportion),
    ymin = lag(cumulative_prop, default = 0),
    ymax = cumulative_prop
  ) %>%
  ungroup()

category_order <- stacked_ribbon_data %>%
  filter(tau_factor == "0.9") %>%
  arrange(proportion) %>%  # arrange(desc(proportion)) %>%
  pull(category)

stacked_ribbon_data <- stacked_ribbon_data %>%
  mutate(category = factor(category, levels = category_order)) %>%
  group_by(tau_factor) %>%
  arrange(tau_factor, category) %>%
  mutate(
    cumulative_prop = cumsum(proportion),
    ymin = lag(cumulative_prop, default = 0),
    ymax = cumulative_prop
  ) %>%
  ungroup()

desired_taus <- c("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9")

color_values <- setNames(
  c("#057dcd", "#65BDBA", "#FDCD94", "#FC757B"),
  category_order
)

enhanced_stacked_plot <- ggplot(stacked_ribbon_data,
                                aes(x = as.numeric(tau_factor))) +
  geom_ribbon(
    aes(ymin = ymin,
        ymax = ymax,
        fill = category),
    alpha = 0.8,
    color = "white",
    linewidth = 0.3
  ) +

  geom_text(
    data = stacked_ribbon_data %>%
      group_by(category) %>%
      filter(tau_factor == 0.5),
    aes(x = as.numeric(tau_factor),
        y = (ymin + ymax) / 2,
        label = paste0(category, "\ \ ", round(proportion * 100, 1), "%")),
    color = "white",
    fontface = "bold",
    size = 5,
    lineheight = 0.8
  ) +

  geom_text(
    data = stacked_ribbon_data %>%
      filter(tau_factor == 0.1),
    aes(x = as.numeric(tau_factor) - 1,
        y = (ymin + ymax) / 2,
        label = paste0(round(proportion * 100, 1), "%")),
    color = "black",
    size = 5,
    hjust = 1
  ) +
  geom_text(
    data = stacked_ribbon_data %>%
      filter(tau_factor == 0.9),
    aes(x = as.numeric(tau_factor) + 1,
        y = (ymin + ymax) / 2,
        label = paste0(round(proportion * 100, 1), "%")),
    color = "black",
    size = 5,
    hjust = 0) +
  geom_hline(yintercept = seq(0, 1, by = 0.25),
             color = "white",
             alpha = 0.3,
             linewidth = 0.2) +
  scale_fill_manual(values = color_values) +
  scale_x_continuous(
    breaks = which(levels(stacked_ribbon_data$tau_factor) %in% c("0.1","0.3","0.5","0.7","0.9")),
    labels = c("0.1","0.3","0.5","0.7","0.9")
  ) +
  scale_y_continuous(labels = NULL) +
  labs(x = expression(tau), y = NULL) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    plot.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5,size = 17, vjust=0.1),
    axis.title.x = element_text(size = 18),
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(color = "grey40", size = 18)
  ) +
  coord_cartesian(clip = "off" )+ #
  theme(plot.margin = margin(23, 28, 43, 18))

# install.packages("gridExtra")
library(gridExtra)
combined_plot <- grid.arrange(plot_with_labels, enhanced_stacked_plot,
                              ncol = 2,  widths = c(1.8, 1))
print(combined_plot)

