
#######################################
#########    MAIN ANALYSIS    #########   
#######################################

##### Load data ##### ##### ##### ##### ##### 
  D2 <- read.csv("LTBI Review Data_Clean_10-18-2017.csv")[,-1]

##### Functions ##### ##### ##### ##### ##### 
  # Model for Types A through I
  mod <- function(a=0,b=0,c=0,d=0,e=0,f=0,n=0,yrs=10,py=100) {
    if( length(d)>1 & length(d)!=n) {
      stop("n different to length(d)") 
    }
    c   <- c/py
    d   <- d/py
    e   <- e/py
    f   <- f/py
    Ls1 <- Ls0 <- 1-a-b
    I1  <- I0  <- a
    Lf1 <- Lf0 <- b
    pI  <- rep(NA,yrs+1)
    ##
    for(y in 1:yrs) { # y=2
      pI[y] <- I1
      if(n>0){ 
        if(y==n+1){  
          Ls0 <-  Ls1 <-  Ls1+Lf1
          Lf0 <- Lf1 <- 0
        }
        if(y<n+1){ 
          di <- d[y]
        }
      } else { 
        di = d
      }
      for(m in 1:py) {
        Ls1 <- Ls0*(1-c-f) + Lf0*e
        Lf1 <- Lf0*(1-di-e) + Ls0*f
        I1  <- I0 + Ls0*c  + Lf0*di
        I0  <- I1
        Ls0 <- Ls1
        Lf0 <- Lf1
      }
    }
    pI[y+1] <- I1
    return(pI)
  }
  
  # test it works
  mod(a=0,b=1,c=0.5,d=rep(0.1,5),e=0,f=0,n=5,yrs=20,py=100)
  
  # Model for Type J
  modJ <- function(c=NA,f=NA,yrs=10,py=100) {
    c  <- c/py
    f  <- f/py
    L1 <- L0 <- rep(0,length(f)+1)
    L1[1] <- L0[1] <- 1
    I1  <- I0  <- 0
    pI  <- rep(NA,yrs+1)
    ii <- length(L1)
    ##
    for(y in 1:yrs) { # y=2
      pI[y] <- I1
      for(m in 1:py) { # m=1
        if(ii==2) {
          L1[1] <- L0[1]*(1-f[1]) 
          L1[2] <- L0[2]*(1-c   )+L0[1]*f[1]
        }
        if(ii==3) {
          L1[1] <- L0[1]*(1-f[1]) 
          L1[2] <- L0[2]*(1-f[2])+L0[1]*f[1]    
          L1[3] <- L0[3]*(1-c   )+L0[2]*f[2]
        }
        if(ii==4) {
          L1[1] <- L0[1]*(1-f[1]) 
          L1[2] <- L0[2]*(1-f[2])+L0[1]*f[1]    
          L1[3] <- L0[3]*(1-f[3])+L0[2]*f[2]     
          L1[4] <- L0[4]*(1-c   )+L0[3]*f[3]
        }
        I1 <- I0 + L0[ii]*c
        I0 <- I1
        L0 <- L1
      }
    }
    pI[y+1] <- I1
    return(pI)
  }
  
  # test it works
  modJ(c=0.5,f=c(1,1,1),yrs=20,py=100)
  
  # Model for Type K
  modK <- function(x1=0,x2=0,yrs=10,py=100) {
    ny <- seq(0,yrs,length.out=py*yrs+1)[-1]
    ny[ny<1] <- 1
    ct <- x1 * ny^x2 / py
    pI <- c(0,(1-cumprod(1-ct))[1:yrs*py])
    return(pI)
  }
  
  # test it works
  modK(x1=0.0001,x2=-1.1,yrs=10,py=365)
  
  # Model for Type L
  modL <- function(x1=0,x2=0,x3=0,yrs=10,py=100) {
    ny <- seq(0,yrs,length.out=py*yrs+1)[-1]
    ct <- x1*(x2+exp(-x3*ny)) / py
    pI <- c(0,(1-cumprod(1-ct))[1:yrs*py])
    return(pI)
  }
  
  # test it works
  modL(x1=0.1,x2=0.1,x3=0.1,yrs=10,py=365)

##### Estimate results ##### ##### ##### ##### ##### 
  y1  <- 20 # no years
  py1 <- 365  # timesteps per year

  CumInc  <- matrix(NA,nrow(D2),y1+1)
  IncRate <- matrix(NA,nrow(D2),y1)

  # All 
  CumInc  <- matrix(NA,nrow(D2),y1+1)
  IncRate <- matrix(NA,nrow(D2),y1)
  
  for(i in 1: nrow(D2)) { # i=1
    z <- list()
    z$n = D2$n_tunnel_d[i]
    if(D2$model_structure[i]%in%c("A","B","C","D","E","F","G","H","I")) {
      if(D2$model_structure[i]=="A") {
        z$a =  0      ; z$b =  0      ; z$c =  D2$c[i]; z$d =  0      ; z$e =  0      ; z$f =  0      ;
      }
      if(D2$model_structure[i]=="B") { 
        z$a =  0      ; z$b =  1      ; z$c =  D2$c[i]; z$d =  as.numeric(D2[i,paste("d",1:(D2$n_tunnel_d[i]),sep="")]); z$e =  0      ; z$f =  0      ;
      }
      if(D2$model_structure[i]=="C") { 
        z$a =  ifelse(is.na(D2$a[i]),0,D2$a[i]);
        z$b =  1      ; z$c =  D2$c[i]; z$d =  D2$d[i]; z$e =  D2$e[i]; z$f =  0      ;
      }
      if(D2$model_structure[i]=="D") { 
        z$a =  1      ; z$b =  0      ; z$c =  0      ; z$d =  0      ; z$e =  0      ; z$f =  0      ;
      }
      if(D2$model_structure[i]=="E") { 
        z$a =  D2$a[i]; z$b =  0      ; z$c =  D2$c[i]; z$d =  0      ; z$e =  0      ; z$f =  0      ;
      }
      if(D2$model_structure[i]=="F") { 
        z$a =  0      ; z$b =  D2$b[i]; z$c =  D2$c[i]; z$d =  D2$d[i]; z$e =  0      ; z$f =  0      ;
      }
      if(D2$model_structure[i]=="G") { 
        z$a =  D2$a[i]; z$b =  1-z$a  ; z$c =  D2$c[i]; z$d =  D2$d[i]; z$e =  D2$e[i]; z$f =  0      ;
      }
      if(D2$model_structure[i]=="H") { 
        z$a =  0      ; z$b =  D2$b[i]; z$c =  0      ; z$d =  D2$d[i]; z$e =  0      ; z$f =  D2$f[i];
      }
      if(D2$model_structure[i]=="I") { 
        z$a =  0      ; z$b =  1      ; z$c =  0      ; z$d =  D2$d[i]; z$e =  D2$e[i]; z$f =  D2$f[i];
      }
      CumInc[i,]  <- mod(a=z$a, b=z$b, c=z$c, d=z$d, e=z$e, f=z$f, n=z$n, yrs=y1, py=py1)
    }
    if(D2$model_structure[i]=="J") {
      z$c =  D2$c[i]; z$f =  as.numeric(D2[i,paste("f",1:(D2$n_tunnel_f[i]),sep="")]) 
      CumInc[i,]  <-  modJ(c=z$c, f=z$f, yrs=y1, py=py1)
    }
    if(D2$model_structure[i]=="K") {
      z$x1 =  D2$x1[i]; z$x2 =  D2$x2[i]; 
      CumInc[i,]  <- modK(x1=z$x1,x2=z$x2,yrs=y1,py=py1)
    }
    if(D2$model_structure[i]=="L") {
      z$x1 =  D2$x1[i]; z$x2 =  D2$x2[i]; z$x3 =  D2$x3[i]; 
      CumInc[i,]  <- modL(x1=z$x1, x2=z$x2, x3=z$x3, yrs=y1, py=py1)
    }
    IncRate[i,] <- diff(c(0,CumInc[i,-1]))/(1-c(0,CumInc[i,-c(1,ncol(CumInc))]))
  }

##### Fix some issues with individual studies ##### ##### ##### ##### ##### 
# Structure D
  IncRate[D2$model_structure=="D",-1] <- NA
  
# Structure A
  IncRate[69 ,-1] <- 1-(1-D2[69,"c"]/py1)^py1
  IncRate[93 ,-1] <- 1-(1-D2[93,"c"]/py1)^py1
  IncRate[94 ,-1] <- 1-(1-D2[94,"c"]/py1)^py1
  IncRate[115,-1] <- 1-(1-D2[115,"c"]/py1)^py1
  IncRate[471,-1] <- 1-(1-D2[471,"c"]/py1)^py1
  IncRate[655,-1] <- 1-(1-D2[655,"c"]/py1)^py1
  IncRate[656,-1] <- 1-(1-D2[656,"c"]/py1)^py1
  IncRate[657,-1] <- 1-(1-D2[657,"c"]/py1)^py1
  
# Structure E
  IncRate[620,-1] <- 1-(1-D2[620,"c"]/py1)^py1
  IncRate[621,-1] <- 1-(1-D2[621,"c"]/py1)^py1
  
##### Organize results ##### ##### ##### ##### ##### 
  vID2 <- list(rep(T,length(D2$standard)),         # all
               D2$standard==1,                     # all std risk, adult
               D2$elevated_risk==1,                # elevated risk
               D2$children==1,                     # kids
               D2$hiv==1,                          # hiv
               D2$early_hiv==1,                    # early hiv
               D2$advanced_hiv==1,                 # advanced HIV
               D2$art==1,                          # ART
               D2$high_burden==1 & D2$standard==1, # high-burden
               D2$low_burden==1  & D2$standard==1, # low-burden
               D2$no_cites==1 & D2$standard==1,    # No cites, std
               D2$no_cites==0 & D2$standard==1,    # No cites, non-std  
               D2$year<=2010 & D2$standard==1,     # Prior to 2010
               D2$year> 2010 & D2$standard==1 )    # 2010 onwards

  vCat <- list("All Studies All Strata",
               "Adults, No Individual Risk Modifiers",
               "Any Individual Elevated Risk",
               "Children",
               "All HIV",
               "Early HIV",
               "Advanced HIV",
               "ART",
               "High Burden Setting, No Individual Risk Modifiers",
               "Low Burden Setting, No Individual Risk Modifiers",
               "No Citations Given for Progression Parameters",
               ">0 Citations Given for Progression Parameters",
               "Publication year 2010 or before",
               "Publication year after 2010")

  save(IncRate,CumInc,vID2,vCat,D2,file="Reactivation_model_results_10-18-2017.rData")

### Risk ratios for named risk groups ##### ##### ##### ##### ##### 
  AuthorID <- unique(D2[,"Unique"])

  ### INFANT 
  RRinf1_inc <- RRinf20 <- RRinf2 <- NULL
  for(i in 1:length(AuthorID)) { # i=2
    id <- D2$Unique==AuthorID[i] & (D2$infants==1 | D2$standard==1) 
    zz <- D2[id ,]
    if(sum(zz$infants==0)*sum(zz$infants==1)>0) {
      z2         <- CumInc[id,]
      RRinf2     <- c(RRinf2,mean(z2[zz$infants==1,3])  / mean(z2[zz$infants==0,3]))
      RRinf20    <- c(RRinf20,mean(z2[zz$infants==1,21])  / mean(z2[zz$infants==0,21]))
      z2         <- IncRate[id,]
      RRinf1_inc <- c(RRinf1_inc,mean(z2[zz$infants==1,1])  / mean(z2[zz$infants==0,1])) 
    }
  }
  ### CHILD
  RRchild1_inc <- RRchild20 <- RRchild2 <- NULL
  for(i in 1:length(AuthorID)) { # i=2
    ch <- as.numeric(D2$children==1 & D2$infants==0)
    id <- D2$Unique==AuthorID[i] & (ch==1 | D2$standard==1)
    zz <- ch[id]
    if(sum(zz==1 )*sum(zz==0)>0) {
      z2           <- CumInc[id ,]
      RRchild2     <- c(RRchild2,mean(z2[zz==1,3])  / mean(z2[zz==0,3]))
      RRchild20    <- c(RRchild20,mean(z2[zz==1,21])  / mean(z2[zz==0,21]))
      z2           <- IncRate[id,]
      RRchild1_inc <- c(RRchild1_inc,mean(z2[zz==1,1])  / mean(z2[zz==0,1])) 
    }
  }
  ### ADOLESCENTS 
  RRadol1_inc <- RRadol20 <- RRadol2 <- NULL
  for(i in 1:length(AuthorID)) { # i=2
    id <- D2$Unique==AuthorID[i] & (D2$adolescents==1 | D2$standard==1)
    zz <- D2[id ,]
    if(sum(zz$adolescents==1)*sum(zz$adolescents==0)>0) {
      z2          <- CumInc[id,]
      RRadol2     <- c(RRadol2,mean(z2[zz$adolescents==1,3])  / mean(z2[zz$adolescents==0,3]))
      RRadol20    <- c(RRadol20,mean(z2[zz$adolescents==1,21])  / mean(z2[zz$adolescents==0,21]))
      z2          <- IncRate[id,]
      RRadol1_inc <- c(RRadol1_inc,mean(z2[zz$adolescents==1,1])  / mean(z2[zz$adolescents==0,1])) 
    }
  }
  ### Any HIV, non ART
  RRhiv1_inc <- RRhiv20_inc <- RRhiv20 <-RRhiv2 <- NULL
  for(i in 1:length(AuthorID)) { # i=2
    id <- D2$Unique==AuthorID[i] & (D2$hiv==1 | D2$standard==1) & D2$art==0
    zz <- D2[id,]
    if(sum(zz$hiv==1)*sum(zz$hiv==0)>0) {
      z2          <- CumInc[id,]
      RRhiv2      <- c(RRhiv2,mean(z2[zz$hiv==1,3])  / mean(z2[zz$hiv==0,3]))
      RRhiv20     <- c(RRhiv20,mean(z2[zz$hiv==1,21])  / mean(z2[zz$hiv==0,21]))
      z2          <- IncRate[id,]
      RRhiv20_inc <- c(RRhiv20_inc,mean(z2[zz$hiv==1,20])  / mean(z2[zz$hiv==0,20])) 
      RRhiv1_inc  <- c(RRhiv1_inc,mean(z2[zz$hiv==1,1])  / mean(z2[zz$hiv==0,1])) 
    }
  }
  ### Early HIV, no ART
  RRhear1_inc <- RRhear20_inc <- RRhear20 <- RRhear2 <- NULL
  for(i in 1:length(AuthorID)) { # i=266
    id <- D2$Unique==AuthorID[i] & (D2$early_hiv==1 | D2$standard==1) & D2$art==0
    zz <- D2[id,]
    if(sum(zz$early_hiv==1)*sum(zz$early_hiv==0)>0) {
      z2           <- CumInc[id,]
      RRhear2      <- c(RRhear2,mean(z2[zz$early_hiv==1,3])  / mean(z2[zz$early_hiv==0,3]))
      RRhear20     <- c(RRhear20,mean(z2[zz$early_hiv==1,21])  / mean(z2[zz$early_hiv==0,21]))
      z2           <- IncRate[id,]
      RRhear20_inc <- c(RRhear20_inc,mean(z2[zz$early_hiv==1,20])  / mean(z2[zz$early_hiv==0,20])) 
      RRhear1_inc  <- c(RRhear1_inc,mean(z2[zz$early_hiv==1,1])  / mean(z2[zz$early_hiv==0,1])) 
    }
  }
  ### Late HIV, no ART
  RRhlat1_inc <- RRhlat20_inc <- RRhlat20 <- RRhlat2 <- NULL
  for(i in 1:length(AuthorID)) { # i=2
    id <- D2$Unique==AuthorID[i] & (D2$advanced_hiv==1 | D2$standard==1) & D2$art==0
    zz <- D2[id,]
    if(sum(zz$advanced_hiv==1)*sum(zz$advanced_hiv==0)>0) {
      z2           <- CumInc[id,]
      RRhlat2      <- c(RRhlat2,mean(z2[zz$advanced_hiv==1,3])  / mean(z2[zz$advanced_hiv==0,3]))
      RRhlat20     <- c(RRhlat20,mean(z2[zz$advanced_hiv==1,21])  / mean(z2[zz$advanced_hiv==0,21]))
      z2           <- IncRate[id,]
      RRhlat20_inc <- c(RRhlat20_inc,mean(z2[zz$advanced_hiv==1,20])  / mean(z2[zz$advanced_hiv==0,20])) 
      RRhlat1_inc  <- c(RRhlat1_inc,mean(z2[zz$advanced_hiv==1,1])  / mean(z2[zz$advanced_hiv==0,1])) 
    }
  }
  ### ART, no ART
  RRhart1_inc <- RRhart20_inc <- RRhart20 <- RRhart2 <- NULL
  for(i in 1:length(AuthorID)) { # i=2
    id <- D2$Unique==AuthorID[i] & D2$hiv==1
    zz <- D2[id,]
    if(sum(zz$art==1)*sum(zz$art==0)>0) {
      z2           <- CumInc[id,]
      RRhart2      <- c(RRhart2,mean(z2[zz$art==1,3])  / mean(z2[zz$art==0,3]))
      RRhart20     <- c(RRhart20,mean(z2[zz$art==1,21])  / mean(z2[zz$art==0,21]))
      z2           <- IncRate[id,]
      RRhart20_inc <- c(RRhart20_inc,mean(z2[zz$art==1,20])  / mean(z2[zz$art==0,20])) 
      RRhart1_inc  <- c(RRhart1_inc,mean(z2[zz$art==1,1])  / mean(z2[zz$art==0,1])) 
    }
  }
  
  save(RRinf2,RRchild2,RRhiv2,RRhear2,RRhlat2,RRhart2,RRinf1_inc,RRchild1_inc,RRhiv20_inc,
       RRhear20_inc,RRhlat20_inc,RRhart20_inc,file="Reactivation_model_results_RRs_10-18-2017.rData")

###########################################
#########    CITATION ANALYSIS    #########   
###########################################

##### Load data  ##### ##### ##### ##### ##### 
  RefDat <- as.data.frame(read.csv("LTBI Citation Data_Clean_10-18-2017.csv"))[,-1]

  ##### List of unique refs ##### ##### ##### ##### ##### 
  UniqueRef0 <- NULL
  for(i in 8:(ncol(RefDat)-1)) {
    UniqueRef0 <- c(UniqueRef0,as.character(unique(RefDat[,i])))
  }
  UniqueRef <- unique(UniqueRef0)
  UniqueRef <- UniqueRef[-1]  # remove "" as a category. 
  cnt <- NULL
  for(i in 1:length(UniqueRef)) {
    cnt[i] <- sum(apply(RefDat[,8:(ncol(RefDat)-1)],1,function(z) sum(z==UniqueRef[i])>0))
  }
  
##### Table of publication by citation ##### ##### ##### ##### ##### 
  PubCit <- matrix(0,length(unique(RefDat$Unique)),length(UniqueRef))
  colnames(PubCit) <- UniqueRef
  for(j in 1:length(UniqueRef)) { 
    for(i in 1:length(unique(RefDat$Unique))) { # i=j=1
      PubCit[i,j] <- ifelse(sum(RefDat[RefDat$Unique==unique(RefDat$Unique)[i],8:(ncol(RefDat)-1)]==UniqueRef[j])>0,1,0) 
    }
    txtProgressBar(1,length(UniqueRef),j,style=3) 
  }
  
  save(PubCit,file="PubByCit_10-18-2017.rData")

#####################################################
#########    FIT MODEL TO EMPIRICAL DATA    #########   
#####################################################
  ferebee    <- read.csv("Ferebee 1970 data_rev3.csv")
  sutherland <- read.csv("Sutherland 1968 data_rev.csv")
  
  logit       <- function(p) log(p/(1-p))
  expit       <- function(v) exp(v)/(exp(v)+1)
  suth_target <- c(0,cumsum(sutherland[,6])/sutherland[1,3])
  fere_target <- c(0,cumsum(ferebee[,3])/ferebee[1,2])
  CumIncRes   <- list()
  RMSE        <- list()
  PAR         <- list()
  
##### STRUCTURE A ##### ##### ##### ##### ##### 
  func0 <- function(v2) {
    pred <- mod(a=0,b=0,c=v2[1],d=0,e=0,f=0,n=0,py=365,yrs=10)[-1]
    mean((pred - suth_target[-1])^2)^0.5*100 
  }
  opt0 <- optimize(func0,interval=c(0,10),tol = .Machine$double.eps^0.5)
  opt0$objective
  RMSE[["A"]] <- opt0$objective
  PAR[["A"]]  <- opt0$minimum
  CumIncRes[["A"]] <- c(0,mod(a=0,b=0,c=opt0$minimum[1],d=0,e=0,f=0,n=0,py=365,yrs=10)[-1])*100
  ## STRUCTURE B
  func0 <- function(v2) {
    pred = mod(a=0,b=1,c=exp(v2[1]),d=exp(v2[2:6]),e=0,f=0,n=5,py=365,yrs=10)[-1]
    mean((pred - suth_target[-1])^2)^0.5*100 
  }
  opt0 = optim(rep(-2,6),func0,method="Nelder-Mead")
  for(i in 1:10) {  
    opt0 = optim(opt0$par,func0,method="Nelder-Mead")
    opt0 = optim(opt0$par,func0,method="BFGS")
    print(opt0$value); flush.console() 
  }
  opt0$value; opt0$convergence
  RMSE[["B"]] <- opt0$value
  PAR[["B"]]  <- exp(opt0$par)
  CumIncRes[["B"]] <- c(0,mod(a=0,b=1,c=exp(opt0$par[1]),d=exp(opt0$par[2:6]),e=0,f=0,n=5,py=365,yrs=10)[-1])*100
  ## STRUCTURE C
  func0 <- function(v2) {
    pred = mod(a=0,b=1,c=exp(v2[1]),d=exp(v2[2]),e=exp(v2[3]),f=0,n=0,py=365,yrs=10)[-1]
    mean((pred - suth_target[-1])^2)^0.5*100 
  }
  opt0 = optim(rep(-4,3),func0,method="Nelder-Mead")
  for(i in 1:10) {  
    opt0 = optim(opt0$par,func0,method="Nelder-Mead")
    opt0 = optim(opt0$par,func0,method="BFGS")
    print(opt0$value); flush.console() 
  }
  opt0$value; opt0$convergence
  RMSE[["C"]] <- opt0$value
  PAR[["C"]]  <- exp(opt0$par)
  CumIncRes[["C"]] <- c(0,mod(a=0,b=1,c=exp(opt0$par[1]),d=exp(opt0$par[2]),e=exp(opt0$par[3]),f=0,n=0,py=365,yrs=10)[-1])*100
  ## STRUCTURE D
  PAR[["D"]]  <- NA
  RMSE[["D"]] <- sqrt(sum((1 - suth_target[-1])^2)/10)*100
  CumIncRes[["D"]] <- c(0,rep(1,10))*100
  ## STRUCTURE E
  func0 <- function(v2) {
    pred = mod(a=expit(v2[1]),b=0,c=exp(v2[2]),d=0,e=0,f=0,n=0,py=365,yrs=10)[-1]
    mean((pred - suth_target[-1])^2)^0.5*100 
  }
  opt0 = optim(rep(-2,2),func0,method="Nelder-Mead")
  for(i in 1:10) {  
    opt0 = optim(opt0$par,func0,method="Nelder-Mead")
    opt0 = optim(opt0$par,func0,method="BFGS")
    print(opt0$value); flush.console() 
  }
  opt0$value; opt0$convergence
  RMSE[["E"]] <- opt0$value
  PAR[["E"]]  <- c(expit(opt0$par[1]),exp(opt0$par[2]))
  CumIncRes[["E"]] <- c(0,mod(a=expit(opt0$par[1]),b=0,c=exp(opt0$par[2]),d=0,e=0,f=0,n=0,py=365,yrs=10)[-1])*100
  ## STRUCTURE F
  func0 <- function(v2) {
    pred = mod(a=0,b=expit(v2[1]),c=exp(v2[2]),d=exp(v2[3]),e=0,f=0,n=0,py=365,yrs=10)[-1]
    mean((pred - suth_target[-1])^2)^0.5*100 
  }
  opt0 = optim(rep(-2,3),func0,method="Nelder-Mead")
  for(i in 1:10) {  
    opt0 = optim(opt0$par,func0,method="Nelder-Mead")
    opt0 = optim(opt0$par,func0,method="BFGS")
    print(opt0$value); flush.console() 
  }
  opt0$value; opt0$convergence
  RMSE[["F"]] <- opt0$value
  PAR[["F"]]  <- c(expit(opt0$par[1]),exp(opt0$par[2:3]))
  CumIncRes[["F"]] <- c(0,mod(a=0,b=expit(opt0$par[1]),c=exp(opt0$par[2]),d=exp(opt0$par[3]),e=0,f=0,n=0,py=365,yrs=10)[-1])*100
  ## STRUCTURE G
  func0 <- function(v2) {
    pred = mod(a=expit(v2[1]),b=1-expit(v2[1]),c=exp(v2[2]),d=exp(v2[3]),e=exp(v2[4]),f=0,n=0,py=365,yrs=10)[-1]
    mean((pred - suth_target[-1])^2)^0.5*100 
  }
  opt0 = optim(rep(-4,4),func0,method="Nelder-Mead")
  for(i in 1:10) {  
    opt0 = optim(opt0$par,func0,method="Nelder-Mead")
    opt0 = optim(opt0$par,func0,method="BFGS")
    print(opt0$value); flush.console() 
  }
  opt0$value; opt0$convergence
  RMSE[["G"]] <- opt0$value
  PAR[["G"]]  <- c(expit(opt0$par[1]),exp(opt0$par[2:4]))
  CumIncRes[["G"]] <- c(0,mod(a=expit(opt0$par[1]),b=1-expit(opt0$par[1]),c=exp(opt0$par[2]),d=exp(opt0$par[3]),e=exp(opt0$par[4]),f=0,n=0,py=365,yrs=10)[-1])*100
  ## STRUCTURE H
  func0 <- function(v2) {
    pred = mod(a=0,b=expit(v2[1]),c=0,d=exp(v2[2]),e=0,f=exp(v2[3]),n=0,py=365,yrs=10)[-1]
    mean((pred - suth_target[-1])^2)^0.5*100 } 
  opt0 = optim(rep(-2,3),func0,method="Nelder-Mead")
  for(i in 1:10) {  
    opt0 = optim(opt0$par,func0,method="Nelder-Mead")
    opt0 = optim(opt0$par,func0,method="BFGS")
    print(opt0$value); flush.console() 
  }
  opt0$value; opt0$convergence
  RMSE[["H"]] <- opt0$value
  PAR[["H"]]  <- c(expit(opt0$par[1]),exp(opt0$par[2:3]))
  CumIncRes[["H"]] <- c(0,mod(a=0,b=expit(opt0$par[1]),c=0,d=exp(opt0$par[2]),e=0,f=exp(opt0$par[3]),n=0,py=365,yrs=10)[-1])*100
  ## STRUCTURE I  
  func0 <- function(v2) {
    pred = mod(a=0,b=1,c=0,d=exp(v2[1]),e=exp(v2[2]),f=exp(v2[3]),n=0,py=365,yrs=10)[-1]
    mean((pred - suth_target[-1])^2)^0.5*100 
  }
  opt0 = optim(rep(-2,3),func0,method="Nelder-Mead")
  for(i in 1:10) {  
    opt0 = optim(opt0$par,func0,method="Nelder-Mead")
    opt0 = optim(opt0$par,func0,method="BFGS")
    print(opt0$value); flush.console() 
  }
  opt0$value; opt0$convergence
  RMSE[["I"]] <- opt0$value
  PAR[["I"]]  <- c(exp(opt0$par[1:3]))
  CumIncRes[["I"]] <- c(0,mod(a=0,b=1,c=0,d=exp(opt0$par[1]),e=exp(opt0$par[2]),f=exp(opt0$par[3]),n=0,py=365,yrs=10)[-1])*100
  ## STRUCTURE J 
  func0 <- function(v2) {
    pred = modJ(c=v2[1],f=c(100,100,100),py=365,yrs=10)[-1]
    mean((pred - suth_target[-1])^2)^0.5*100 
  }
  opt0 = optimize(func0,interval=0:1,tol = .Machine$double.eps^0.5)
  opt0$objective
  RMSE[["J"]] <- opt0$objective
  PAR[["J"]]  <- opt0$minimum
  CumIncRes[["J"]] <- c(0,modJ(c=opt0$minimum,f=c(100,100,100),py=365,yrs=10)[-1])*100
  ## STRUCTURE K
  func0 <- function(v2) {
    pred = modK(x1=exp(v2[1]), x2=v2[2], py=365, yrs=10)[-1]
    mean((pred - suth_target[-1])^2)^0.5*100 
  }
  opt0 = optim(rep(-2,2),func0,method="Nelder-Mead")
  for(i in 1:10) {  
    opt0 = optim(opt0$par,func0,method="Nelder-Mead")
    opt0 = optim(opt0$par,func0,method="BFGS")
    print(opt0$value); flush.console() 
  }
  opt0$value; opt0$convergence
  RMSE[["K"]] <- opt0$value
  PAR[["K"]]  <- c(exp(opt0$par[1]),opt0$par[2])
  CumIncRes[["K"]] <- c(0,modK(x1=exp(opt0$par[1]),x2=opt0$par[2],py=365,yrs=10)[-1])*100
  ## STRUCTURE L
  func0 <- function(v2) {
    pred = modL(x1=exp(v2[1]), x2=exp(v2[2]), x3=exp(v2[3]), py=365, yrs=10)[-1]
    mean((pred - suth_target[-1])^2)^0.5*100 
  }
  opt0 = optim(rep(-2,3),func0,method="Nelder-Mead")
  for(i in 1:10) {  
    opt0 = optim(opt0$par,func0,method="Nelder-Mead")
    opt0 = optim(opt0$par,func0,method="BFGS")
    print(opt0$value); flush.console() 
  }
  opt0$value; opt0$convergence
  RMSE[["L"]] <- opt0$value
  PAR[["L"]]  <- exp(opt0$par)
  CumIncRes[["L"]] <- c(0,modL(x1=exp(opt0$par[1]),x2=exp(opt0$par[2]),x3=exp(opt0$par[3]),py=365,yrs=10)[-1])*100
  ### CumIncRes
  IncRateRes = NULL
  for(i in 1:length(CumIncRes)) { IncRateRes[[i]] <- diff(CumIncRes[[i]])/(100-CumIncRes[[i]][-length(CumIncRes[[i]])])*1000 }
  
  save(IncRateRes,CumIncRes,RMSE,PAR,file="Reactivation_model_results_fits_10-18-2017.rData")
  