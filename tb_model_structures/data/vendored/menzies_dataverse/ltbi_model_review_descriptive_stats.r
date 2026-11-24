## BASIC DESCRIPTIVE STATS
setwd("/Users/nicolasmenzie/Google Drive/Harvard/CDC Large Grant/Reactiveation rate review")

load("Reactivation_model_results_10-18-2017.rData") # IncRate,CumInc,vID2,vCat,D2

# How many studies?
length(unique(D2$Unique))

# How many obs?
length((D2$Unique))

# Oldest?
min(D2$year)

# % before 2000
D2u <- NULL
for(i in 1:length(unique(D2$Unique)))  D2u <- rbind(D2u,D2[D2$Unique==(unique(D2$Unique)[i]),][1,])
sum(D2u$year<2000)/length(D2u$year)
median(D2u$year)

sum(D2u$year>=2000 & D2u$year<2004)
sum( D2u$year<2000 )

# % high burden
zzz <- 0
for(i in 1:length(unique(D2$Unique))) {
  if(sum(D2[D2$Unique==(unique(D2$Unique)[i]),"high_burden"],na.rm=T)>0) {  zzz <- zzz+1 }
}
zzz/length(D2u$year)

# % low burden
zzz <- 0
for(i in 1:length(unique(D2$Unique))) {
  if(sum(D2[D2$Unique==(unique(D2$Unique)[i]),"low_burden"],na.rm=T)>0) {  zzz <- zzz+1 }
}
zzz/length(D2u$year)

# any risk factor
zzz <- 0
for(i in 1:length(unique(D2$Unique))) {
  if(sum(D2[D2$Unique==(unique(D2$Unique)[i]),"elevated_risk"],na.rm=T)+
     sum(D2[D2$Unique==(unique(D2$Unique)[i]),"reduced_risk"],na.rm=T)+
     sum(D2[D2$Unique==(unique(D2$Unique)[i]),"children"],na.rm=T)>0) {  zzz <- zzz+1 }
}
zzz/length(D2u$year)

# HIV
zzz <- 0
for(i in 1:length(unique(D2$Unique))) {
  if(sum(D2[D2$Unique==(unique(D2$Unique)[i]),"hiv"],na.rm=T)>0) {  zzz <- zzz+1 }
}
zzz/length(D2u$year)

# age
zzz <- 0
for(i in 1:length(unique(D2$Unique))) {
  if(sum(D2[D2$Unique==(unique(D2$Unique)[i]),c("infants","older_children","children","adolescents")],na.rm=T)>0) {  zzz <- zzz+1 }
}
zzz/length(D2u$year)

##  pub frequency
load("PubByCit_10-18-2017.rData")  # PubCit
dim(PubCit)
colSums(PubCit)[order(-colSums(PubCit))][1:10]/nrow(PubCit)

PubPerCit <- colSums(PubCit)
vec <- c(sum(rowSums(PubCit)==0),PubPerCit[order(-PubPerCit)][1:14])
vec # top 15
vec/length(unique(D2$Unique))


###  median incidence
id <- vID2[[2]]
apply(IncRate[id,],2,function(x) quantile(x,0.5))[c(1,20)]*1e3
apply(CumInc[id,],2,function(x) quantile(x,0.5))[c(2,21)]*1e2

apply(IncRate[id,],2,function(x) quantile(x,0.9))[1]/apply(IncRate[id,],2,function(x) quantile(x,0.1))[1]
apply(IncRate[id,],2,function(x) quantile(x,0.9))[1]*1e3
apply(IncRate[id,],2,function(x) quantile(x,0.1))[1]*1e3

apply(IncRate[id,],2,function(x) quantile(x,0.9))[20]/apply(IncRate[id,],2,function(x) quantile(x,0.1))[20]
apply(IncRate[id,],2,function(x) quantile(x,0.9))[20]*1e3
apply(IncRate[id,],2,function(x) quantile(x,0.1))[20]*1e3

apply(CumInc[id,],2,function(x) quantile(x,0.9))[21]/apply(CumInc[id,],2,function(x) quantile(x,0.1))[21]
apply(CumInc[id,],2,function(x) quantile(x,0.9))[21]*1e2
apply(CumInc[id,],2,function(x) quantile(x,0.1))[21]*1e2

## Fraction wildly dissimilar to Ferebee/Sutherland
load("Reactivation_model_results_10-18-2017.rData") # IncRate,CumInc,vID2,vCat,D2

ferebee    <- read.csv("Ferebee 1970 data_rev3.csv")
sutherland <- read.csv("Sutherland 1968 data_rev.csv")
id <- D2$standard==1
id2 <- ((D2$model_structure %in% c("A","D","J")) == F)  & id
id3 <- ((D2$model_structure %in% c("A","D","J")) == T)  & id
sum(id2);sum(id3)

f1 = (cumsum(ferebee[,3])/ferebee[1,2])[10]
c1 = (cumsum(sutherland[,6])/sutherland[1,3])[10]

# Fraction above 50%, below 0.5%
sum(CumInc[id,11]>0.5 | CumInc[id,11]<0.005) / length(CumInc[id,11])*100
sum(CumInc[id,11]>0.5 ) / length(CumInc[id,11])*100
sum(CumInc[id,11]<0.01) / length(CumInc[id,11])*100

# all wth no risk factors
odr = 2
100-sum(CumInc[id,11]>(max(f1,c1)*odr) | CumInc[id,11]<(min(f1,c1)/odr)) / length(CumInc[id,11])*100

odr = 5
100-sum(CumInc[id,11]>(max(f1,c1)*odr) | CumInc[id,11]<(min(f1,c1)/odr)) / length(CumInc[id,11])*100

# excluding A, D, J
odr = 2
100-sum(CumInc[id2,11]>(max(f1,c1)*odr) | CumInc[id2,11]<(min(f1,c1)/odr)) / length(CumInc[id2,11])*100

odr = 5
100-sum(CumInc[id2,11]>(max(f1,c1)*odr) | CumInc[id2,11]<(min(f1,c1)/odr)) / length(CumInc[id2,11])*100

# only A, D, J

odr = 2
100-sum(CumInc[id3,11]>(max(f1,c1)*odr) | CumInc[id3,11]<(min(f1,c1)/odr)) / length(CumInc[id3,11])*100

odr = 5
100-sum(CumInc[id3,11]>(max(f1,c1)*odr) | CumInc[id3,11]<(min(f1,c1)/odr)) / length(CumInc[id3,11])*100


############  CHECK TABLE

tabA <- matrix(NA,6,13)
for(i in 1:6) { for (j in 1:12) {
  zz <- D2$Unique[D2$year%in%(1950:1959+i*10) & D2$model_structure==c("A","B","C","D","E","F","G","H","I","J","K","L")[j]]
  tabA[i,j] <- length(unique(zz))
  
}
  zz <- D2$Unique[D2$year%in%(1950:1959+i*10) ]
  tabA[i,13] <- length(unique(zz)) 
}
tabA[,13]
round(tabA[,13]/sum(tabA[,13   ])*1e2,1)

length(grep("2017",as.character(D2$Unique)))

##
zz <- NULL
for(i in c("A","B","C","D","E","F","G","H","I","J","K","L")){
  zz[i] <- length(unique(D2$Unique[D2$model_structure==i]))
}
zz
round(zz/312*100,1)

##
tabB <- matrix(NA,3,13)
for (j in 1:12) {
  zz <- D2$Unique[ D2$high_burden==1 & is.na(D2$high_burden)==F & D2$model_structure==c("A","B","C","D","E","F","G","H","I","J","K","L")[j]]
  tabB[1,j] <- length(unique(zz))
  zz <- D2$Unique[ D2$low_burden==1 & is.na(D2$low_burden)==F & D2$model_structure==c("A","B","C","D","E","F","G","H","I","J","K","L")[j]]
  tabB[2,j] <- length(unique(zz))
  cnd <- ((D2$low_burden==1 & is.na(D2$low_burden)==F) | (D2$high_burden==1 & is.na(D2$high_burden)==F))==F
  zz <- D2$Unique[ cnd & D2$model_structure==c("A","B","C","D","E","F","G","H","I","J","K","L")[j]]
  tabB[3,j] <- length(unique(zz))
  
}
  zz <- D2$Unique[D2$high_burden==1 & is.na(D2$high_burden)==F]
  tabB[1,13] <- length(unique(zz)) 
  zz <- D2$Unique[D2$low_burden==1 & is.na(D2$low_burden)==F]
  tabB[2,13] <- length(unique(zz)) 
  zz <- D2$Unique[cnd]
  tabB[3,13] <- length(unique(zz))
  
  tabB[,13]
  round(tabB[,13]/sum(tabA[,13   ])*1e2,1)
####
  zd <- (D2$infants==0 & D2$older_children==0 & D2$children==0 & D2$adolescents==0)==F
  sum((aggregate(zd,by=list(D2$Unique),function(z) sum(z)>0))[,2])
  sum((aggregate(zd,by=list(D2$Unique),function(z) sum(z)>0))[,2])/291
  
####
  zd <- D2$hiv==1
  sum((aggregate(zd,by=list(D2$Unique),function(z) sum(z)>0))[,2])
  sum((aggregate(zd,by=list(D2$Unique),function(z) sum(z)>0))[,2])/291
  
  ####
  zd <- (D2$elevated_risk==0 & D2$reduced_risk==0)==F
  sum((aggregate(zd,by=list(D2$Unique),function(z) sum(z)>0))[,2])
  sum((aggregate(zd,by=list(D2$Unique),function(z) sum(z)>0))[,2])/291
  
 zz <- 0
 for(i in unique(D2$Unique)){
   if(sum(D2$standard[D2$Unique==i]==0)>0) { zz <- zz+1 }
 }
 zz
 
######################

