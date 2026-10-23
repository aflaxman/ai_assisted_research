
  library(RColorBrewer)
  library(plotrix)
  mTrsp <- function(cl,a)  { apply(col2rgb(cl), 2, function(x){ rgb(x[1],x[2],x[3],a,maxColorValue=255)}) }

### Load data
  load("Reactivation_model_results_10-18-2017.rData") # IncRate,CumInc,vID2,vCat,D2
  ferebee    <- read.csv("Ferebee 1970 data_rev3.csv")
  sutherland <- read.csv("Sutherland 1968 data_rev.csv")
  load("PubByCit_10-18-2017.rData") # PubCit
  # PubCit
  load("Reactivation_model_results_RRs_10-18-2017.rData")
        #  RRinf2,RRchild2,RRhiv2,RRhear2,RRhlat2,RRhart2,RRinf1_inc,RRchild1_inc,
        #  RRhiv20_inc,RRhear20_inc,RRhlat20_inc,RRhart20_inc
  load("Reactivation_model_results_fits_10-18-2017.rData") 
        # IncRateRes,CumIncRes,RMSE,PAR
  
### Large figure of cumulative incidence results stratified by structure and categorization
  cls <-  c(brewer.pal(9, "Set1")[1:4],brewer.pal(8, "Dark2")[c(1:5,7)],"grey20","blue")
  pdfnam <- "Figure_cumulative_incidence_detailed.pdf"
  pdf(file=pdfnam,width=9, height=6.5) 
  par(oma=c(4.5,5,2.5,1),mfrow=c(3,4),mar=c(0,0,2,0))

  for(q in 1:length(vCat)) {
    for(k in 1:12) { # k=7
      plot(0,0,col="white",xlim=c(.25,19.75),ylim=c(2,98),xlab="",ylab="",las=1,axes=F)
      id1 <- D2$model_structure==c("A","B","C","D","E","F","G","H","I","J","K","L")[k]; id2 = vID2[[q]]
      id  <- id1 & id2
      zz  <- CumInc[id,]
      if(sum(id)>0) {
        if(sum(id)>1) {
          z2 <- zz[!duplicated(rowSums(zz)), ]
          if(length(z2)==ncol(zz)) {
            z2 <- t(z2)
          }
          wd <- apply(z2,1,function(x) sum(rowSums(zz)==sum(x)))
          m <- ifelse(nrow(z2)<6,2,1)
          for(i in 1:nrow(z2) ) {
            lines(c(0,0:(ncol(z2)-1)), c(0,z2[i,])*100,col=mTrsp(cls[k],75*m),lwd=wd[i])  
          }
        } else {
          lines(c(0,0:(length(zz)-1)), c(0,zz)*100,col=mTrsp(cls[k],150))  
        }
      }
      if(k%in%c(1,5,9)) {
        axis(2,las=1)
      }
      if(k%in%(9:12)) {
        axis(1)
      }
      if(k==9) { 
        mtext(paste("Cumulative Incidence, ",vCat[[q]]," (",sum(id2),")",sep=""),3,0.8,font=2,outer=T)
      }
      if(k==5) {
        mtext("Percent progressed to active TB",2,3.5)
      }
      if(k==8) {
        mtext("Years since infection",1,2.5,outer=T)
      }
      box()
      mtext(paste("Structure ",c("A","B","C","D","E","F","G","H","I","J","K","L")[k]," (",sum(id),")",sep=""),3,0.2)
    }
  }
  dev.off();system(paste("open", pdfnam))

  
### Large figure of incidence rate results stratified by structure and categorization
  pdfnam <- "Figure_incidence_rate_detailed.pdf"
  pdf(file=pdfnam,width=9, height=6.5) 
  par(oma=c(4.5,5,2.5,1),mfrow=c(3,4),mar=c(0,0,2,0))

  for(q in 1:length(vCat)) {
    for(k in 1:12) { # k=7
      plot(0,0,col="white",xlim=c(.25,19.75),ylim=c(2,98),xlab="",ylab="",las=1,axes=F)
      id1 <- D2$model_structure==c("A","B","C","D","E","F","G","H","I","J","K","L")[k]; id2 = vID2[[q]]
      id  <- id1 & id2
      zz  <- IncRate[id,]
      if(sum(id)>0) {
        if(sum(id)>1) {
          z2 <- zz[!duplicated(rowSums(zz)), ]
          if(length(z2)==ncol(zz)) {
            z2 <- t(z2)
          }
          wd <- apply(z2,1,function(x) sum(rowSums(zz)==sum(x)))
          m <- ifelse(nrow(z2)<6,2,1)
          for(i in 1:nrow(z2) ){
            lines(c(0:(ncol(z2)-1))+0.5, c(z2[i,])*100,col=mTrsp(cls[k],75*m),lwd=wd[i])  
          }
        } else {
          lines(c(0:(length(zz)-1))+0.5, zz*100,col=mTrsp(cls[k],150))  
        }
      }
      if(k%in%c(1,5,9)) {
        axis(2,las=1)
      }
      if(k%in%(9:12)) {
        axis(1)
      }
      if(k==9) {
        mtext(paste("Annual Incidence, ",vCat[[q]]," (",sum(id2),")",sep=""),3,0.8,font=2,outer=T)
      }
      if(k==5){
        mtext("Annual Risk of TB (%)",2,3.5)
      }
      if(k==8) {
        mtext("Years since infection",1,2.5,outer=T)
      } 
      box()
      mtext(paste("Structure ",c("A","B","C","D","E","F","G","H","I","J","K","L")[k]," (",sum(id),")",sep=""),3,0.2)
    } 
  }
  dev.off();system(paste("open", pdfnam))


### Figure 3: Model predictions for annual and cumulative incidence of active TB 
### by years since M. tb infection, for population groups with no individual risk factors.
  pdfnam <- "Figure_3_main_text.pdf"
  pdf(file=pdfnam,width=10, height=6) 
  ly <- rbind(cbind(rep(1,12),rep(2,12)),c(3,3))
  layout(ly)
  cl <- brewer.pal(9, "Set1")[1]
  ### INC RATE
  par(oma=c(0,0.1,0.0,0),mar=c(3.5,4.8,0.3,.3))
  id <- vID2[[2]]
  zz <- IncRate[id,]
  plot(100,1,xlim=c(.6,19.4),ylim=c(0.085,850),las=1,xlab="",ylab="",log="y",axes=F)
  axis(1,0:4*5,las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.4)
  axis(2,10^(-3:3),as.character(10^(-3:3)),las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.4)
  mtext("Years since initial infection",1,2,cex=0.9)
  mtext("Annual risk of active TB (per 1000, log-scale)",2,3.5,cex=1.0)
  for(i in 1:nrow(zz)) {
    lines(1:ncol(zz)-.5, zz[i,]*1e3,col=mTrsp("grey20",100),lwd=1) 
  }
  lines(1:20-.5,apply(zz,2,function(x) quantile(x,0.5))[1:20]*1e3,col="white",lwd=9)
  lines(1:20-.5,apply(zz,2,function(x) quantile(x,0.5))[1:20]*1e3,col=cl,lwd=4)
  box()
  ##  CUM INC
  par(mar=c(3.5,5.3,0.3,.3))
  id = vID2[[2]]
  plot(100,1,xlim=c(.35,19.65),ylim=c(2,99),las=1,xlab="",ylab="",axes=F)
  axis(1,las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.4)
  axis(2,las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.4)
  mtext("Years since initial infection",1,2,cex=0.9)
  mtext("Cumulative risk of active TB (%)",2,3,cex=1.0)
  zz  <- CumInc[id,]
  for(i in 1:nrow(zz)) {
    lines(0:20, c(0,zz[i,2:21])*100,col=mTrsp("grey20",100),lwd=1) 
  }
  lines(0:20,c(0,apply(zz,2,function(x) quantile(x,0.5))[2:21]*1e2),col="white",lwd=9)
  lines(0:20,c(0,apply(zz,2,function(x) quantile(x,0.5))[2:21]*1e2),col=cl,lwd=4)
  box()
  ##
  par(mar=c(0,0.5,0,0.5))
  plot(1,1,col="white",axes=F,xlab="",ylab="")
  legend("center",c("Individual models/strata","Median"),
         lwd=c(1,2),col=c("grey40",cl),lty=c(1,1),ncol=2,cex=1.4)
  dev.off();system(paste("open", pdfnam))

  
### Figure 4: Distribution of model predictions for cumulative incidence of
### active TB at two and twenty years since M. tb infection, stratified by 
### model structure, individual risk factors, and other study characteristics*.
  nyr1 <- 3
  bpm1 <- list("All models, all strata" = CumInc[,nyr1],
               "No individual risk factors"  = CumInc[D2$standard==1 ,nyr1],
               "Structure A*" = CumInc[D2$standard==1 & D2$model_structure=="A",nyr1],
               "Structure B*" = CumInc[D2$standard==1 & D2$model_structure=="B",nyr1],
               "Structure C*" = CumInc[D2$standard==1 & D2$model_structure=="C",nyr1],
               "Structure E*" = CumInc[D2$standard==1 & D2$model_structure=="E",nyr1],
               "Structure F*" = CumInc[D2$standard==1 & D2$model_structure=="F",nyr1],
               "Low-burden setting*" = CumInc[D2$low_burden==1 & D2$standard==1,nyr1],
               "High-burden setting*" = CumInc[D2$high_burden==1 & D2$standard==1,nyr1],
               "No citations*" = CumInc[D2$no_cites==1 & D2$standard==1,nyr1],
               "Any citations*" = CumInc[D2$no_cites==0 & D2$standard==1,nyr1],
               "Published 2010 or earlier*" = CumInc[D2$standard==1,nyr1],
               "Published after 2010*" = CumInc[D2$standard==1,nyr1],
               "Any individual risk factor" = CumInc[D2$elevated_risk==1,nyr1],
               "Infants (0-2 years)**" = CumInc[D2$infants==1,nyr1],
               "Children (excl. infants)**" = CumInc[D2$children==1 & D2$infants==0 ,nyr1],
               "All HIV**" = CumInc[D2$hiv==1,nyr1],
               "Early HIV**" = CumInc[D2$early_hiv==1,nyr1],
               "Late HIV**" = CumInc[D2$advanced_hiv==1,nyr1],
               "HIV, on ART**" = CumInc[D2$art==1,nyr1] )
  
  nyr2 <- 21
  bpm10 <- list("All models, all strata" = CumInc[,nyr2],
                "No individual risk factors"  = CumInc[D2$standard==1,nyr2],
                "Structure A*" = CumInc[D2$standard==1 & D2$model_structure=="A",nyr2],
                "Structure B*" = CumInc[D2$standard==1 & D2$model_structure=="B",nyr2],
                "Structure C*" = CumInc[D2$standard==1 & D2$model_structure=="C",nyr2],
                "Structure E*" = CumInc[D2$standard==1 & D2$model_structure=="E",nyr2],
                "Structure F*" = CumInc[D2$standard==1 & D2$model_structure=="F",nyr2],
                "Low-burden setting*" = CumInc[D2$low_burden==1 & D2$standard==1,nyr2],
                "High-burden setting*" = CumInc[D2$high_burden==1 & D2$standard==1,nyr2],
                "No citations*" = CumInc[D2$no_cites==1 & D2$standard==1,nyr2],
                "Any citations*" = CumInc[D2$no_cites==0 & D2$standard==1,nyr2],
                "Published 2010 or earlier*" = CumInc[D2$year<=2010 & D2$standard==1,nyr2],
                "Published after 2010*" = CumInc[D2$year>2010 & D2$standard==1,nyr2],
                "Any individual risk factor" = CumInc[D2$elevated_risk==1,nyr2],
                "Infants (0-2 years)**" = NA,
                "Children (excl. infants)**" = NA,
                "All HIV**" = NA,
                "Early HIV**" = NA,
                "Late HIV**" = NA,
                "HIV, on ART**" = NA )

  ###
  pdfnam <- "Figure_4_main_text.pdf"
  pdf(file=pdfnam,width=10, height=10) 
  par(mfrow=c(2,1),mar=c(0,0,0,0),oma=c(10,4,.5,.5))
  cls <-  brewer.pal(8, "Pastel2")
  cl2 <- c("grey50",cls[c(8,rep(2,5),3,3,6,6,1,1,8,5,5,4,4,4,4)])
  ## 1 year
  plot(-1,-20,xlab="",ylab="",axes=F,xlim=c(1,length(bpm1)),ylim=c(-.08,.98))
  abline(h=0:1,col="grey90",lty="11")
  for(i in 1:length(bpm1)) {
    lines(c(i,i),c(0,1),col="grey90",lty="11")
  }
  boxplot(bpm1,col=cl2,axes=F,add=T)
  text(1:length(bpm1),rep(-.035,length(bpm1)),format(round(as.numeric(lapply(bpm1,median))*100,1),nsmall=1),cex=0.8)
  text(1:length(bpm1),rep(-.09,length(bpm1)),as.numeric(lapply(bpm1,length)),cex=0.8)
  box()
  axis(2,0:5*.2,0:5*20,las=1,tcl=-.25,mgp=c(3, 0.5, 0),cex.axis=1.1)
  axis(2,c(-0.04,-0.09),c("Median","N"),las=1,tcl=-.25,mgp=c(3, 0.5, 0),cex.axis=.8)
  mtext("Cumulative incidence after 2 years (%)",2,2.9)
  ## 10 years
  plot(-1,-10,xlab="",ylab="",axes=F,xlim=c(1,length(bpm1)),ylim=c(-.08,.98))
  abline(h=0:1,col="grey90",lty="11")
  for(i in 1:length(bpm1)) {
    lines(c(i,i),c(0,1),col="grey90",lty="11")
  }
  boxplot(bpm10,col=cl2,axes=F,add=T)
  text(1:(length(bpm10)-6),rep(-.035,length(bpm10)-6),format(round(as.numeric(lapply(bpm10,median))*100,1),nsmall=1)[1:14],cex=0.8)
  text(1:(length(bpm10)-6),rep(-.09,length(bpm10)-6),as.numeric(lapply(bpm10,length))[1:14],cex=0.8)
  box()
  axis(2,0:5*.2,0:5*20,las=1,tcl=-.25,mgp=c(3, 0.5, 0),cex.axis=1.1)
  axis(2,c(-0.04,-0.09),c("Median","N"),las=1,tcl=-.25,mgp=c(3, 0.5, 0),cex.axis=.8)
  tlab <- 1:length(bpm10) # where labels
  axis(1, at=tlab, labels=FALSE)
  text(x=tlab, y=par()$usr[3]-0.05*(par()$usr[4]-par()$usr[3]),labels=names(bpm1), srt=45, adj=1, xpd=NA)
  mtext("Cumulative incidence after 20 years (%)",2,2.9)
  ###
  dev.off();system(paste("open", pdfnam))

  
### Figure 5: Comparison between model predictions and empirical evidence for 
### annual and cumulative incidence of active TB by years since M. tb 
### infection, for groups with no individual risk factors*.
  pdfnam <- "Figure_5_main_text.pdf"
  pdf(file=pdfnam,width=10, height=6) 
  ly <- rbind(cbind(rep(1,6),rep(2,6)),c(3,3))
  layout(ly)
  cls <-  brewer.pal(5, "Set1")
  ### INC RATE
  par(oma=c(0,0.5,0,0),mar=c(3.5,5.1,0.3,.5))
  id = vID2[[2]]
  uu=0.04
  plot(100,1,xlim=c(1.1,9.9),ylim=c(0.07,300),las=1,xlab="",ylab="",log="y",axes=F)
  axis(1,1:10,las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.4)
  axis(2,las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.4)
  mtext("Year since initial infection",1,2,cex=0.9)
  mtext("Annual risk of active TB (per 1000, log-scale)",2,4.3,cex=1.0)
  polygon(c(1:10,10:1),c(apply(IncRate[id,],2,function(x) quantile(x,0.05))[1:10],
                         apply(IncRate[id,],2,function(x) quantile(x,0.95))[10:1])*1e3,col="grey90",border="grey88")
  polygon(c(1:10,10:1),c(apply(IncRate[id,],2,function(x) quantile(x,0.25))[1:10],
                         apply(IncRate[id,],2,function(x) quantile(x,0.75))[10:1])*1e3,col="grey80",border="grey78")
  lines(1:10,apply(IncRate[id,],2,function(x) quantile(x,0.5))[1:10]*1e3,col="grey45",lwd=2,lend="butt")
  lines(ferebee[,1]-uu,ferebee[,3]/ferebee[,2]*1e3,col=cls[1],lty=3)
  points(ferebee[,1]-uu,ferebee[,3]/ferebee[,2]*1e3,col=cls[1],pch=16,cex=1.15)
  for(i in 1:nrow(ferebee)) {
    lines(c(i,i)-uu,qbeta(c(1,39)/40,ferebee[i,3],ferebee[i,2]-ferebee[i,3])*1e3,col=cls[1],lwd=2)
  }
  lines(sutherland[,1]+uu,sutherland[,6]/sutherland[,3]*1e3,col=cls[2],lty=3)
  points(sutherland[,1]+uu,sutherland[,6]/sutherland[,3]*1e3,col=cls[2],pch=16,cex=1.15)
  for(i in 1:nrow(sutherland)) {
    lines(c(i,i)+uu,qbeta(c(1,39)/40,sutherland[i,6],sutherland[i,3]-sutherland[i,6])*1e3,col=cls[2],lwd=2)
  }
  points(9:10,rep(0.059,2),col=cls[2],pch=8,cex=1.1,lwd=1)
  box()
  ##  CUM INC
  par(mar=c(3.5,4.4,0.3,.3))
  id = vID2[[2]]
  uu=0.035
  plot(100,1,xlim=c(0.1,9.9),ylim=c(3,97),las=1,xlab="",ylab="",axes=F)
  axis(1,0:10,las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.4)
  axis(2,las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.4)
  mtext("Year since initial infection",1,2,cex=0.9)
  mtext("Cumulative risk of active TB (%)",2,2.9,cex=1.0)
  polygon(c(0:10,10:0),c(0,apply(CumInc[id,],2,function(x) quantile(x,0.05))[2:11],
                         apply(CumInc[id,],2,function(x) quantile(x,0.95))[11:2],0)*1e2,col="grey90",border="grey88")
  polygon(c(0:10,10:0),c(0,apply(CumInc[id,],2,function(x) quantile(x,0.25))[2:11],
                         apply(CumInc[id,],2,function(x) quantile(x,0.75))[11:2],0)*1e2,col="grey80",border="grey78")
  lines(0:10,c(0,apply(CumInc[id,],2,function(x) quantile(x,0.5))[2:11]*1e2),col="grey45",lwd=2,lend="butt")
  lines(c(0,ferebee[,1])-uu,c(0,cumsum(ferebee[,3])/ferebee[1,2]*1e2),col=cls[1],lty=3)
  points(c(0,ferebee[,1])-uu,c(0,cumsum(ferebee[,3])/ferebee[1,2]*1e2),col=cls[1],pch=16,cex=1.1)
  for(i in 1:nrow(ferebee)) {
    lines(c(i,i)-uu,qbeta(c(1,39)/40,cumsum(ferebee[,3])[i],ferebee[1,2]-cumsum(ferebee[,3])[i])*1e2,col=cls[1],lwd=2)
  }
  lines(c(0,sutherland[,1])+uu,c(0,cumsum(sutherland[,6])/sutherland[1,3])*1e2,col=cls[2],lty=3)
  points(c(0,sutherland[,1])+uu,c(0,cumsum(sutherland[,6])/sutherland[1,3])*1e2,col=cls[2],pch=16,cex=1.1)
  for(i in 1:nrow(sutherland)) {
    lines(c(i,i)+uu,qbeta(c(1,39)/40,cumsum(sutherland[,6])[i],sutherland[1,3]-cumsum(sutherland[,6])[1])*1e2,col=cls[2],lwd=2)
  }
  box()
  ##
  par(mar=c(0,5,0,0.5))
  plot(1,1,col="white",axes=F,xlab="",ylab="")
  legend("center",c("Models: median","Models: 25%-75% interval","Models: 5%-95% interval",
         "Ferebee 1970, mean and 95% interval","Sutherland 1968, mean and 95% interval","Zero new TB cases in year"),lwd=c(2,NA,NA,2,2,NA),
         pt.lwd = c(NA,NA,NA,NA,NA,1),pch=c(NA,15,15,16,16,8),col=c("grey45","grey75","grey90",cls[1:2],cls[2]),
         pt.cex=c(NA,2.7,2.7,1.15,1.15,1.1),ncol=2,cex=1.25)
  dev.off();system(paste("open", pdfnam))


### Figure S1: Histogram of included studies by publication year and subgroup.
  pdfnam <- "Figure_S1_supplement.pdf"
  pdf(file=pdfnam,width=9, height=7.5) 
  par(mfrow=c(3,2),mar=c(1,1,0.5,0.5),oma=c(2,2.5,0,0))

  # All
  plot(0,0,xlim=c(1960,2016),ylim=c(-1,32),axes=F,xlab=NA,ylab=NA)
  abline(h=axTicks(2),col="grey90")
  text(rep(1959,length(axTicks(2)[-1])),axTicks(2)[-1],axTicks(2)[-1],pos=1,col="grey70",cex=1.2)
  axis(1,tick=F,line=0,padj=-2,col.axis="grey70",cex.axis=1.2)
  ww <- table(D2[!duplicated(D2[,"Unique"]),"year"])
  for(i in 1:length(ww)) polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="grey40",border="grey80")
  text(2017,ww[length(ww)],"*",pos=3,offset=-0.2,cex=1.8,col="grey40")
  abline(h=0,col="grey70")
  tt <- "All publications"
  polygon(c(1965,1990,1990,1965),c(29,29,24,24),col="white",border="grey90")
  textbox(c(1965,1990),28,tt,cex=1.3,col="black",border=NA)
  
  #####  Differential by anything
  plot(0,0,xlim=c(1960,2016),ylim=c(-1,32),axes=F,xlab=NA,ylab=NA)
  abline(h=axTicks(2),col="grey90")
  text(rep(1959,length(axTicks(2)[-1])),axTicks(2)[-1],axTicks(2)[-1],pos=1,col="grey70",cex=1.2)
  axis(1,tick=F,line=0,padj=-2,col.axis="grey70",cex.axis=1.2)
  ww <- table(D2[!duplicated(D2[,"Unique"]),"year"])
  for(i in 1:length(ww)) polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="grey80",border="grey95")
  text(2017,ww[length(ww)],"*",pos=3,offset=-0.2,cex=1.8,col="grey80")
  jj = D2[D2$standard==0,]
  ww <- table(jj[!duplicated(jj[,"Unique"]),"year"])
  for(i in 1:length(ww)) polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="red3",border="pink")
  abline(h=0,col="grey70")
  tt <- "Publications with any stratification of progression risks"
  polygon(c(1965,1990,1990,1965),c(29,29,19,19),col="white",border="grey90")
  textbox(c(1965,1990),28,tt,cex=1.3,col="black",border=NA)
  
  #####  Differential by Age
  plot(0,0,xlim=c(1960,2016),ylim=c(-1,32),axes=F,xlab=NA,ylab=NA)
  abline(h=axTicks(2),col="grey90")
  text(rep(1959,length(axTicks(2)[-1])),axTicks(2)[-1],axTicks(2)[-1],pos=1,col="grey70",cex=1.2)
  axis(1,tick=F,line=0,padj=-2,col.axis="grey70",cex.axis=1.2)
  ww <- table(D2[!duplicated(D2[,"Unique"]),"year"])
  for(i in 1:length(ww)) polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="grey80",border="grey95")
  text(2017,ww[length(ww)],"*",pos=3,offset=-0.2,cex=1.8,col="grey80")
  jj = D2[D2$child==1,]
  ww <- table(jj[!duplicated(jj[,"Unique"]),"year"])
  for(i in 1:length(ww)) polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="red3",border="pink")
  abline(h=0,col="grey70")
  tt <- "Publications with age-stratification of progression risks"
  polygon(c(1965,1990,1990,1965),c(29,29,19,19),col="white",border="grey90")
  textbox(c(1965,1990),28,tt,cex=1.3,col="black",border=NA)
  
  ####  Differential by HIV
  plot(0,0,xlim=c(1960,2016),ylim=c(-1,32),axes=F,xlab=NA,ylab=NA)
  abline(h=axTicks(2),col="grey90")
  text(rep(1959,length(axTicks(2)[-1])),axTicks(2)[-1],axTicks(2)[-1],pos=1,col="grey70",cex=1.2)
  axis(1,tick=F,line=0,padj=-2,col.axis="grey70",cex.axis=1.2)
  ww <- table(D2[!duplicated(D2[,"Unique"]),"year"])
  for(i in 1:length(ww)) polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="grey80",border="grey95")
  text(2017,ww[length(ww)],"*",pos=3,offset=-0.2,cex=1.8,col="grey80")
  jj = D2[D2$hiv==1,]
  ww <- table(jj[!duplicated(jj[,"Unique"]),"year"])
  for(i in 1:length(ww)) polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="red3",border="pink")
  abline(h=0,col="grey70")
  tt <- "Publications with HIV-stratification of progression risks"
  polygon(c(1965,1990,1990,1965),c(29,29,19,19),col="white",border="grey90")
  textbox(c(1965,1990),28,tt,cex=1.3,col="black",border=NA)
  
  #####  Differential by Other
  plot(0,0,xlim=c(1960,2016),ylim=c(-1,32),axes=F,xlab=NA,ylab=NA)
  abline(h=axTicks(2),col="grey90")
  text(rep(1959,length(axTicks(2)[-1])),axTicks(2)[-1],axTicks(2)[-1],pos=1,col="grey70",cex=1.2)
  axis(1,tick=F,line=0,padj=-2,col.axis="grey70",cex.axis=1.2)
  ww <- table(D2[!duplicated(D2[,"Unique"]),"year"])
  for(i in 1:length(ww)) polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="grey80",border="grey95")
  text(2017,ww[length(ww)],"*",pos=3,offset=-0.2,cex=1.8,col="grey80")
  jj = D2[D2$elevated_risk==1 & D2$hiv==0 & D2$child==0,]
  ww <- table(jj[!duplicated(jj[,"Unique"]),"year"])
  for(i in 1:length(ww)) polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="red3",border="pink")
  abline(h=0,col="grey70")
  
  tt <- "Publications with other stratification of progression risks"
  polygon(c(1965,1990,1990,1965),c(29,29,19,19),col="white",border="grey90")
  textbox(c(1965,1990),28,tt,cex=1.3,col="black",border=NA)
  
  #####  High-burden setting
  plot(0,0,xlim=c(1960,2016),ylim=c(-1,32),axes=F,xlab=NA,ylab=NA)
  abline(h=axTicks(2),col="grey90")
  text(rep(1959,length(axTicks(2)[-1])),axTicks(2)[-1],axTicks(2)[-1],pos=1,col="grey70",cex=1.2)
  axis(1,tick=F,line=0,padj=-2,col.axis="grey70",cex.axis=1.2)
  ww <- table(D2[!duplicated(D2[,"Unique"]),"year"])
  for(i in 1:length(ww)) polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="grey80",border="grey95")
  text(2017,ww[length(ww)],"*",pos=3,offset=-0.2,cex=1.8,col="grey80")
  jj = D2[D2$high_burden==1,]
  ww <- table(jj[!duplicated(jj[,"Unique"]),"year"])
  for(i in 1:length(ww)) polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="red3",border="pink")
  abline(h=0,col="grey70")
  
  tt <- "Publications focused on high-burden settings"
  polygon(c(1965,1990,1990,1965),c(29,29,22,22),col="white",border="grey90")
  textbox(c(1965,1990),28,tt,cex=1.3,col="black",border=NA)
  
  ##
  mtext("Number of Publications",2,1,outer=T)
  mtext("Publication Year",1,.8,outer=T)
  
  dev.off();system(paste("open", pdfnam))

  
###  Figure S2: Histogram of included studies by publication year and model structure.
  pdfnam <- "Figure_S2_supplement.pdf"
  pdf(file=pdfnam,width=9, height=11) 
  par(mfrow=c(6,2),mar=c(1,1,0.5,0.5),oma=c(2,2.5,0,0))
  ######
  for(j in 1:length(unique(D2$model_structure))) { # j=1
    plot(0,0,xlim=c(1960,2016),ylim=c(-1,32),axes=F,xlab=NA,ylab=NA)
    abline(h=axTicks(2),col="grey90")
    text(rep(1959.5,length(axTicks(2)[-1])),axTicks(2)[-1]+.7,axTicks(2)[-1],pos=1,col="grey70",cex=1.2)
    axis(1,tick=F,line=0,padj=-2,col.axis="grey70",cex.axis=1.2)
    ww <- table(D2[!duplicated(D2[,"Unique"]),"year"])
    for(i in 1:length(ww)) {
      polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="grey80",border="grey95")
    }
    text(2017,ww[length(ww)],"*",pos=3,offset=-0.2,cex=1.8,col="grey80")
    jj = D2[D2$model_structure==unique(D2$model_structure)[j],]
    ww <- table(jj[!duplicated(jj[,"Unique"]),"year"])
    for(i in 1:length(ww)) {
      polygon(c(-1,1,1,-1)*0.5+as.numeric(names(ww))[i],c(0,0,ww[i],ww[i]),col="blue1",border="lightblue")
    }
    abline(h=0,col="grey70")
    tt <- paste(" Structure",unique(D2$model_structure)[j])
    polygon(c(1965,1987,1987,1965),c(28,28,23,23),col="white",border="grey90")
    textbox(c(1965,1995),26.5,tt,cex=1.4,col="black",border=NA)
  }
  mtext("Number of Publications",2,1,outer=T)
  mtext("Publication Year",1,.8,outer=T)
  dev.off();system(paste("open", pdfnam))

### Figure S3: Most cited sources for parameters describing progression from infection to active TB disease.
  PubPerCit <- colSums(PubCit)
  vec <- c(sum(rowSums(PubCit)==0),PubPerCit[order(-PubPerCit)][1:14])
  names(vec) <- c("No citations given",names(PubPerCit[order(-PubPerCit)][1:14]))
  vec # top 15
  vecCat <- vec # 0 = no citation given 1 = modeling, 2 = empirical, 3 = review, 4 = 0ther
  vecCat["No citations given"] <- 0
  vecCat[c("Vynnycky & Fine 1997","Blower 1995","Dye 1998","Dye & Williams 2000")] <- 1
  vecCat[c("Horsburgh 2010","Daley 1992","Sutherland 1982","Sutherland 1968","Di Perri 1989")] <- 2
  vecCat[c("Ferebee 1970","Comstock 1982","Styblo 1991","Styblo 1986")] <- 3
  vecCat["Dye 1999"] <- 4
  
  names(vec)[names(vec)=="Blower 1995"] <- "Blower et al 1995"
  names(vec)[names(vec)=="Dye 1998"] <- "Dye et al 1998"
  names(vec)[names(vec)=="Horsburgh 2010"] <- "Horsburgh et al 2010"
  names(vec)[names(vec)=="Daley 1992"] <- "Daley et al 1992"
  names(vec)[names(vec)=="Sutherland 1982"] <- "Sutherland et al 1982"
  names(vec)[names(vec)=="Dye 1999"] <- "Dye et al 1999"
  names(vec)[names(vec)=="Di Perri 1989"] <- "Di Perri et al 1989"
  
  pdfnam <- "Figure_S3_supplement.pdf"
  cls <-  c("grey30",brewer.pal(4, "Set1")[4:1])
  pdf(file=pdfnam,width=10, height=6.0) 
  par(mar=c(5,10,1,1))
  zz <- barplot(vec[15:1],horiz=T,las=1,col=mTrsp(cls[vecCat[15:1]+1],100),border=mTrsp(cls[vecCat[15:1]+1],150))
  text(vec[15:1],zz,vec[15:1],pos=2,col=mTrsp(cls[vecCat[15:1]+1],200))
  legend(39,5,c("No citations given","Transmission modelling study","Empirical study","Review study","Other study"),pch=22,
         col=mTrsp(cls,150),pt.bg=mTrsp(cls,100),cex=1.0,pt.cex=2.6,bty="n")
  mtext("Number of publications citing each source",1,3,cex=1.1)
  dev.off();system(paste("open", pdfnam))
  

###  Figure S4: Model predictions for annual and cumulative incidence of active TB by 
###  years since M. tb infection, for groups with no individual risk factors: 
###  median for each model type.  
  
  pdfnam <- "Figure_S4_supplement.pdf"
  cls <-  c(brewer.pal(9, "Set1")[1:4],"forestgreen",brewer.pal(8, "Dark2")[c(2:3,7)],"grey50","blue","hotpink","goldenrod","black")
  pdf(file=pdfnam,width=10, height=6) 
  ly <- rbind(cbind(rep(1,6),rep(2,6)),c(3,3))
  layout(ly)
  # INC RATE
  par(oma=c(0,0.5,0.0,0),mar=c(3.5,4.8,0.3,.3))
  id <- vID2[[2]]
  zz <- IncRate[id,]
  plot(100,1,xlim=c(.5,19.5),ylim=c(0.0095,240),las=1,xlab="",ylab="",log="y",axes=F)
  axis(1,las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.4)
  axis(2,10^(-3:2),as.character(10^(-3:2)),las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.4)
  mtext("Years since initial infection",1,2,cex=0.9)
  mtext("Annual risk of active TB (per 1000, log-scale)",2,3.9,cex=1.0)
  for(i in (1:12)[-4]) { # H removed
    k <- which(D2$model_structure[id]==c("A","B","C","D","E","F","G","H","I","J","K","L")[i])
    if(length(k)>1){
      lines(1:20-.5,apply(zz[k,],2,function(x) quantile(x,0.5))[1:20]*1e3,col=cls[i],lwd=2,lend="butt")
    } else {
      lines(1:20-.5, zz[k,1:20]*1e3,col=cls[i],lwd=2) 
    }
  }
  lines(1:20-.5,apply(zz,2,function(x) quantile(x,0.5))[1:20]*1e3,col="black",lwd=2,lend="butt",lty="22")
  box()
  # CUM INC
  kk <- NULL
  for(k in 1:12) {
    kk[k] <- length(which(D2$model_structure[id]==c("A","B","C","D","E","F","G","H","I","J","K","L")[k]))
  }
  par(mar=c(3.5,5.3,0.3,.3))
  id = vID2[[2]]
  plot(100,1,xlim=c(.25,19.75),ylim=c(2,62),las=1,xlab="",ylab="",axes=F)
  axis(1,las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.4)
  axis(2,las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.4)
  mtext("Years since initial infection",1,2,cex=0.9)
  mtext("Cumulative risk of active TB (%)",2,3,cex=1.0)
  zz  <- CumInc[id,]
  for(i in (1:11)[-4]) { # H removed
    k <- which(D2$model_structure[id]==c("A","B","C","D","E","F","G","H","I","J","K","L")[i])
    if(length(k)>1){
      lines(0:20,c(0,apply(zz[k,],2,function(x) quantile(x,0.5))[2:21]*1e2),col=cls[i],lwd=2,lend="butt")
    } else {
      lines(0:20, c(0,zz[k,2:21])*100,col=cls[i],lwd=2) 
    }
  }
  lines(0:20,c(0,apply(zz,2,function(x) quantile(x,0.5))[2:21]*1e2),col="black",lwd=2,lend="butt",lty="22")
  box()
  ##
  par(mar=c(0,0.5,0,0.5))
  plot(1,1,col="white",axes=F,xlab="",ylab="")
  legend("center",c(paste("Structure ",c("A","B","C","D","E","F","G","H","I","J","K","L")[-4]," (",kk[-4],")",sep=""),"Median of all (380)"),
         lwd=c(rep(1,11),2),col=c(cls[-4],1),pch=c(rep(15,11),NA),pt.cex=c(rep(2.7,11),NA),
         lty=c(rep(NA,11),"22"),ncol=4,cex=1.3)
  ###
  dev.off();system(paste("open", pdfnam))
  
  
###  Figure S5: Model predictions for relative risk of active TB during first 
###  and twentieth year since M. tb infection, for named risk factors*.

  ttl <- c("Infants (0-2 years)\ncompared to adults*",
           "Children (excl. infants)\ncompared to adults*",
           "HIV compared to\nnon-HIV",
           "Early HIV compared\nto non-HIV",
           "Late HIV compared\nto non-HIV",
           "HIV with ART compared\nto HIV without ART")[6:1]

  jjL2  <- list(RRinf2,
                RRchild2,
                RRhiv2,
                RRhear2,
                RRhlat2,
                RRhart2)[6:1]

  jjL20 <- list(RRinf1_inc[is.na(RRinf1_inc)==F],
                RRchild1_inc[is.na(RRchild1_inc)==F],
                RRhiv20_inc[is.na(RRhiv20_inc)==F & RRhiv20_inc>0],
                RRhear20_inc[is.na(RRhear20_inc)==F],
                RRhlat20_inc[is.na(RRhlat20_inc)==F],
                RRhart20_inc[is.na(RRhart20_inc)==F])[6:1]

  pdfnam <- "Figure_S5_supplement.pdf"
  pdf(file=pdfnam,width=9, height=5.5) 
  par(mfrow=c(1,2),mar=c(0,0,0,0),oma=c(3,8.5,1.5,.5))
  cls <-  brewer.pal(8, "Pastel2")
  cl2 <- cls[c(4,4,4,4,5,5)]
  ## 1st 2 years
  ty <- min(as.numeric(lapply( jjL2,min)))/1.6
  plot(1,-10,xlab="",ylab="",axes=F,ylim=c(0.5,6.5),xlim=c(ty,max(as.numeric(lapply( jjL2,max)))),log="x")
  abline(v=1,col="grey70",lty="22")
  abline(v=ty*1.5,col="grey90",lty="11")
  for(i in 1:length(jjL2)) {
    lines(c(ty*1.5,1000),c(i,i),col="grey90",lty="11")
  }
  boxplot(jjL2,col=cl2,axes=F,add=T,horizontal=T)
  text(rep(ty*1.1,length(jjL2)),1:length(jjL2),format(round(as.numeric(lapply(jjL2,median)),2),nsmall=2),cex=0.8)
  box()
  axis(1,axTicks(1)[-1],as.character(axTicks(1)[-1]),las=1,tcl=-.25,mgp=c(3, 0.5, 0),cex.axis=.95)
  axis(1,ty*1.1,"median",las=2,tcl=-.25,mgp=c(3, 0.5, 0),cex.axis=.85)
  mtext("Relative risk of TB in first two years (log-scale)",3,.5,cex=0.9)
  tlab <- 1:length(jjL20) # where labels
  ttl2 <- paste(ttl," (",as.numeric(lapply(jjL2,length)),")",sep="")
  axis(2,tlab,ttl2,las=1,cex.axis=0.85,tcl=-.25,mgp=c(3, 0.7, 0))
  ## 20 years
  ty <- min(as.numeric(lapply( jjL20[-(5:6)],min)))/2.0
  plot(1,-10,xlab="",ylab="",axes=F,ylim=c(0.5,6.5),xlim=c(ty,max(as.numeric(lapply( jjL20,max)))),log="x")
  abline(v=1,col="grey70",lty="22")
  abline(v=ty*1.8,col="grey90",lty="11")
  for(i in 1:length(jjL20)) {
    lines(c(ty*1.8,1000),c(i,i),col="grey90",lty="11")
  }
  boxplot(jjL20[-(5:6)],col=cl2,axes=F,add=T,horizontal=T)
  text(rep(ty*1.1,length(jjL20)-2),1:(length(jjL20)-2),signif(as.numeric(lapply(jjL20,median)),c(2,3,3,3,2))[-(5:6)],cex=0.8)
  box()
  axis(1,axTicks(1)[-1],as.character(axTicks(1)[-1]),las=1,tcl=-.25,mgp=c(3, 0.5, 0),cex.axis=0.95)
  axis(1,ty*1.1,"median",las=2,tcl=-.25,mgp=c(3, 0.5, 0),cex.axis=.85)
  mtext("Relative risk of TB in twentieth year (log-scale)",3,.5,cex=0.9)
  dev.off();system(paste("open", pdfnam))

  
  
###  Figure S6: Cumulative TB incidence projections for each model structure
###  with parameters fitted to empirical data (Sutherland 1968)*.
  pdfnam <- "Figure_S6_supplement.pdf"
  cls <-  c(brewer.pal(9, "Set1")[1:4],"forestgreen",brewer.pal(8, "Dark2")[c(2:3,7)],"grey50","blue","hotpink","goldenrod")
  pdf(file=pdfnam,width=9, height=6) 
  par(oma=c(0,2,0,0),mar=c(3.0,2.0,0.5,.5))
  uu=0.035
  plot(100,1,xlim=c(0.1,9.9),ylim=c(0.25,12.3),las=1,xlab="",ylab="",axes=F)
  axis(1,0:10,las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.1)
  axis(2,las=1,tcl=-.25,mgp=c(3, 0.6, 0),cex.axis=1.1)
  mtext("Cumulative incidence (%)",2,2.7,cex=1)
  mtext("Year since initial infection",1,1.8,cex=1)
  for(i in 1:12) {
    lines(0:10,CumIncRes[[i]]+i/100,col=cls[i],lwd=2)
  }
  lines(c(0,sutherland[,1])+uu,c(0,cumsum(sutherland[,6])/sutherland[1,3])*1e2,col="white",lty=1,lwd=2.5)
  lines(c(0,sutherland[,1])+uu,c(0,cumsum(sutherland[,6])/sutherland[1,3])*1e2,col=1,lty="22",lwd=2.3)
  text(.05,11,"Structure D",pos=4,cex=0.9,col=cls[4])
  text(2.9,7.3,"Structure E",pos=4,cex=0.9,col=cls[5])
  text(8.9,11,"Structure A",pos=4,cex=0.9,col=cls[1])
  text(7.9,11.7,"Structure J",pos=4,cex=0.9,col=cls[10])
  
  box()
  legend("bottomright",c(paste("Structure ",c("A","B","C","D","E","F","G","H","I","J","K","L"),sep=""),"Sutherland 1968"),
         lwd=c(rep(1,12),2),col=c(cls,1),pch=c(rep(15,12),NA),pt.cex=c(rep(2.3,12),NA),
         lty=c(rep(NA,12),"22"),ncol=2,cex=1.0)
  dev.off();system(paste("open", pdfnam))
  

