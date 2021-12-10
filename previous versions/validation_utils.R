#cramer (correlation) table function - vars is a string vector of the variables of interest
cramerfunc <- function(vars, dat) sapply(vars, function(y) sapply(vars, function(x) assocstats(table(dat[,x], dat[,y]))$cramer))

#KMO test (Kaiser-Meyer-Olkin Measure of Sampling Adequacy)
# Function by G. Jay Kerns, Ph.D., Youngstown State University (http://tolstoy.newcastle.edu.au/R/e2/help/07/08/22816.html)
kmo = function( data ){
  library(MASS, lib.loc = "/usr/local/lib/R/site-library/") 
  X <- cor(as.matrix(data)) 
  iX <- ginv(X) 
  S2 <- diag(diag((iX^-1)))
  AIS <- S2%*%iX%*%S2                      # anti-image covariance matrix
  IS <- X+AIS-2*S2                         # image covariance matrix
  Dai <- sqrt(diag(diag(AIS)))
  IR <- ginv(Dai)%*%IS%*%ginv(Dai)         # image correlation matrix
  AIR <- ginv(Dai)%*%AIS%*%ginv(Dai)       # anti-image correlation matrix
  a <- apply((AIR - diag(diag(AIR)))^2, 2, sum)
  AA <- sum(a) 
  b <- apply((X - diag(nrow(X)))^2, 2, sum)
  BB <- sum(b)
  MSA <- b/(b+a)                        # indiv. measures of sampling adequacy
  AIR <- AIR-diag(nrow(AIR))+diag(MSA)  # Examine the anti-image of the correlation matrix. That is the  negative of the partial correlations, partialling out all other variables.
  kmo <- BB/(AA+BB)                     # overall KMO statistic
  # Reporting the conclusion 
  if (kmo >= 0.00 && kmo < 0.50){test <- 'The KMO test yields a degree of common variance unacceptable for FA.'} 
  else if (kmo >= 0.50 && kmo < 0.60){test <- 'The KMO test yields a degree of common variance miserable.'} 
  else if (kmo >= 0.60 && kmo < 0.70){test <- 'The KMO test yields a degree of common variance mediocre.'} 
  else if (kmo >= 0.70 && kmo < 0.80){test <- 'The KMO test yields a degree of common variance middling.' } 
  else if (kmo >= 0.80 && kmo < 0.90){test <- 'The KMO test yields a degree of common variance meritorious.' }
  else { test <- 'The KMO test yields a degree of common variance marvelous.' }
  
  ans <- list( overall = kmo,
               report = test,
               individual = MSA,
               AIS = AIS,
               AIR = AIR )
  return(ans)
} 

check_pca = function(cor_matrix, model){
  #find the residuals matrix, are they problematic?
  residuals <- factor.residuals(cor_matrix, model$loadings)
  
  #find residuals that are above 0.05 (considered large residuals)
  large.resid <- abs(residuals) > 0.05
  cat('Number of large residuals:')
  print(sum(large.resid))
  cat('Proportion of large residuals (should be less than .50): ')
  #find percentage of large residuals (should be less than 50%)
  print(sum(large.resid)/(nrow(residuals)*ncol(residuals))) #percentage large resid = 0.3846154%
  cat('Mean residual value (should be less than 0.08): ')
  #find the mean residual value (should be less than 0.08)
  print(sqrt(mean(as.matrix(residuals^2)))) #0.1024799 - problematic?
  
  #plotting the residuals
  hist(as.matrix(residuals), breaks = 20, xlim=c(-0.5, 1)) 
  # X <- cor(as.matrix(data)) 
  # iX <- ginv(X) 
}

check_cfa = function(model, show_mod_ind=FALSE) {
  cat('Goodness of fit:\n')
  print(fitMeasures(model)[c('cfi', 'rmsea','srmr', 'gfi')])
  if (show_mod_ind){
    cat('Modification indices:\n')
    print(modindices(model, sort. = T)[c(1:15),c(1:4, 7)])
  }
}