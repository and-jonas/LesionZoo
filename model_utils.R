
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 08.06.2021
# Model equations, function for breakpoint model fitting
# ======================================================================================================================

# Contrained Gompertz equation
Gompertz_constrained <- function(B, alpha, b, M, posY) {
  value <- B + alpha *exp(-exp(-b*(posY-M)))
  return(value)
}

# Unconstrained four-parameter gompertz equation
gompertz <- function(posY, alpha, beta, k) {
  result <- alpha * exp(-beta * exp(-k * t));
  return(result)
}

# Contrained Gompertz equation
Gompertz_constrained <- function(b, M, posY) {
  value <- 1*exp(exp(-b*(posY-M)))
  return(value)
}

# Four-parameter Logistic equation
logistic <- function(A, C, b, M, posY) {
  value <- A + C/(1+exp(-b*(M-posY)))
  return(value)
}

# Four-parameter Logistic equation
logistic_c <- function(b, M, posY) {
  value <- 1/(1+exp(-b*(posY-M)))
  return(value)
}

# Four-parameter Logistic equation
logistic <- function(c, d, b, e, posY) {
  value <- c + (d-c)/(1 + exp(b*(posY-e)))
  return(value)
}

# Weibull equation
weibull <- function(c, a, b, posY) {
  value <- c*(abs(posY)/a)^(b-1)*exp(-(abs(posY)/a)^b)
  return(value)
}

# Breakpoint model
breakpoint_lm <- function(data, returnBreakpoint=FALSE){
  data$posY <- data$posY + 32
  model <- lm(mean~posY, data) # The normal linear regression model serves as a basis.
  model.seg <- segmented(model, seg.Z = ~posY, psi = 32) # Give predictor to break and starting value
  breakpoint <- summary.segmented(model.seg)$psi[1,2]
  if(breakpoint >=20 & breakpoint <=44){
    slp1 <- slope(model.seg)$posY[,1][1] %>% unname()
    slp2 <- slope(model.seg)$posY[,1][2] %>% unname() # Estimated slopes below and above breakpoint with 95% CI
    slp_rat <- slp1/slp2
    slp_diff <- slp1 - slp2
  } else{
    sub1 <- data[data$posY <=32,]
    sub2 <- data[data$posY >=32,]
    lm1 <- lm(mean ~ posY, sub1)
    slp1 <- summary(lm1)$coefficients[2]
    lm2 <- lm(mean ~ posY, sub2)
    slp2 <- summary(lm2)$coefficients[2]
    slp_rat <- slp1/slp2
    slp_diff <- slp1 - slp2
  }
  if(returnBreakpoint==TRUE){
    return(data.frame(slp1 = slp1, slp2 = slp2, slprat = slp_rat, slpdiff = slp_diff, bpoint=breakpoint))
  } else {
    return(data.frame(slp1 = slp1, slp2 = slp2, slprat = slp_rat, slpdiff = slp_diff))
  }
}
