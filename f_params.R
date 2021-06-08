
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 08.06.2021
# Train classifier for lesion classification
# ======================================================================================================================

#' Extract model parameters as high-level features for lesion classification
#' @param dat A dataframe holding the data for the color channels of interest: v_Luv and H_HSV
#' @return A dataframe with the model parameters
#' @examples
get_params <- function(dat){
  
  # ======================================================================================================================
  # AUXILIARY FUNCTIONS
  # ======================================================================================================================
  
  # Four-parameter Logistic equation
  logistic <- function(c, d, b, e, posY) {
    grenness <- c + (d-c)/(1 + exp(b*(posY-e)))
    return(grenness)
  }
  
  # Breakpoint model
  breakpoint_lm <- function(data){
    model <- lm(mean~posY, data) # The normal linear regression model serves as a basis.
    model.seg <- segmented(model, seg.Z = ~posY) # Give predictor to break and starting value
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
    return(data.frame(slp1 = slp1, slp2 = slp2, slprat = slp_rat, slpdiff = slp_diff))
  }
  
  # ======================================================================================================================
  # ======================================================================================================================
  
  # Reformat input dataframe 
  DATA <- t(dat) %>% as.data.frame() %>% rownames_to_column()
  colnames(DATA) <- c("rowname", "V1")
  DATA$channel <- strsplit(names(dat),"_") %>% lapply("[[", 1) %>% unlist()
  DATA$color_space <- strsplit(names(dat),"_") %>% lapply("[[", 2) %>% unlist()
  DATA$posY <-  strsplit(names(dat),"_") %>% lapply("[[", 3) %>% unlist() %>% 
    # conversion to r_df transforms "-" in name to "." --> re-transform
    gsub("\\.", "-", .) %>% as.numeric()
  # column names are not identical when transforming to r_df --> adjust
  DATA <- DATA %>% mutate(mean = V1)
  #select required columns
  dat <- DATA %>% dplyr::select(channel, color_space, posY, mean)

  # ======================================================================================================================
  
  # Breakpoint model for v_Luv ----
  # select data
  data_v <- dat %>% 
    filter(channel == "v", color_space == "Luv") %>% 
    mutate(posY = posY + 32)
  
  # fit breakpoint model
  data_fits <- data_v %>%
    dplyr::select(posY, mean) %>%
    tidyr::nest(data = c(posY, mean)) %>%
    mutate(fit_bp_lm = purrr::map(data, breakpoint_lm)) %>%
    tidyr::unnest(fit_bp_lm)
  bp_pars_v <- data_fits %>% dplyr::select(slp1:slpdiff)
  names(bp_pars_v)[2:length(bp_pars_v)] <- paste0(names(bp_pars_v)[2:length(bp_pars_v)], "_v")
  
  # ======================================================================================================================
 
   # Breakpoint model for R_RGB ----
  # select data
  data_R <- all_means %>% 
    filter(channel == "R", color_space == "RGB") %>% 
    group_by(label, .id)
  
  ## fit breakpoint models
  data_fits <- data_R %>%
    dplyr::select(.id, label, posY, mean) %>% 
    as.data.frame() %>%
    tidyr::nest(data = c(posY, mean)) %>%
    group_by(.id) %>%
    mutate(fit_bp_lm = purrr::map(data, breakpoint_lm)) %>% 
    unnest(fit_bp_lm)
  
  bp_pars_R <- data_fits %>% dplyr::select(.id, slp1:slpdiff)
  names(bp_pars_R)[2:length(bp_pars_R)] <- paste0(names(bp_pars_R)[2:length(bp_pars_R)], "_R")
  
  # ======================================================================================================================
  
  # Logistic model for H_HSV ----
  # select data
  data_H <- dat %>%
    filter(channel == "H", color_space == "HSV") %>%
    mutate(posY = posY + 32)

  # fit logistic model
  data_fits <- data_H %>%
    dplyr::select(posY, mean) %>%
    tidyr::nest(data = c(posY, mean)) %>%
    mutate(fit_log = purrr::map(data,
                                ~ nls_multstart(mean ~ logistic(c, d, b, e, posY = posY),
                                                data = .x,
                                                iter = 750,
                                                start_lower = c(c = -0.5, d = 0.5, e = 20, b = -0.5),
                                                start_upper = c(c = 0.5, d = 1.5, e = 44, b = 0),
                                                convergence_count = 150,
                                                supp_errors = 'Y'))) %>%
    tidyr::gather(met, fit, fit_log:fit_log)

  # gather model parameters
  mod_pars <- data_fits %>%
    #reconverto to wide
    dplyr::filter(met == "fit_log") %>%
    tidyr::spread(met, fit) %>%
    mutate(p = purrr::map(fit_log, broom::tidy)) %>%
    tidyr::unnest(p) %>%
    dplyr::select(1:4) %>%
    tidyr::spread(term, estimate)
  log_pars <- mod_pars %>% dplyr::select(b, c, d, e)
  
  # ======================================================================================================================
  
  # combine output
  params <- bind_cols(bp_pars_v, bp_pars_R, log_pars)
  params <- as.data.frame(params)

  return(params)
  
}

# ======================================================================================================================
