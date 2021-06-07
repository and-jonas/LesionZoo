
dat <- read_csv("Z:/Public/Jonas/001_LesionZoo/test.csv")




get_params <- function(dat){
  
  datt <- t(dat) %>% as.data.frame() %>% rownames_to_column() %>% as_tibble()
  colnames(datt) <- c("rowname", "V1")
  
  datt$channel <- strsplit(names(dat),"_") %>% lapply("[[", 1) %>% unlist()
  datt$color_space <- strsplit(names(dat),"_") %>% lapply("[[", 2) %>% unlist()
  datt$posY <-  strsplit(names(dat),"_") %>% lapply("[[", 3) %>% unlist() %>% as.numeric()
  datt <- datt %>% mutate(mean = V1)
  
  dings <- datt %>% dplyr::select(channel, color_space, posY, mean)
  dat <- dings
  
  # Breakpoint model for v_Luv ----
  sub1 <- dat %>% 
    filter(channel == "v",
           color_space == "Luv")
  
  data_fits <- sub1 %>%
    dplyr::select(posY, mean) %>% 
    tidyr::nest(data = c(posY, mean)) %>%
    mutate(fit_bp_lm = purrr::map(data, breakpoint_lm)) %>% 
    unnest(fit_bp_lm)
  bp_pars <- data_fits %>% dplyr::select(slp1:slpdiff)
  
  # Logistic model for H_HSV ----
  sub2 <- dat %>% 
    filter(channel == "H",
           color_space == "HSV") %>% 
    mutate(posY = posY + 32)
  
  #add model fits
  data_fits <- sub2 %>%
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
  
  # get model parameters
  mod_pars <- data_fits %>%
    #reconverto to wide
    dplyr::filter(met == "fit_log") %>% 
    tidyr::spread(met, fit) %>% 
    mutate(p = purrr::map(fit_log, broom::tidy)) %>% 
    unnest(p) %>% 
    dplyr::select(1:4) %>% 
    tidyr::spread(term, estimate) 
  log_pars <- mod_pars %>% dplyr::select(b, c, d, e)
  
  params <- bind_cols(bp_pars, log_pars)
  
  return(params)
  
}
