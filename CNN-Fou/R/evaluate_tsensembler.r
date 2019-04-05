library(tsensembler)
library(data.table)
library(Metrics)
library(ggplot2)


train_ratio = 0.9
seqlen <- 50 # no lags
seqlen <- seqlen+1 # authors use a definition that differs by 1 from definition in the paper

standardize <- function(dat){
  train <- dat[1:floor(length(dat)*train_ratio)]
  sd <- sd(train)
  m <- mean(train)
  dat <- (dat-m)/sd
  return (dat)
}

fc_tsensembler <- function(train, test, fc_horizon){
  specs <- model_specs(
    learner = c("bm_ppr","bm_glm","bm_svr","bm_mars"), 
    learner_pars = list(
      bm_glm = list(alpha = c(0, .5, 1)),
      bm_svr = list(kernel = c("rbfdot", "polydot"),
                    C = c(1,3)),
      bm_ppr = list(nterms = 4)
    ))
  model <- ADE(target ~., train, specs)
  # predict
  predictions <- predict(model, test)
  rmse <- rmse(test[, "target"], predictions@y_hat)
  return (predictions@y_hat)
}

# Default: Rolling forecasts
evaluate_tsensembler_rolling <- function(train, test){
  preds <- fc_tsensembler(train, test, len(test))
  actuals <- test[, "target"]
  res_rmse <- rmse(actuals, preds)
  cat("\ntsensembler RMSE:", as.character(round(res_rmse, digits=3)))
  return (list(preds, actuals, res_rmse))
}

# Adapt standard prediction logic to multi-step forecasting
evaluate_tsensembler_multistep <- function(train, test, fc_horizon){
  specs <- model_specs(
    learner = c("bm_ppr","bm_glm","bm_svr","bm_mars"), 
    learner_pars = list(
      bm_glm = list(alpha = c(0, .5, 1)),
      bm_svr = list(kernel = c("rbfdot", "polydot"),
                    C = c(1,3)),
      bm_ppr = list(nterms = 4)
    ))
  model <- ADE(target ~., train, specs)
  # predict
  predictions <- predict(model, test)
  rmse <- rmse(test[, "target"], predictions@y_hat)
  
  preds_init <- predictions@y_hat
  test <- as.data.table(test)
  test[, next_prediction := preds_init]
  preds_all <- list() # each list entry is of length fc_horizon
  actuals_all <- list() # each list entry is of length fc_horizon
  for (test_row_index in 1:(nrow(test)-fc_horizon)){
    test_row_preds <- c()
    test_observation <- test[test_row_index]
    preds <- test_observation[, next_prediction] # the first of fc_horizon forecasts. Serves as input for the next forecast (indirect method)
    for (fc_index in 1:(fc_horizon-1)){
      # Update lags (shift old ones and insert new one)
      old_values <- as.numeric(test_observation[1])
      updated_values <- copy(old_values)
      updated_values[2] <- preds[length(preds)]
      updated_values[3:(length(updated_values))] <- old_values[2:(length(old_values)-1)]
      updated_values <- updated_values[2:length(updated_values)]
      test_observation[, names(test_observation)[2:ncol(test_observation)] := as.list(updated_values)]
      # Score model
      pred <- predict(model, test_observation)
      pred <- pred@y_hat
      preds <- c(preds, pred)
      if (fc_index == (fc_horizon-1)){
        # all predictions are done. Store them and go to next test row.
        preds_all[[test_row_index]] <- preds
        # add actuals
        actuals_all[[test_row_index]] <- test[test_row_index:(test_row_index+fc_horizon-1), target]
      }
    }
  }
  res_rmse <- rmse(unlist(actuals_all), unlist(preds_all))
  cat("\ntsensembler RMSE:", as.character(round(res_rmse, digits=3)))
  return (list(preds_all, actuals_all, res_rmse))
}


# 30 data sets evaluation
load("tseries.rdata")
rmses <- data.table(dataset=1:length(tseries), value=NA_real_)
i <- 1
for (dat in tseries){
  dat <- dat["target"]
  train_lim <- floor((nrow(dat))*train_ratio)-seqlen-1
  test_size <- nrow(dat)-train_lim
  
  dat <- standardize(dat[, "target"])
  
  dat_embed <- embed_timeseries(dat, seqlen)
  train_embed <- dat_embed[1:(nrow(dat_embed)-test_size),]
  test_embed <- dat_embed[(nrow(dat_embed)-test_size+1):nrow(dat_embed),]
  
  res <- evaluate_tsensembler_multistep(train_embed, test_embed, fc_horizon = 10)
  preds <- res[[1]]
  actuals <- res[[2]]
  
  rmses[dataset==i, value := res[[3]]]
  i <- i+1
}

# Plot predictions
i_ts <- 1
ggplot() + geom_line(aes(x=1:length(preds[[i_ts]]),y=preds[[i_ts]]), color='red') + 
  geom_line(aes(x=1:length(preds[[i_ts]]),y=actuals[[i_ts]]), color='blue') + 
  ylab('Value')+xlab('Time')
