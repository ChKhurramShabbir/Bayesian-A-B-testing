# Import the necessary libraries
library(cmdstanr)
library(tidyverse)
library(posterior)

# Read in the data
df <- arrow::read_parquet("data/ab_test_data.parquet")

# Calculate sufficient statistics
df_agg <- df %>%
  group_by(arm_id, treat) %>%
  summarise(
    conversions = sum(conversions),
    trials = sum(trials)
  )

# Compile the Stan model
install_cmdstan(cores = 6)
model <- cmdstan_model(
    "models/binomial_conversions.stan",
    include_paths = "C:/Users/adamn/Dropbox/bayes-workshop/models/"
    )

cmdstanr::cmdstan_model(in)
# Data for the model
stan_data <- list(
   N = nrow(df_agg),
   Y = df_agg$conversions,
   K = df_agg$trials,
   alpha_prior = c(1, 1),
   beta_prior = c(1, 1)
)

# Fit the model
fit <- model$sample(
    data = stan_data, 
    chains = 4, 
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    in
)

# Extract the posterior draws
draws <- fit$draws(format = "draws_df")