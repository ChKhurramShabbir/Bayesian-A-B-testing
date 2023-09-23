# Import the necessary libraries
library(cmdstanr)
library(tidyverse)
library(posterior)
library(ggdist)
library(scales)
library(glue)

# Disable scientific notation
options(scipen = 999)
source("scripts/R/plot_helpers.R")

# Read in the data
df <- arrow::read_parquet("data/ab_test_data.parquet")

# Calculate sufficient statistics
df_agg <- df %>%
  group_by(arm_id, treat) %>%
  summarise(
    conversions = sum(conversions),
    trials = sum(trials)
  ) %>%
  mutate(utility = if_else(arm_id == 1, 49.99, 49.99*(1-0.7))) %>%
  ungroup()

 # Change this to your own directory
set_cmdstan_path("C:/Users/adamn/Documents/.cmdstan/cmdstan-2.33.0")

# Compile the model
model <- cmdstan_model("models/binomial_conversions_revenue.stan")

# Data for the model
stan_data <- list(
   N = nrow(df_agg),
   Y = df_agg$conversions,
   K = df_agg$trials,
   alpha_prior = c(1, 1),
   beta_prior = c(1, 1),
   tau = df_agg$utility,
   ref = 1
)

# Fit the model
fit <- model$sample(
    data = stan_data, 
    chains = 4, 
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000
)

# Extract the posterior draws
draws <- fit$draws(format = "draws_df")

# Summarize the posterior draws
summ_draws <- summarize_draws(draws)

## Visualization----------

## Set the base theme settings for ggplot2
theme_set(plot_theme(
    xaxis_size = 24,
    yaxis_size = 24,
    title_size = 30,
    caption_size = 18,
    subtitle_size = 24,
    axis_text_size = 24,
    strip_face = "bold",
    y_axis_face = "bold",
    x_axis_face = "bold",
    plot.margin = margin(5, 5, 5, 5, "mm"),
    plot.caption.position = "plot",
    plot.title.position = "plot",
    strip_size = 24,
    legend_text_size = 24,
    legend.position = "top",
    caption.hjust = 0, 
    caption.vjust = -1,
    transparent = TRUE
  ))

# Plot the posterior draws
ate_draws <- draws %>% 
  select(contains("ATE"))

# Summarize the draws
ate_draws_summ <- ate_draws %>% 
  summarize(
    mean = mean(ATE_revenue),
    median = median(ATE_revenue),
    lower = quantile(ATE_revenue, 0.16),
    upper = quantile(ATE_revenue, 0.84),
    xmin = quantile(ATE_revenue, 0.001),
    xmax = quantile(ATE_revenue, 0.999)
  )

# Labels and annotations for the graphs
upper_cri <- comma(ate_draws_summ$upper)
lower_cri <- comma(ate_draws_summ$lower)
cred_width <- c(68, 89)
caption_text <- glue("Notes: Estimates represent the posterior predictive distribution 
of the average treatment effect on expected revenue from the promotional campaign. Inner 
and out point intervals represent {cred_width[1]}% and {cred_width[2]}% Bayesian posterior 
predictive intervals.")
subtitle_text <- glue(
  "\u2022 The promotional campaign cost between {lower_cri} and {upper_cri} in expected revenue.\n",
  "\u2022 This adverse impact is due to the positive effect on conversions being too small to make up for the 70% discount."
)

# Generate the plot
ate_plot <- ggplot(ate_draws, aes(x = ATE_revenue/1000)) +
  # Add a halfeye geom
  stat_halfeye(
    aes(
      slab_alpha = after_stat(pdf), 
      slab_fill = after_stat(x > 0), 
      point_fill = after_stat(x > 0),
      shape = after_stat(x > 0)
    ),
      slab_size = 2,
      fill_type = "gradient",
      point_interval = "median_qi",
      .width = c(0.68, 0.89),
      point_size = 6,
      stroke = 1
    ) +
    # Adjust the y axis scale
    scale_x_continuous(
      breaks = scales::pretty_breaks(n = 8),
      labels = dollar_format(prefix = "$", big.mark = ",", suffix = "K"),
    ) +
    # Adjust fill scale
    scale_fill_manual(
      "Direction of Effect",
      values = "#14A4D0", 
      aesthetics = "slab_fill",
      guide = guide_legend(
        override.aes = list(
          size = 4,
          alpha = 1,
          fill = "#14A4D0",
          shape = 22
        )
      ),
      labels = "Negative"
    ) +
    # Adjust fill scale
    scale_fill_manual(
      values = "#14D075", 
      aesthetics = "point_fill",
      guide = 'none'
    ) +
    # Adjust alpha scale
    scale_slab_alpha_continuous(guide = 'none') +
    # Adjust shape scale
    scale_shape_manual(guide = 'none', values = 22) +
    # Add the text
    labs(
      title = "Promotional Campaign Does Not Increase Conversions Enough to Offset 70% Discount",
      x = "Average Treatment Effect on Expected Revenue",
      y = "Posterior Density",
      subtitle = subtitle_text,
      caption = str_wrap(caption_text, 180)
    )

# Save the ggplot object
ggsave(
  filename = "Revenue_Example_ATEs.png",
  plot = ate_plot,
  device = "png",
  width = 18,
  height = 10,
  units = "in",
  dpi = "retina",
  type = "cairo",
  bg = "white"
)
