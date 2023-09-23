# Import the necessary libraries
from polars import read_parquet, DataFrame, col, Int64, when
from cmdstanpy import CmdStanModel, install_cmdstan
from arviz import summary, from_cmdstanpy

# Read in the data
df = read_parquet("data/ab_test_data.parquet")

# Aggregate the data to the arm level
data_agg = (
    df
    .groupby(['treat', 'arm_id'])
    .agg([
        col('conversions').sum().alias('conversions'),
        col('trials').sum().alias('trials').cast(Int64)
    ])
).sort(by=['arm_id'])

# Add data for fixed payoffs
data_utilities = (
    data_agg
    .with_columns(
        utility = when(col('treat') == 0)
            .then(49.99)
            .otherwise(49.99*(1-0.70))
    )
)

# Prepare the data to pass to Stan
stan_data = {
    "N": data_agg.shape[0],
    "Y": data_agg["conversions"].to_numpy(),
    "K": data_agg["trials"].to_numpy(),
    "alpha_prior": [1, 1],
    "beta_prior": [1, 1],
    "tau": data_utilities['utility'].to_numpy()
}

# Install CmdStan if necessary
install_cmdstan(compiler=True, cores=6, progress=True) 

# Compile the Stan model
model = CmdStanModel(stan_file="models/binomial_conversions_revenue.stan")

# Fit the model
fit = model.sample(
    data = stan_data,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 2000,
    iter_sampling = 2000,
    seed = 12345,
    show_progress = True
)

# Check Model Diagnostics and Summary
idata = from_cmdstanpy(fit)
summary(idata)
