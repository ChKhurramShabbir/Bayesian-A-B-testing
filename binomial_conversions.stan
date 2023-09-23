/* 
    Model: Binomial Conversion Model for Bayesian A/B Testing 
    Author: A. Jordan Nafa
    Date: 2023-08-07
    License: MIT
*/

functions {
   #include abtest_conversion_functions.stan
}

data {
    int<lower=1> N;                     // Number of Variants
    array[N] int<lower=1> K;            // Number of Trials
    array[N] int<lower=0, upper=K> Y;   // Number of Successes
    int<lower=1> ref;                            // Reference Variant

    // Priors for Each Variant
    vector<lower=0>[N] alpha_prior;     // Prior Successes
    vector<lower=0>[N] beta_prior;      // Prior Failures
}

transformed data {
    vector[N] alpha;                    // pi(alpha) + Observed Successes
    vector[N] beta;                     // pi(beta) + Observed Failures
    for (n in 1:N) {
        alpha[n] = alpha_prior[n] + Y[n];
        beta[n] = beta_prior[n] + (K[n] - Y[n]);
    }
}

parameters {
    vector<lower=0, upper=1>[N] theta;    // Conversion Rate
}

model {
    // Vectorized Priors and Likelihood
    target += beta_lpdf(theta | alpha, beta);
    target += binomial_lpmf(Y | K, theta);
}
