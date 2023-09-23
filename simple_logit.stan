data {
  // Data Dimensions
  int<lower=0> N;                   // N Observations
  int<lower=1> K;                   // K Columns in Design Matrix

  // Data for the Model
  vector<lower=0, upper=1>[N] Y;    // Response Variable
  matrix[N, K - 1] X;               // Design Matrix
}

parameters {
  real alpha;                       // Global Intercept
  vector[K-1] beta;                 // Population-level Coefficients
}

transformed parameters {
  // Inverse Logit Transformation
  vector[N] theta = inv_logit(alpha + X * beta);
}

model {
  // Priors for the model parameters
  target += normal_lpdf(alpha | 0, 1);
  target += normal_lpdf(beta | 0, 0.75);

  // Likelihood
  target += bernoulli_logit_glm_lpmf(Y | X, alpha, beta);
}