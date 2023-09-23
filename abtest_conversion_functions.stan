
// Average Treatment Effect/Absolute Lift
vector average_treatment_effect_rng(vector alpha, vector beta, int ref) {
    int N = size(alpha);
    vector[N] ATE;
    for (n in 1:N) {
      real Y0 = beta_rng(1 + alpha[ref], 1 + beta[ref]);
      real Y1 = beta_rng(1 + alpha[n], 1 + beta[n]);
      ATE[n] = Y1 - Y0;
    }
    return ATE;
}

// Relative Lift/Excess Risk Ratio
vector relative_lift_rng(vector alpha, vector beta, int ref) {
    int N = size(alpha);
    vector[N] relative_lift;
    for (n in 1:N) {
      real Y0 = beta_rng(1 + alpha[ref], 1 + beta[ref]);
      real Y1 = beta_rng(1 + alpha[n], 1 + beta[n]);
      relative_lift[n] = (Y1 - Y0) / Y0;
    }
    return relative_lift * 100;
}

// Probability of Being Best
vector probability_best(vector theta, int N){
    vector[N] prob_best;
    real best_choice = max(theta);    
    for (n in 1:N) {
        prob_best[n] = (theta[n] > best_choice);
    }

    // Uniform in the Case of a Tie
    prob_best = prob_best / sum(prob_best);
    return prob_best;
}

// Expected Posterior Loss
vector expected_loss_rng(vector alpha, vector beta, real epsilon) {
    int N = size(alpha);
    vector[N] loss = zeros_vector(N);
    real toc = 0 - epsilon;
    for (n in 1:N) {      // Loop over each action d in D
      real choice_d = beta_rng(1 + alpha[n], 1 + beta[n]);
      for (j in 1:N) {  // Loop over each action d' in D
        real choice_d_prime = beta_rng(2 + alpha[j], 1 + beta[j]);
        loss[n] += fmin(choice_d - choice_d_prime, epsilon);
      }
      loss[n] /= N - 1; // Normalize by the number of actions
    }
    return loss;
}

