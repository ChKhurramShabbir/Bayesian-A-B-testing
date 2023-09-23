// Average Treatment Effect/Absolute Lift
vector average_treatment_effect_rng(vector theta,
                                    vector tau,  
                                    int trials,
                                    int ref) {
    int N = size(theta);
    vector[N] ATE;
    for (n in 1:N) {
        // Draw from the Predictive Distribution
        real Y0 = binomial_rng(trials, theta[ref]);
        Y0 = Y0 * tau[ref];
        real Y1 = binomial_rng(trials, theta[n]);
        Y1 = Y1 * tau[n];
        ATE[n] = Y1 - Y0;
    }
    return ATE;
}

// Relative Lift/Excess Risk Ratio
vector relative_lift_rng(vector theta,
                         vector tau,  
                         int trials,
                         int ref) {
    int N = size(theta);
    vector[N] relative_lift;
    for (n in 1:N) {
        // Draw from the Predictive Distribution
        real Y0 = binomial_rng(trials, theta[ref]);
        Y0 = Y0 * tau[ref];
        real Y1 = binomial_rng(trials, theta[n]);
        Y1 = Y1 * tau[n];
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
vector expected_posterior_loss(vector theta, 
                               vector tau, 
                               int trials,
                               real epsilon) {
    int N = size(theta);
    vector[N] loss = zeros_vector(N);
    real toc = 0 - epsilon;
    for (n in 1:N) {      // Loop over each action d in D
      real choice_d = binomial_rng(trials, theta[n]);
      choice_d = choice_d * tau[n];
      for (j in 1:N) {  // Loop over each action d' in D
        real choice_d_prime = binomial_rng(trials, theta[n]);
        choice_d_prime = choice_d_prime * tau[n];
        loss[n] += fmin(choice_d - choice_d_prime, epsilon);
      }
      loss[n] /= N - 1; // Normalize by the number of actions in D
    }
    return loss;
}