#include <iostream>
#include <cmath>

using namespace std;

class Environment{
    public:

    int max_car;
    int max_move;

    double rental_credit;
    double move_cost;
    double discount;

    double lambda_A_request;
    double lambda_A_return;
    double lambda_B_request;
    double lambda_B_return;

    double*** p_A;
    double*** p_B;

    double poisson_pmf(int k, double lambda){
        return pow(lambda, k)*exp(-lambda)/tgamma(k+1);
    }

    void transition_probability(){

        int max_state = max_car+1;

        p_A = new double** [max_state];
        p_B = new double** [max_state];
        for (int i=0; i<max_state; ++i){
            p_A[i] = new double* [max_state];
            p_B[i] = new double* [max_state];
            for (int j=0; j<max_state; ++j){
                p_A[i][j] = new double [max_state];
                p_B[i][j] = new double [max_state];
            }
        }

        double *p_A_request_arr = new double [max_state];
        double *p_A_return_arr = new double [max_state];
        double *p_B_request_arr = new double [max_state];
        double *p_B_return_arr = new double [max_state];
        double p_A_request_sum = 0;
        double p_A_return_sum = 0;
        double p_B_request_sum = 0;
        double p_B_return_sum = 0;
        for (int i=0; i<max_car; ++i){
            p_A_request_arr[i] = poisson_pmf(i, lambda_A_request);
            p_A_return_arr[i] = poisson_pmf(i, lambda_A_return);
            p_B_request_arr[i] = poisson_pmf(i, lambda_B_request);
            p_B_return_arr[i] = poisson_pmf(i, lambda_B_return);
            p_A_request_sum += p_A_request_arr[i];
            p_A_return_sum += p_A_return_arr[i];
            p_B_request_sum += p_B_request_arr[i];
            p_B_return_sum += p_B_return_arr[i];
        }
        p_A_request_arr[max_car] = 1-p_A_request_sum;
        p_A_return_arr[max_car] = 1-p_A_return_sum;
        p_B_request_arr[max_car] = 1-p_B_request_sum;
        p_B_return_arr[max_car] = 1-p_B_return_sum;

        for (int s_morning=0; s_morning<=max_car; ++s_morning){
            for (int s_request=0; s_request<=s_morning; ++s_request){
                int s_after_request = s_morning-s_request;
                double p_A_request = p_A_request_arr[s_request];
                double p_B_request = p_B_request_arr[s_request];
                if (s_request==s_morning){
                    for (int i=s_request+1; i<=max_car; ++i){
                        p_A_request += p_A_request_arr[i];
                        p_B_request += p_B_request_arr[i];
                    }
                }
                for (int s_return=0; s_return<=max_car-s_after_request; ++s_return){
                    double p_A_return = p_A_return_arr[s_return];
                    double p_B_return = p_B_return_arr[s_return];
                    if (s_return==max_car-s_after_request){
                        for (int i=max_car-s_after_request+1; i<=max_car; ++i){
                            p_A_return += p_A_return_arr[i];
                            p_B_return += p_B_return_arr[i];
                        }
                    }
                    p_A[s_morning][s_request][s_return] = p_A_request*p_A_return;
                    p_B[s_morning][s_request][s_return] = p_B_request*p_B_return;
                }
            }
        }
    }
};

double evaluate_q_at_one_state(Environment env, int s_A, int s_B, int action, double** value);
int evaluate_value_given_policy(Environment env, int **policy, double **value, double theta);
bool greedify_policy_given_value(Environment env, int **policy, double **value);
int policy_iteration(Environment env, int** policy, double** value, double theta=1e-4);
int value_iteration(Environment env, int** policy, double** value, double theta=1e-4);

int main(){

    Environment env;

    env.max_car = 20;
    env.max_move = 5;
    env.rental_credit = 10;
    env.move_cost = 2;
    env.discount = 0.9;
    env.lambda_A_request = 3;
    env.lambda_A_return = 3;
    env.lambda_B_request = 4;
    env.lambda_B_return = 2;

    env.transition_probability();

    int **policy1 = new int* [env.max_car+1];
    int **policy2 = new int* [env.max_car+1];
    double **value1 = new double *[env.max_car+1];
    double **value2 = new double *[env.max_car+1];
    for (int i=0; i<=env.max_car; ++i){
        policy1[i] = new int [env.max_car+1];
        policy2[i] = new int [env.max_car+1];
        value1[i] = new double [env.max_car+1];
        value2[i] = new double [env.max_car+1];
    }

    for (int i=0; i<=env.max_car; ++i){
        for (int j=0; j<=env.max_car; ++j){
            policy1[i][j] = 0;
            policy2[i][j] = 0;
            value1[i][j] = 0;
            value2[i][j] = 0;
        }
    }

    int n_policy_iteration = policy_iteration(env, policy1, value1, 1e-4);
    int n_value_iteration = value_iteration(env, policy2, value2, 1e-4);

    for (int s_A=0; s_A<=env.max_car; ++s_A){
        for (int s_B=0; s_B<=env.max_car; ++s_B){
            if (policy1[s_A][s_B]!=policy2[s_A][s_B]){
                cout << "At s_A=" << s_A << ", s_B=" << s_B << ", the two policies are different!" << endl;
            }
            if (abs(value1[s_A][s_B]-value2[s_A][s_B])/value1[s_A][s_B]>1e-3){
                cout << "At s_A=" << s_A << ", s_B=" << s_B << ", the two values are different by more than 1e-3!" << endl;
            }
        }
    }
}

double evaluate_q_at_one_state(Environment env, int s_A, int s_B, int action, double** value){
    double q = 0;
    int s_A_morning = s_A-action;
    int s_B_morning = s_B+action;
    double move_cost_reward = env.move_cost*abs(action);
    for (int s_A_request=0; s_A_request<=s_A_morning; ++s_A_request){
        int s_A_after_request = s_A_morning-s_A_request;
        for (int s_A_return=0; s_A_return<=env.max_car-s_A_after_request; ++s_A_return){
            int s_A_night = s_A_after_request+s_A_return;
            for (int s_B_request=0; s_B_request<=s_B_morning; ++s_B_request){
                int s_B_after_request = s_B_morning-s_B_request;
                for (int s_B_return=0; s_B_return<=env.max_car-s_B_after_request; ++s_B_return){
                    int s_B_night = s_B_after_request+s_B_return;
                    double reward = env.rental_credit*(s_A_request+s_B_request)-move_cost_reward;
                    q += env.p_A[s_A_morning][s_A_request][s_A_return]
                        *env.p_B[s_B_morning][s_B_request][s_B_return]
                        *(reward+env.discount*value[s_A_night][s_B_night]);
                }
            }
        }
    }
    return q;
}

int evaluate_value_given_policy(Environment env, int **policy, double **value, double theta){
    int n_iteration = 0;
    double max_diff;
    do{
        max_diff = 0;
        for (int s_A=0; s_A<=env.max_car; ++s_A){
            for (int s_B=0; s_B<=env.max_car; ++s_B){
                //=========================================================
                //This only works for deterministic policy! Otherwise the
                //evaluation of value function requires summing over all
                //possible actions!
                int action = policy[s_A][s_B];
                double v_new = evaluate_q_at_one_state(env, s_A, s_B, action, value);
                max_diff = max(max_diff, abs(v_new-value[s_A][s_B]));
                value[s_A][s_B] = v_new;
            }
        }
        n_iteration++;
        cout << "at " << n_iteration << " iteration, max_diff = " << max_diff << endl;
    } while (max_diff>theta);

    return n_iteration;
}

bool greedify_policy_given_value(Environment env, int **policy, double **value){
    bool policy_stable = true;
    for (int s_A=0; s_A<=env.max_car; ++s_A){
        for (int s_B=0; s_B<=env.max_car; ++s_B){
            int move_min = max(-env.max_move, max(-s_B, s_A-env.max_car));
            int move_max = min(env.max_move, min(s_A, env.max_car-s_B));
            double q_best = -1e8;
            int action_best = 0;
            for (int action=move_min; action<=move_max; ++action){
                double q = evaluate_q_at_one_state(env, s_A, s_B, action, value);
                if (q>=q_best){
                    q_best = q;
                    action_best = action;
                }
            }
            policy_stable = policy_stable&&(action_best==policy[s_A][s_B]);
            policy[s_A][s_B] = action_best;
        }
    }
    return policy_stable;
}


int policy_iteration(Environment env, int** policy, double** value, double theta=1e-4){
    int n_policy_iteration = 0;
    bool policy_stable = false;

    while (policy_stable==false){
        int n_iteration_policy_evaluation = evaluate_value_given_policy(env, policy, value, theta);
        policy_stable = greedify_policy_given_value(env, policy, value);
        n_policy_iteration++;
    }

    return n_policy_iteration;
}

int value_iteration(Environment env, int** policy, double** value, double theta=1e-4){
    int n_value_iteration = 0;
    double max_diff;
    do{
        max_diff = 0;
        for (int s_A=0; s_A<=env.max_car; ++s_A){
            for (int s_B=0; s_B<=env.max_car; ++s_B){
                int move_min = max(-env.max_move, max(-s_B, s_A-env.max_car));
                int move_max = min(env.max_move, min(s_A, env.max_car-s_B));
                double q_best = -1e8;
                for (int action=move_min; action<=move_max; ++action){
                    double q = evaluate_q_at_one_state(env, s_A, s_B, action, value);
                    if (q>=q_best) q_best = q;
                }
                max_diff = max(max_diff, abs(q_best-value[s_A][s_B]));
                value[s_A][s_B] = q_best;
            }
        }
        n_value_iteration++;
        cout << "at " << n_value_iteration << " iteration, max_diff = " << max_diff << endl;
    } while (max_diff>theta);

    greedify_policy_given_value(env, policy, value);

    return n_value_iteration;
}

