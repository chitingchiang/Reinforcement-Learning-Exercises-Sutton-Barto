import numpy as np

class Environment(object):
    def __init__(self, goal, p_h, discount=1):
        self.goal = goal
        self.p_h = p_h
        self.discount = discount

    def transition(self, s, a):
        return [(s+a, self.p_h), (s-a, 1-self.p_h)]

def evaluate_q_at_one_state(env, s, a, value):
    q = 0
    for (sp, p) in env.transition(s, a):
        q += p*env.discount*value[sp]
    return q

def evaluate_value_given_policy(env, policy, value, theta):
    n_iteration = 0
    max_diff = 0
    while True:
        max_diff = 0
        for s in range(1, env.goal):
            #=========================================================
            #This only works for deterministic policy! Otherwise the
            #evaluation of value function requires summing over all
            #possible actions!
            a = policy[s]
            v_new = evaluate_q_at_one_state(env, s, a, value)
            max_diff = max(max_diff, abs(v_new-value[s]))
            value[s] = v_new
        n_iteration += 1
        print("at", n_iteration, "iteration, max_diff=", max_diff)
        if max_diff<theta:
            break
    return n_iteration

def greedify_policy_given_value(env, policy, value):

    policy_stable = True
    for s in range(1, env.goal):
        q_best = -1e8
        a_best = 0
        for a in range(1, min(s, env.goal-s)+1):
            q = evaluate_q_at_one_state(env, s, a, value)
            if (q>q_best):
                q_best = q
                a_best = a
        policy_stable = policy_stable&(a_best==policy[s])
        policy[s] = a_best
    return policy_stable

def policy_iteration(env, policy, value, theta=1e-4):
    n_policy_iteration = 0
    policy_stable = False

    while policy_stable==False:
        n_iteration_policy_evaluation = evaluate_value_given_policy(env, policy, value, theta)
        policy_stable = greedify_policy_given_value(env, policy, value)
        n_policy_iteration += 1

    return n_policy_iteration

def value_iteration(env, policy, value, theta=1e-4):
    n_value_iteration = 0
    max_diff = 0.

    while True:
        max_diff = 0
        for s in range(1, env.goal):
            a_best = 0
            q_best = -1e8
            for a in range(1, min(s, env.goal-s)+1):
                q = evaluate_q_at_one_state(env, s, a, value)
                if (q>q_best):
                    a_best = a
                    q_best = q
            max_diff = max(max_diff, abs(q_best-value[s]))
            policy[s] = a_best
            value[s] = q_best
        n_value_iteration += 1
        print("at", n_value_iteration, "iteration, max_diff=", max_diff)
        if max_diff<theta:
            break

    return n_value_iteration

def greedify_policy_given_value2(env, value):

    policy = []
    for s in range(1, env.goal):
        max_bet = min(s, env.goal-s)
        actions = np.arange(1, max_bet+1)
        q = np.zeros(len(actions), dtype=np.float64)
        for i, a in enumerate(actions):
            q[i] = evaluate_q_at_one_state(env, s, a, value)
        q_best = np.max(q)
        policy.append(actions[q==q_best])

    return policy


if __name__ == "__main__":

    env = Environment(goal=100, p_h=0.4)

    policy1 = np.ones(env.goal+1, dtype=np.int32)
    policy2 = np.ones(env.goal+1, dtype=np.int32)
    value1 = np.zeros(env.goal+1, dtype=np.float64)
    value2 = np.zeros(env.goal+1, dtype=np.float64)

    policy1[0] = 0
    policy1[-1] = 0
    policy2[0] = 0
    policy2[-1] = 0

    value1[-1] = 1
    value2[-1] = 1

    n_policy_iteration = policy_iteration(env, policy1, value1, 1e-14)
    n_value_iteration = value_iteration(env, policy2, value2, 1e-14)

    policy1 = greedify_policy_given_value2(env, value1)
    policy2 = greedify_policy_given_value2(env, value2)

    for i in range(env.goal-1):
        print(i+1, policy1[i], policy2[i])
