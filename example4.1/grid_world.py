import numpy as np

class Environment(object):
    def __init__(self, n_grid, reward, discount):
        self.n_grid = n_grid
        self.reward = reward
        self.discount = discount

    def transition(self, s, a):
        s_x = s[0]
        s_y = s[1]
        if a==0:
            sp_x = max(0, s_x-1)
            sp_y = s_y
        elif a==1:
            sp_x = min(self.n_grid-1, s_x+1)
            sp_y = s_y
        elif a==2:
            sp_x = s_x
            sp_y = max(0, s_y-1)
        else:
            sp_x = s_x
            sp_y = min(self.n_grid-1, s_y+1)

        sp = (sp_x, sp_y)

        return (sp, self.reward)

def evaluate_q_at_one_state(env, s, a, value):
    sp, r = env.transition(s, a)
    q = r+env.discount*value[sp]
    return q

def evaluate_value_given_policy(env, policy, value, theta):
    n_iteration = 0
    while True:
        value_new = np.zeros((4, 4))
        for s in [(s_x, s_y) for s_x in range(env.n_grid) for s_y in range(env.n_grid)]:
            if (s!=(0, 0)) and (s!=(3, 3)):
                v_new = 0
                #random policy
                actions = policy[s]
                for a, pi in enumerate(actions):
                    q = evaluate_q_at_one_state(env, s, a, value)
                    value_new[s] += pi*q
        max_diff = np.max(np.abs(value-value_new))
        value = value_new
        n_iteration += 1
        if max_diff<theta:
            break
    return n_iteration, value

if __name__=='__main__':
    env = Environment(4, -1, 1)

    value = np.zeros((4, 4))
    policy = np.ones((4, 4, 4))/4

    n_iteration, value = evaluate_value_given_policy(env, policy, value, 1e-8)

    print(value)
