#Bungein J Cheng
import random
from collections import defaultdict

actions = ["P", "R", "S"]
states = [
    "RU_8p", "TU_10p", "RU_10p", "RD_10p",
    "RU_8a", "RD_8a", "TU_10a", "RU_10a",
    "RD_10a", "TD_10a", "Terminal"
]

rewards = {
    ("RU_8p", "P"): [(1.0, 2, "TU_10p")],
    ("RU_8p", "R"): [(1.0, 0, "RU_10p")],
    ("RU_8p", "S"): [(1.0, -1, "RD_10p")],
    ("TU_10p", "P"): [(1.0, 2, "RU_10a")],
    ("TU_10p", "R"): [(1.0, 0, "RU_8a")],
    ("RU_10p", "P"): [(0.5, 2, "RU_8a"), (0.5, 2, "RU_10a")],
    ("RU_10p", "R"): [(1.0, 0, "RU_8a")],
    ("RU_10p", "S"): [(1.0, -1, "RD_8a")],
    ("RD_10p", "P"): [(0.5, 2, "RD_8a"), (0.5, 2, "RD_10a")],
    ("RD_10p", "R"): [(1.0, 0, "RD_8a")],
    ("RU_8a", "P"): [(1.0, 2, "TU_10a")],
    ("RU_8a", "R"): [(1.0, 0, "RU_10a")],
    ("RU_8a", "S"): [(1.0, -1, "RD_10a")],
    ("RD_8a", "P"): [(1.0, 2, "TD_10a")],
    ("RD_8a", "R"): [(1.0, 0, "RD_10a")],
    ("TU_10a", "any"): [(1.0, -1, "Terminal")],
    ("RU_10a", "any"): [(1.0, 0, "Terminal")],
    ("RD_10a", "any"): [(1.0, 4, "Terminal")],
    ("TD_10a", "any"): [(1.0, 3, "Terminal")],
}

# Q-learning parameters
alpha = 0.2 #learning rate
gamma = 0.99 #discount rate
epsilon = 1.0 #initial exploration rate 
epsilon_decay = 0.995
alpha_decay = 0.995
min_alpha = 0.01
min_epsilon = 0.01
threshold = 0.001 #convergence threshold for q value changes

Q = defaultdict(lambda: defaultdict(float))# Initialize Q-values
def max_q_value(state):#helper function to get the maximum Q-value for a state
    return max(Q[state].values(), default=0)

episodes = 0#run Q-learning
while True:
    state = "RU_8p" # start state
    max_delta = 0#track max Q-value change in this episode
    episode = []#store episode steps for debugging

    while state != "Terminal":#generate an episode
        possible_actions = [a for a in actions if (state, a) in rewards]#get valid actions
        if not possible_actions:#no valid actions
            break
        action = random.choice(possible_actions)#choose a random action
        transitions = rewards[(state, action)]#get possible transitions
        
        probs, rewards_list, next_states = zip(*transitions)#sample next state based on probabilities
        sampled_index = random.choices(range(len(probs)), weights=probs, k=1)[0]
        reward, next_state = rewards_list[sampled_index], next_states[sampled_index]
       
        old_q = Q[state][action] # Q-value update
        next_max_q = max_q_value(next_state)
        Q[state][action] = old_q + alpha * (reward + gamma * next_max_q - old_q)
 
        print(f"State: {state}, Action: {action}, Reward: {reward}, Old Q: {old_q:.3f}, New Q: {Q[state][action]:.3f}, Max Next Q: {next_max_q:.3f}")
        
        max_delta = max(max_delta, abs(Q[state][action] - old_q))# Track maximum Q-value change
        state = next_state# Update state
    
    alpha = max(min_alpha, alpha * alpha_decay)# decay learning rate
    if max_delta < threshold:# check for convergence
        break
   
    episodes += 1 #increment episode count

optimal_policy = {}#extract the optimal policy
for state in states:
    if state != "Terminal":#exclude terminal state
        optimal_policy[state] = max(Q[state], key=Q[state].get, default=random.choice(actions))

print(f"\nConverged after {episodes} episodes.")
print("\nFinal Q-values:")
for state, actions in Q.items():#print final Q-values
    for action, q_value in actions.items():#for each action
        print(f"State: {state}, Action: {action}, Q-value: {q_value:.3f}")
print("\nOptimal Policy:")
for state, action in optimal_policy.items():#print optimal policy
    print(f"State: {state}, Best Action: {action}")