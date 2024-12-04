#Bungein J Cheng
import random
from collections import defaultdict

states = ["RU_8p", "TU_10p", "RU_10p", "RD_10p", "RU_8a", "RD_8a", "TU_10a", "RU_10a", "RD_10a", "TD_10a", "Terminal"]
actions = ["P", "R", "S"]

rewards = {
    #first level of the tree
    ("RU_8p", "P"): [(1.0, 2, "TU_10p")],
    ("RU_8p", "R"): [(1.0, 0, "RU_10p")],
    ("RU_8p", "S"): [(1.0, -1, "RD_10p")],
    
    # seconde level
    ("TU_10p", "P"): [(1.0, 2, "RU_10a")],
    ("TU_10p", "R"): [(1.0, 0, "RU_8a")],

    ("RU_10p", "P"): [(0.5, 2, "RU_8a"), (0.5, 2, "RU_10a")],
    ("RU_10p", "R"): [(1.0, 0, "RU_8a")],
    ("RU_10p", "S"): [(1.0, -1, "RD_8a")],  
    
    ("RD_10p", "R"): [(1.0, 0, "RD_8a")],
    ("RD_10p", "P"): [(0.5, 2, "RD_8a"), (0.5, 2, "RD_10a")],
    
    #thierd level
    ("RU_8a", "P"): [(1.0, 2, "TU_10a")],
    ("RU_8a", "R"): [(1.0, 0, "RU_10a")],
    ("RU_8a", "S"): [(1.0, -1, "RD_10a")],
    
    ("RD_8a", "R"): [(1.0, 0, "RD_10a")],
    ("RD_8a", "P"): [(1.0, 2, "TD_10a")],
    
    #last level
    ("TU_10a", "any"): [(1.0, -1, "Terminal")],
    
    ("RU_10a", "any"): [(1.0, 0, "Terminal")],
  
    ("RD_10a", "any"): [(1.0, 4, "Terminal")],
    
    ("TD_10a", "any"): [(1.0, 3, "Terminal")],
 
}
def get_possible_actions(state):# list of possible actions for a state
    if state == "Terminal":
        return []
    if "10a" in state:
        return ["any"]
    return [a for a in actions if (state, a) in rewards]

def value_iteration():#value iteration
    gamma = 0.99# discount factor
    theta = 0.001# convergence threshold
    V = {state: 0 for state in states}# Initialize values to 0
    policy = {}
    iterations = 0
    
    while True:
        delta = 0
        iterations += 1
        for state in states:#update value for each state
            if state == "Terminal":
                continue
                
            old_v = V[state]
            possible_actions = get_possible_actions(state)# get possible actions
            
            action_values = {}
            for action in possible_actions:
                
                value = 0#handle each action's value considering all possible outcomes
                if action == "any":#terminal states have only one outcome
                    prob, reward, next_state = rewards[(state, "any")][0]
                    value = prob * (reward + gamma * V[next_state])
                else:#sum up value for all possible outcomes of this action
                    for prob, reward, next_state in rewards[(state, action)]:
                        value += prob * (reward + gamma * V[next_state])
                action_values[action] = value
            
            if action_values:#update value and policy
                best_action = max(action_values.items(), key=lambda x: x[1])
                V[state] = best_action[1]
                policy[state] = best_action[0]
                
                print(f"\nState: {state}")
                print(f"Previous value: {old_v:.3f}")
                print(f"New value: {V[state]:.3f}")
                print("Action values:", {a: f"{v:.3f}" for a, v in action_values.items()})
                print(f"Selected action: {best_action[0]}")
            
            delta = max(delta, abs(old_v - V[state])) # track max delta
        
        if delta < theta:#check for convergence
            break
    
    print("\nFinal Results:")
    print(f"Iterations: {iterations}")
    print("\nFinal Values:")
    for state in states:
        print(f"{state}: {V[state]:.3f}")
    print("\nOptimal Policy:")
    for state in states:
        if state != "Terminal":
            print(f"{state}: {policy[state]}")

if __name__ == "__main__":
    value_iteration()