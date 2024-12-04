#Bungein J Cheng
import random
from collections import defaultdict

states = ["RU_8p", "TU_10p", "RU_10p", "RD_10p", "RU_8a", "RD_8a", "TU_8a", "TD_8a", "TU_10a", "RU_10a", "RD_10a", "TD_10a", "Terminal"]
actions = ["P", "R", "S"]

rewards = {# addrewards with probabilities
    ("RU_8p", "P"): [(1.0, 2, "TU_10p")],
    ("RU_8p", "R"): [(1.0, 0, "RU_10p")],
    ("RU_8p", "S"): [(1.0, -1, "RD_10p")],
    
    #second level of the tree
    ("TU_10p", "P"): [(1.0, 2, "RU_10a")],
    ("TU_10p", "R"): [(1.0, 0, "RU_8a")],
    ("RU_10p", "P"): [(0.5, 2, "RU_8a"), (0.5, 2, "RU_10a")], 
    ("RU_10p", "R"): [(1.0, 0, "RU_8a")],
    ("RU_10p", "S"): [(1.0, -1, "RD_8a")],  
    
    ("RD_10p", "R"): [(1.0, 0, "RD_8a")],
    ("RD_10p", "P"): [(0.5, 2, "RD_8a"), (0.5, 2, "RD_10a")], 
    
    ("RU_8a", "P"): [(1.0, 2, "TU_10a")],
    ("RU_8a", "R"): [(1.0, 0, "RU_10a")],
    ("RU_8a", "S"): [(1.0, -1, "RD_10a")],
    
    ("RD_8a", "R"): [(1.0, 0, "RD_10a")],
    ("RD_8a", "P"): [(1.0, 2, "TD_10a")],
    
     #third level of the tree
    ("TU_10a", "any"): [(1.0, -1, "Terminal")],
    ("RU_10a", "any"): [(1.0, 0, "Terminal")],
    ("RD_10a", "any"): [(1.0, 4, "Terminal")],
    ("TD_10a", "any"): [(1.0, 3, "Terminal")]
}

def get_possible_actions(state):#list of possible actions for a state
    if state == "Terminal":
        return []
    if "10a" in state:
        return ["any"]
    return [a for a in actions if (state, a) in rewards]

def choose_outcome(outcomes):#choose outcome based on probabilities
    rand = random.random()
    cumulative_prob = 0
    for prob, reward, next_state in outcomes:# probabilities, rewards, and next states
        cumulative_prob += prob
        if rand < cumulative_prob:
            return reward, next_state
    return outcomes[-1][1:]# return the last reward and next state
def format_experience(experience):#format experience
    state, action, reward = experience
    return f"({state}, {action}, r={reward})"

def run_monte_carlo(num_episodes=50, alpha=0.1):#monte carlo simulation
    state_values = {state: 0.0 for state in states}
    total_rewards = []
    
    print("\nRunning Monte Carlo simulation with probabilistic transitions:")
    print("-" * 50)
    
    for episode in range(num_episodes):# run episodes 
        state = "RU_8p"
        episode_experience = []
        episode_reward = 0
        
        while state != "Terminal":#generate an episode
            possible_actions = get_possible_actions(state)
            if not possible_actions:
                break
           
            action = random.choice(possible_actions)# get random action
            reward, next_state = choose_outcome(rewards[(state, action)])#get reward and next state based on probabilities   
            episode_experience.append((state, action, reward))# add experience to episode
            episode_reward += reward
            state = next_state
        
        print(f"\nEpisode {episode + 1}:")
        print("Sequence:", " -> ".join(format_experience(exp) for exp in episode_experience))
        print(f"Total Reward: {episode_reward}")
        
        total_rewards.append(episode_reward)
        
        visited_states = set()#perform first-visit Monte Carlo updates
        for idx, (s, _, _) in enumerate(episode_experience):
            if s not in visited_states:
                visited_states.add(s)
                G = sum(x[2] for x in episode_experience[idx:])#calculate return
                state_values[s] += alpha * (G - state_values[s])#update state value
    
    print("\n" + "=" * 50)
    print("Final Results after", num_episodes, "episodes:")
    print("\nState Values:")
    for state in states:#print state values
        if state != "Terminal":
            print(f"{state:8}: {state_values[state]:6.2f}")
    
    avg_reward = sum(total_rewards) / num_episodes
    print(f"\nAverage Reward per Episode: {avg_reward:.2f}")
    
    return state_values, total_rewards

if __name__ == "__main__":
    state_values, rewards = run_monte_carlo()