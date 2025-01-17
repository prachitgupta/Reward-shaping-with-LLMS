import os
import numpy as np
import pandas as pd
from tqdm import trange
import random

# # Environment
import gymnasium as gym
import highway_env

# Agent
from stable_baselines3 import DQN

import sys
from tqdm.notebook import trange

import boto3
import os
# Define Boto3 client for Bedrock
client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Claude API settings
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
CLAUDE_API_URL = 'https://api.anthropic.com/v1/complete'

##no of vehicles in lane = same lane id  ka total
##no of vehicles in left lane = lane id -1  ka sum
##relative velocity of closest vehicle = directly from changi congig or - ego(2nd row se)
##closest vehicle ka distance = 2 nd row say nikalo
##left closest vehicle ka distance = 2 nd row say nikalo
##rightt closest vehicle ka distance = 2 nd row say nikalo 
## and unki relative velocities



def claude_action(prompt1, assist1, prompt2, model='claude-v1', max_tokens_to_sample=50, temperature=0.7):
    """
    Sends prompts to Claude.ai and retrieves the recommended action.
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {CLAUDE_API_KEY}',
    }

    full_prompt = f"{prompt1}\n\n{assist1}\n\n{prompt2}"
    
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": full_prompt}],
            }
        ],
    }
    request = json.dumps(native_request)
    try:
        response = client.invoke_model(modelId=model_id, body=request)
        model_response = json.loads(response["body"].read())
        response_json = model_response["content"][0]["text"]
        #action_text = response_json.get('completion', '').strip()
        if 'Final decision:' in response_json:
            action = response_json.split('Final decision:')[-1].strip().upper()
            return action
        else:
            print(f"Unexpected response format: {action_text}")
            return 'IDLE'
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Claude.ai: {e}")
        time.sleep(1)
        return 'IDLE'


class MyHighwayEnv_llm(gym.Env):
    def __init__(self, vehicleCount = 10):
        super(MyHighwayEnv_llm, self).__init__()
        # base setting
        self.vehicleCount = vehicleCount
        self.prev_action  = 'FASTER'

        # environment setting
        self.config = {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": vehicleCount,
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(0, 32, 9),
            },
            "duration": 40,
            "vehicles_density": 2,
            "show_trajectories": True,
            "render_agent": True,
        }
        self.env = gym.make("highway-v0")
        self.env.configure(self.config)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,high=np.inf,shape=(10,5),dtype=np.float32
        )

    def find_smallest_positive(self, arr):
        smallest_positive = float('inf')
        index = -1

        for i, value in enumerate(arr):
            if 0 < value < smallest_positive:
                smallest_positive = value
                index = i

        return smallest_positive, index

    def prompt_design_safe(self, obs_):

        # Part 1: Initial prompt introducing the scenario
        prompt1 = 'You are claude, a large language model. You are now acting as a mature driving assistant, who can give accurate and correct advice for human drivers in complex urban driving scenarios. The information in the current scenario:\n\
                    You, the \'ego\' car, are now driving on a highway. You have already driven for 0 seconds.\n\
                    The decision made by the agent LAST time step was \'FASTER\' (accelerate the vehicle).'

        # Part 2: Driving rules that must be followed
        rules = "There are several rules you need to follow when you drive on a highway:\n\
                1. Try to keep a safe distance from the car in front of you.\n\
                2. DON’T change lanes frequently. If you want to change lanes, double-check the safety of vehicles in the target lane."
        
        prompt1 += rules

        # Part 3: Attention points for decision making
        att_points = "Here are your attention points:\n\
                        1. You must output a decision when you finish this task. Your final output decision must be unique and not ambiguous. For example, you cannot say \"I can either keep lane or accelerate at the current time\".\n\
                        2. You need to always remember your current lane ID, your available actions, and available lanes before you make any decision.\n\
                        3. Once you have a decision, you should check the safety with all the vehicles affected by your decision.\n\
                        4. If you verify a decision is unsafe, you should start a new one and verify its safety again from scratch."
        
        prompt1 += att_points

        # Part 4: Request for additional scenario details
        assist1 = 'Understood. Please provide the current scenario or conditions, such as traffic density, speed of surrounding vehicles, your current speed, and any other relevant information, so I can recommend the best action.'

        # Part 5: Describing the current highway scenario
        prompt2 = 'Here is the current scenario:\n\
        There are four lanes on the highway: Lane-1 (leftmost), Lane-2, Lane-3, Lane-4 (rightmost).\n\n'

        # Extract information from the observations
        x, y, vx, vy = obs_[:, 1], obs_[:, 2], obs_[:, 3], obs_[:, 4]

        # Ego vehicle details
        ego_x, ego_y = x[0], y[0]
        ego_vx, ego_vy = vx[0], vy[0]

        # Other vehicles' relative positions
        veh_x, veh_y = x[1:] - ego_x, y[1:] - ego_y
        veh_vx, veh_vy = vx[1:], vy[1:]

        # Determine lane information for ego and other vehicles
        lanes = y // 4 + 1
        ego_lane = lanes[0]
        veh_lanes = lanes[1:]

        # Left and right lane availability based on the ego vehicle's current lane
        if ego_lane == 1:
            ego_left_lane = 'Left lane: Not available\n'
            ego_right_lane = 'Right lane: Lane-' + str(ego_lane + 1) + '\n'
        elif ego_lane == 4:
            ego_left_lane = 'Left lane: Lane-' + str(ego_lane - 1) + '\n'
            ego_right_lane = 'Right lane: Not available\n'
        else:
            ego_left_lane = 'Left lane: Lane-' + str(ego_lane - 1) + '\n'
            ego_right_lane = 'Right lane: Lane-' + str(ego_lane + 1) + '\n'

        # Append ego vehicle information to prompt2
        prompt2 += 'Ego vehicle:\n\
        \tCurrent lane: Lane-' + str(ego_lane) + '\n' + '\t' + ego_left_lane + '\t' + ego_right_lane + '\tCurrent speed: ' + str(ego_vx) + ' m/s\n\n'

        # Lane information including vehicles ahead in each lane
        lane_info = 'Lane info:\n'
        for i in range(4):
            inds = np.where(veh_lanes == i + 1)[0]
            num_v = len(inds)
            if num_v > 0:
                # Find the closest vehicle in the current lane
                val, ind = self.find_smallest_positive(veh_x[inds])
                true_ind = inds[ind]
                lane_info += '\tLane-' + str(i + 1) + ': There are ' + str(num_v) + ' vehicle(s) in this lane ahead of ego vehicle, closest being ' + str(veh_x[true_ind]) + ' m ahead traveling at ' + str(veh_vx[true_ind]) + ' m/s.\n'
            else:
                lane_info += '\tLane-' + str(i + 1) + ': No other vehicle ahead of ego vehicle.\n'
        
        # Append lane information to prompt2
        prompt2 += lane_info

        # Part 6: Adding additional attention points and the final decision instruction
        safety_verification = '\nAttention points:\n\
        \t1.Safety is the main priority, You can stay IDLE or even Go slower but in no circumstance you should collide with lead vehicle.\n\
        \t2.You are not supposed to change lane frequently only when its neccessary to keep the vehicle safe. Before changing lane check safety like safe distance and speed fro other vehicles\n\
        \t3. Safety is a priority, but do not forget efficiency.\n\
        \t4. you should only make a decesion once you have verified safety with other vehicles otherwise make a new decesion and verify its safety from scratch\n \
        \t5. Your suggested action has to be one from the five listed actions - IDLE, SLOWER, FASTER, LANE_LEFT, LANE_RIGHT.\n\
        Your last action was ' + self.prev_action + '.Please recommend action for the current scenario only in this format and DONT propound anything else other than \'Final decision: <final decision>\'.\n'

        # Append the attention information to prompt2
        prompt2 += safety_verification

        # Return the three prompts
        return prompt1, assist1, prompt2


    def step(self, action):

        # Step the wrapped environment and capture all returned values
        obs, dqn_reward, done, truncated, info = self.env.step(action)

        self.prev_action = action_dict[action]

        Reward = 1 / (1 + np.exp(-dqn_reward))

        return obs, Reward, done, truncated, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs  # Make sure to return the observation




def generate_dataset_model(env, model, file_name, episodes=50):
    observations = []
    actions = []

    for episode in trange(episodes, desc="Dataset Generation"):
        (obs, info), done, truncated = env.reset(), False, False
        while not (done or truncated):
            # Generate prompt for LLM
            prompt1, assist1, prompt2 = env.prompt_design_safe(obs)
            llm_act = claude_action(prompt1, assist1, prompt2, env.prev_action).strip().split('.')[0]

            # Convert LLM action to a numerical label
            if 'LANE_LEFT' in llm_act:
                action_label = 0
            elif 'IDLE' in llm_act:
                action_label = 1
            elif 'LANE_RIGHT' in llm_act:
                action_label = 2
            elif 'FASTER' in llm_act:
                action_label = 3
            elif 'SLOWER' in llm_act:
                action_label = 4

            # Store observation and corresponding LLM action
            observations.append(obs.flatten())
            actions.append(action_label)

            # Predict action using model and step the environment
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))

    # Convert to numpy arrays
    observations = np.array(observations)
    actions = np.array(actions)

    # Save the dataset as a CSV file
    data = pd.DataFrame(observations)
    data['action'] = actions

    # Create a directory to save the dataset if it doesn't exist
    dataset_dir = 'datasetsDQN'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Save the dataset in the Colab workspace
    dataset_path = os.path.join(dataset_dir, file_name)
    data.to_csv(dataset_path, index=False)

    print(f"Dataset saved to {dataset_path}")

    return observations, actions



if __name__ == "__main__" :
    env = MyHighwayEnv_llm()

    modelDqn = DQN.load("dqn_model")

    # Generate dataset from environment and LLM
    observations, actions = generate_dataset(env, modelDqn, "highway_with_dqn.csv")