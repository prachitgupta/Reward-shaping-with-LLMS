# # Environment
import gymnasium as gym
import highway_env

##claude
import os
import requests
import boto3
import json
from botocore.exceptions import ClientError

# Agent
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

##utils
import sys
from tqdm.notebook import trange
import pprint
from matplotlib import pyplot as plt
import numpy as np
import joblib
from tqdm import trange
import base64
import random
import pandas as pd
from pathlib import Path
import time

# ##visualization
# from gymnasium.wrappers import RecordVideo
# from IPython import display as ipythondisplay
# from pyvirtualdisplay import Display


##record and show videos
def record_videos(env, video_folder="videos"):
    wrapped = RecordVideo(
        env, video_folder=video_folder, episode_trigger=lambda e: True
    )

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped


def show_videos(path="videos"):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
    
####CLAUDE####
    
# Define Boto3 client for Bedrock
client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Claude API settings
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
CLAUDE_API_URL = 'https://api.anthropic.com/v1/complete'

##claude actions
def claude_action(prompt1, assist1, prompt2, model='claude-v1', max_tokens_to_sample=50, temperature=0.7):
    """
    Sends prompts to Claude.ai and retrieves the recommended action.
    """
    print("claude is acting")
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {CLAUDE_API_KEY}',
    }

    full_prompt = f"{prompt1}\n\n{assist1}\n\n{prompt2}"
    
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.1,
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
    
#######

###GROQ###
from groq import *

client = Groq(api_key = "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")

def groq_action(prompt1, assist1, prompt2, last_act='FASTER'):

    chat_completion = client.chat.completions.create(messages=[{"role": "user", "content": prompt1},
                                                           {"role": "assistant", "content": assist1},
                                                           {"role": "user", "content": prompt2}], model="llama3-groq-70b-8192-tool-use-preview")
    
    try:
        action = chat_completion.choices[0].message.content.strip().split('Final decision: ')[1].strip().split('\'')[0]
    except:
        action = last_act

    return action


########

def map_llm_action_to_label(llm_act):
    """
    Maps the LLM-recommended action string to a numerical label.
    """
    action_map = {
        'LANE_LEFT': 0,
        'IDLE': 1,
        'LANE_RIGHT': 2,
        'FASTER': 3,
        'SLOWER': 4,
    }
    return action_map.get(llm_act.upper(), 1)  # Default to IDLE if unrecognized

action_dict = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

class MyHighwayEnvLLM(gym.Env):
    """
    Custom Gym environment for highway driving with LLM prompts.
    """
    def __init__(self, vehicleCount):
        super(MyHighwayEnvLLM, self).__init__()
        self.vehicleCount = vehicleCount
        self.prev_action = 'FASTER'

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
            "vehicles_density":1.5,
            "show_trajectories": True,
            "render_agent": True,
        }
        self.env = gym.make("highway-v0",render_mode='rgb_array', config= self.config)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(vehicleCount, 5), dtype=np.float32)
    
        
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
        # . SO You may Go faster but only when there is no vehicle in front of you or its SUFFICIENTLY far off
        safety_verification = '\nAttention points:\n\
        \t1.Safety is the main priority, You can stay IDLE or even Go slower but in no circumstance you should collide with lead vehicle.\n\
        \t2.You are not supposed to change lane frequently only when its neccessary to keep the vehicle safe. Before changing lane check safety like safe distance and speed fro other vehicles\n\
        \t3. Safety is a priority, but do not forget efficiency. \n\
        \t4. you should only make a decesion once you have verified safety with other vehicles otherwise make a new decesion and verify its safety from scratch\n \
        \t5. Your suggested action has to be one from the five listed actions - IDLE, SLOWER, FASTER, LANE_LEFT, LANE_RIGHT.\n\
        Your last action was ' + self.prev_action + '.Please recommend action for the current scenario only in this format and DONT propound anything else other than \'Final decision: <final decision>\'.\n'

        # Append the attention information to prompt2
        prompt2 += safety_verification

        # Return the three prompts
        return prompt1, assist1, prompt2
    
    def groq_query(self,obs):
        # Generate prompt for LLM
        prompt1, assist1, prompt2 = self.prompt_design_safe(obs)
        ##ask for claude response
        llm_act = groq_action(prompt1, assist1, prompt2, self.prev_action).strip().split('.')[0]
        ##int action
        action = map_llm_action_to_label(llm_act)
        return action
    
        ##claude action
    def claude_query(self, obs):
        # Generate prompt for LLM
        prompt1, assist1, prompt2 = self.prompt_design_safe(obs)
        ##ask for claude response
        llm_act = claude_action(prompt1, assist1, prompt2, self.prev_action).strip().split('.')[0]
        ##int action
        action = map_llm_action_to_label(llm_act)
        return action
    
    def step(self, action):

        # Step the wrapped environment and capture all returned values
        obs, dqn_reward, done, truncated, info = self.env.step(action)

        ##combine rewards randomly
        if np.random.rand() <= 1:
            print("llm involved")
            # ##claude
            # llm_response = self.claude_query(obs)
            ##groq
            llm_response = self.groq_query(obs)

            l_acts  = 0

            if llm_response is not None:
                l_acts += 1
                ##modify action accordingly
                llm_act = llm_response

            if l_acts == 1:
                if llm_act == action:
                    llm_reward = 1
                else:
                    llm_reward = 0
                comb_reward = 0.7*dqn_reward + 0.3*llm_reward
            else:
                comb_reward = dqn_reward

        else:
            comb_reward = dqn_reward

        self.prev_action = action_dict[action]

        Reward = 1 / (1 + np.exp(-comb_reward))

        return obs, Reward, done, truncated, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs  # Make sure to return the observation
        
    
if __name__ == "__main__":    

    env_llm = MyHighwayEnvLLM(vehicleCount = 10)

    model = DQN('MlpPolicy', env_llm,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1)

    model.learn(int(2e4))
    model.save('models/prachit_model_claude')