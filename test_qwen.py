from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import requests
import time
import boto3
import json
import random
import pandas as pd
from botocore.exceptions import ClientError
import numpy as np

# Environment
import gymnasium as gym
import highway_env
from tqdm import trange

# Register highway environment
gym.register_envs(highway_env)

from gymnasium.wrappers import RecordVideo


##for videos
import base64
from pathlib import Path

from gymnasium.wrappers import RecordVideo


import os
import time
from huggingface_hub import InferenceClient

# Load API token from environment variable
api_key = os.getenv("HF_API_TOKEN")

# Initialize Hugging Face client
client = InferenceClient(
    model="Qwen/QwQ-32B",
    api_key=api_key
)

def qwen_action(prompt1, assist1, prompt2, temperature=0.7, max_retries=3):
    """
    Generates a response using the Qwen model and extracts the final decision.
    
    Args:
        prompt1 (str): Initial user query.
        assist1 (str): Context or assistant's prior response.
        prompt2 (str): Follow-up user query.
        temperature (float): Controls randomness (default: 0.7).
        max_retries (int): Number of retry attempts if API fails (default: 3).
    
    Returns:
        str: Extracted decision from the model output or "IDLE" if unsuccessful.
    """
    
    full_prompt = f"{prompt1}\n\n{assist1}\n\n{prompt2}"
    
    for attempt in range(max_retries):
        try:
            # Call Hugging Face Inference API
            output = client.text_generation(
                full_prompt,
                max_new_tokens=50,
                temperature=temperature
            )

            # Ensure output is valid
            if output and "Final decision:" in output:
                action = output.split("Final decision:")[-1].strip().upper()
                return action.split('.')[0]  # Return only the first part of decision
            
            return "IDLE"  # Default return if format is unexpected

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)  # Short delay before retrying

    return "IDLE"  # Return "IDLE" if all retries fail

# Keep the same action mapping function
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

##setup env , "cosd","sind", " longoff", " latoff", " angoff" 
## cosd = {cosd}, sind = {sind}, longoff = {longoff}, latoff = {latoff}, angoff = {angoff}
class MyHighwayEnvLLM(gym.Env):
    """
    Custom Gym environment for highway driving with LLM prompts.
    """
    def __init__(self, vehicleCount):
        super(MyHighwayEnvLLM, self).__init__()
        self.vehicleCount = vehicleCount
        self.prev_action = 'FASTER'

        self.config = {
            # Set conditions that would force actions from minority classes (left, right, fast)
        
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "cos_d","sin_d"],
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
            "vehicles_density":1,
            'initial_spacing': 10,
            
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
    
##claude action
def claude_query(env,obs):
    # Generate prompt for LLM
    prompt1, assist1, prompt2 = env.prompt_design_safe(obs)
    ##ask for claude response
    llm_act = claude_action(prompt1, assist1, prompt2, env.prev_action).strip().split('.')[0]
    ##int action
    action = map_llm_action_to_label(llm_act)
    return action


def generate_dataset_with_claude(env, total_samples, file_name,
                               vehicles_density_range, spacing_range, 
                               lane_id_range=[0, 1, 2, 3], ego_spacing_range=(1, 3)):
    """
    Generates a labeled dataset by varying all four environment configurations,
    capturing observations, using Groq for action recommendations, labeling actions, and saving the dataset.
    """
    observations = []

    # Generate samples by iterating through combinations of configurations
    for sample in trange(total_samples, desc="Dataset Generation"):
        # Randomly sample each configuration parameter from the provided ranges
        vehicles_density = random.uniform(*vehicles_density_range)
        initial_spacing = random.uniform(*spacing_range)
        initial_lane_id = random.choice(lane_id_range)
        ego_spacing = random.uniform(*ego_spacing_range)

        # Apply the configurations to the environment
        env.config['vehicles_density'] = vehicles_density
        env.config['initial_spacing'] = initial_spacing
        env.config['initial_lane_id'] = initial_lane_id
        env.config['ego_spacing'] = ego_spacing

        print(f"\nConfig: Density={vehicles_density}, Initial Spacing={initial_spacing}, "
              f"Initial Lane ID={initial_lane_id}, Ego Spacing={ego_spacing}")

        # Reset the environment with the new configuration
        obs = env.reset()

        # Capture the initial observation
        if isinstance(obs, tuple):
            obs, info = obs  # If reset returns (obs, info)
        else:
            info = {}
        
        # Store observation and corresponding LLM action
        observations.append(obs.flatten())
        
        #observations = np.array(observations)
    
        data = pd.DataFrame(observations)

        dataset_dir = 'datasets'
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        dataset_path = os.path.join(dataset_dir, file_name)
        data.to_csv(dataset_path, index=False)

        print(f"Dataset saved to {dataset_path}")
        

if __name__ == "__main__":
    # Base setting
    vehicleCount = 10

    # Environment configuration
    config = {
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
        "vehicles_density": 1.55,
        'ego_Spacing': 0,
        "show_trajectories": True,
        "render_agent": True,
    }

    # Create directory for action predictions storage if it doesn't exist
    predictions_dir = 'predictions/ANALYSIS'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    env = gym.make('highway-v0', render_mode='rgb_array', config=config)

    num_episodes = 5

    # Results storage
    results = {
        'qwen': {
            'num_collisions': 0,
            'total_score': 0,
            'episode_data': []
        },
    }

    # Run test episodes for both models
    for episode in trange(num_episodes, desc='Evaluating episodes'):
        # Determine which model to use
        collision_occurred = False
        episode_score = 0
        episode_actions = []
        
        actions_file_path = os.path.join(predictions_dir, f'episode_{episode}_actions.txt')
        
        with open(actions_file_path, 'w') as action_file:
            (obs, info), done, truncated = env.reset(), False, False
            while not (done or truncated):
                # Generate prompt for LLM
                prompt1, assist1, prompt2 = MyHighwayEnvLLM(vehicleCount).prompt_design_safe(obs)
                
                # Get action from current model
                llm_act = qwen_action(prompt1, assist1, prompt2).strip().split('.')[0]
                action = map_llm_action_to_label(llm_act)
                
                obs, reward, done, truncated, info = env.step(int(action))
                episode_score += reward
                episode_actions.append(action)
                
                # Log the action
                action_file.write(f"Action: {action}\n")
                
                # Check for collision
                if "crashed" in info and info["crashed"]:
                    collision_occurred = True

        # Store results
        results['qwen']['episode_data'].append({
            'episode': episode,
            'score': episode_score,
            'collision': collision_occurred,
            'actions': episode_actions
        })
        
        if collision_occurred:
            results['qwen']['num_collisions'] += 1
        results['qwen']['total_score'] += episode_score

    # Close the environment
    env.close()

    # Print and save results
    
    print(f"\nResults for Qwen:")
    print(f"Total Collisions: {results['qwen']['num_collisions']}/{num_episodes}")
    print(f"Total Score: {results['qwen']['total_score']}")
    print(f"Average Score per Episode: {results['qwen']['total_score'] / num_episodes:.2f}")
    print(f"Collision Rate: {results['qwen']['num_collisions'] /num_episodes:.2%}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(model_results['episode_data'])
    results_df.to_csv(f'{predictions_dir}/{model_name}_results.csv', index=False)



