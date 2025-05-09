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


def save_and_go(observations, actions, file_name):
    """
    Saves the generated dataset to a CSV file.
    """
    observations = np.array(observations)
    actions = np.array(actions)

    data = pd.DataFrame(observations)
    data['action'] = actions

    dataset_dir = 'datasets'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    dataset_path = os.path.join(dataset_dir, file_name)
    data.to_csv(dataset_path, index=False)

    print(f"Dataset saved to {dataset_path}")


def generate_dataset_with_groq(env, file_name, total_samples,
                               vehicles_density_range=(1, 5), spacing_range=(1, 3), 
                               lane_id_range=[0, 1, 2, 3], ego_spacing_range=(1, 3)):
    """
    Generates a labeled dataset by varying all four environment configurations,
    capturing observations, using Groq for action recommendations, labeling actions, and saving the dataset.
    """
    observations = []
    actions = []

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

        # Generate prompts for Groq
        prompt1, assist1, prompt2 = env.prompt_design(obs)
        llm_act = groq_action(prompt1, assist1, prompt2)

        # Convert LLM action to a numerical label
        action_label = map_llm_action_to_label(llm_act)
        print(f"Action label: {action_label}")

        # Store observation and corresponding LLM action
        observations.append(obs.flatten())
        actions.append(action_label)

        # Save data after all samples are generated
        save_and_go(observations, actions, file_name)
            


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
            "vehicles_density": 2,
            "show_trajectories": True,
            "render_agent": True,
        }
        self.env = gym.make("highway-v0",config= self.config)
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

    def prompt_design(self, obs_):
        prompt1 = (
            "You are a smart driving assistant. You, the 'ego' car, are now driving on a highway. "
            "You need to recommend ONLY ONE best action among the following set of actions based on the current scenario: "
            "1. IDLE -- maintain the current speed in the current lane "
            "2. FASTER -- accelerate the ego vehicle "
            "3. SLOWER -- decelerate the ego vehicle "
            "4. LANE_LEFT -- change to the adjacent left lane "
            "5. LANE_RIGHT -- change to the adjacent right lane"
        )
        assist1 = (
            "Understood. Please provide the current scenario or conditions, such as traffic density, speed of surrounding vehicles, "
            "your current speed, and any other relevant information, so I can recommend the best action."
        )

        # Extract and organize vehicle state information from observations
        x, y, vx, vy = obs_[:, 1], obs_[:, 2], obs_[:, 3], obs_[:, 4]
        ego_x, ego_y, ego_vx, ego_vy = x[0], y[0], vx[0], vy[0]
        veh_x, veh_y, veh_vx, veh_vy = x[1:] - ego_x, y[1:] - ego_y, vx[1:], vy[1:]

        lanes = y // 4 + 1
        ego_lane = lanes[0]
        veh_lanes = lanes[1:]

        if ego_lane == 1:
            ego_left_lane, ego_right_lane = 'Left lane: Not available\n', f'Right lane: Lane-{ego_lane + 1}\n'
        elif ego_lane == 4:
            ego_left_lane, ego_right_lane = f'Left lane: Lane-{ego_lane - 1}\n', 'Right lane: Not available\n'
        else:
            ego_left_lane, ego_right_lane = f'Left lane: Lane-{ego_lane - 1}\n', f'Right lane: Lane-{ego_lane + 1}\n'

        prompt2 = (
            f"Ego vehicle:\n\tCurrent lane: Lane-{ego_lane}\n\t{ego_left_lane}\t{ego_right_lane}\tCurrent speed: {ego_vx} m/s \n\n"
            "Lane info:\n"
        )
        for i in range(4):
            inds = np.where(veh_lanes == i + 1)[0]
            num_v = len(inds)
            if num_v > 0:
                val, ind = self.find_smallest_positive(veh_x[inds])
                true_ind = inds[ind]
                prompt2 += (
                    f"\tLane-{i + 1}: There are {num_v} vehicle(s) in this lane ahead of ego vehicle, "
                    f"closest being {veh_x[true_ind]} m ahead traveling at {veh_vx[true_ind]} m/s.\n"
                )
            else:
                prompt2 += f"\tLane-{i + 1} No other vehicle ahead of ego vehicle.\n"

        prompt2 += (
            "\nAttention points:\n"
            "\t1. SLOWER has least priority and should be used only when no other action is safe.\n"
            "\t2. DO NOT change lanes frequently.\n"
            "\t3. Safety is priority, but do not forget efficiency.\n"
            "\t4. Your suggested action has to be one from one of the above five listed actions - IDLE, SLOWER, FASTER, LANE_LEFT, LANE_RIGHT. \n"
            f"Your last action was {self.prev_action}. Please recommend action for the current scenario only in this format and DONT propound anything else other than 'Final decision: <final decision>'.\n"
        )

        return prompt1, assist1, prompt2

    def step(self, action):
        """
        Steps the environment with the given action.
        """
        action_dict = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}
        obs, dqn_reward, done, truncated, info = self.env.step(action)

        self.prev_action = action_dict.get(action, 'IDLE')
        reward = 1 / (1 + np.exp(-dqn_reward))

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        """
        Resets the environment.
        """
        obs = self.env.reset(**kwargs)
        return obs
    
    

if __name__ == "__main__":
    env = MyHighwayEnvLLM(vehicleCount=10)

    # Optionally, verify environment setup
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Generate the dataset with varied configuration parameters
    generate_dataset_with_groq(
        env=env,
        file_name='highway_dataset_groq.csv',
        total_samples=1000,  # Generate 100 samples with varied configurations
        vehicles_density_range=(1, 5),
        spacing_range=(2, 20),
        lane_id_range=[0, 1, 2, 3],  # Define initial lanes to explore
        ego_spacing_range=(1, 20)  # Define range for ego vehicle spacing
    )
