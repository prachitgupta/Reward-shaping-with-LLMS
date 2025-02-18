import os
import requests
import time
import boto3
import json
import pandas as pd
import numpy as np
import random

# Define Boto3 client for Bedrock
client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Claude API settings
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
CLAUDE_API_URL = 'https://api.anthropic.com/v1/complete'


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
        if 'Final decision:' in response_json:
            action = response_json.split('Final decision:')[-1].strip().upper()
            return action
        else:
            print(f"Unexpected response format: {response_json}")
            return 'IDLE'
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Claude.ai: {e}")
        time.sleep(1)
        return 'IDLE'


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

    data = pd.DataFrame(observations, columns=[
    'vehicles_in_ego_lane',
    'vehicles_in_left_lane',
    'vehicles_in_right_lane',
    'closest_in_ego_lane_dist',
    'closest_left_lane_dist',
    'closest_right_lane_dist',
    'relative_velocity_ego_lane',
    'relative_velocity_left_lane',
    'relative_velocity_right_lane',
    'previous_actio',
    'action'
])

    data['action'] = actions

    dataset_dir = 'datasets_synthesiesd'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    dataset_path = os.path.join(dataset_dir, file_name)
    data.to_csv(dataset_path, index=False)

    print(f"Dataset saved to {dataset_path}")



# Load the dataset
file_path = 'datasets_try/processed_features.csv'
dataset = pd.read_csv(file_path)

# Initialize an empty list to store the observations and actions
observations = []
actions = []

# Simulate generating prompts, sending them to the Claude API, and recording the response
for _ in range(1):  # Generate 10 scenarios
    # Randomly sample from the dataset
    sample = dataset.sample()

    # Extract features for the current sample
    vehicles_in_ego_lane = sample['vehicles_in_ego_lane'].values[0]
    vehicles_in_left_lane = sample['vehicles_in_left_lane'].values[0]
    vehicles_in_right_lane = sample['vehicles_in_right_lane'].values[0]
    closest_in_ego_lane_dist = sample['closest_in_ego_lane_dist'].values[0]
    closest_left_lane_dist = sample['closest_left_lane_dist'].values[0]
    closest_right_lane_dist = sample['closest_right_lane_dist'].values[0]
    relative_velocity_ego_lane = sample['relative_velocity_ego_lane'].values[0]
    relative_velocity_left_lane = sample['relative_velocity_left_lane'].values[0]
    relative_velocity_right_lane = sample['relative_velocity_right_lane'].values[0]
    prev_action = 'SLOWER'  # Previous action is always SLOWER

    # Generate the prompt based on the features
    ego_lane = random.randint(1, 4)
    ego_speed = random.uniform(20, 26)  # Speed between 10 and 30 m/s

    ego_left_lane = f"Left lane: {'Not available' if ego_lane == 1 else 'Lane-' + str(ego_lane - 1)}\n"
    ego_right_lane = f"Right lane: {'Not available' if ego_lane == 4 else 'Lane-' + str(ego_lane + 1)}\n"

    lane_info = f"Lane info:\n\
    \tLane-1: {vehicles_in_left_lane} vehicles in left lane, closest at {closest_left_lane_dist} m.\n\
    \tLane-2: {vehicles_in_ego_lane} vehicles in ego lane, closest at {closest_in_ego_lane_dist} m.\n\
    \tLane-3: {vehicles_in_right_lane} vehicles in right lane, closest at {closest_right_lane_dist} m.\n"

    prompt1 = f"You are claude, a large language model. You are now acting as a mature driving assistant, who can give accurate and correct advice for human drivers in complex urban driving scenarios. The information in the current scenario:\n\
                You, the 'ego' car, are now driving on a highway. You have already driven for 0 seconds.\n\
                The decision made by the agent LAST time step was 'SLOWER' (decelerate the vehicle)."

    rules = "There are several rules you need to follow when you drive on a highway:\n\
             1. Try to keep a safe distance from the car in front of you.\n\
             2. DON’T change lanes frequently. If you want to change lanes, double-check the safety of vehicles in the target lane."

    att_points = "Here are your attention points:\n\
                        1. You must output a decision when you finish this task. Your final output decision must be unique and not ambiguous. For example, you cannot say \"I can either keep lane or accelerate at the current time\".\n\
                        2. You need to always remember your current lane ID, your available actions, and available lanes before you make any decision.\n\
                        3. Once you have a decision, you should check the safety with all the vehicles affected by your decision.\n\
                        4. If you verify a decision is unsafe, you should start a new one and verify its safety again from scratch."

    assist1 = "Understood. Please provide the current scenario or conditions, such as traffic density, speed of surrounding vehicles, your current speed, and any other relevant information, so I can recommend the best action."

    prompt2 = f"Here is the current scenario:\n\
    There are four lanes on the highway: Lane-1 (leftmost), Lane-2, Lane-3, Lane-4 (rightmost).\n\n\
    Ego vehicle:\n\
    \tCurrent lane: Lane-{ego_lane}\n{ego_left_lane}\t{ego_right_lane}\tCurrent speed: {ego_speed} m/s\n\n\
    {lane_info}"

    safety_verification = f"\nAttention points:\n\
    \t1. Safety is the main priority. You can stay IDLE or even go slower, but in no circumstance should you collide with the lead vehicle.\n\
    \t2. You are not supposed to change lanes frequently. Only change lanes when it's necessary to keep the vehicle safe. Before changing lanes, check safety like safe distance and speed from other vehicles.\n\
    \t3. Safety is a priority, but do not forget efficiency.\n\
    \t4. You should only make a decision once you have verified safety with other vehicles; otherwise, make a new decision and verify its safety from scratch.\n\
    \t5. Your suggested action has to be one from the five listed actions - IDLE, SLOWER, FASTER, LANE_LEFT, LANE_RIGHT.\n\
    Your last action was '{prev_action}'. Please recommend action for the current scenario only in this format and don't propose anything else other than 'Final decision: <final decision>'.\n"

    prompt2 += safety_verification

    # Send to Claude API
    action = claude_action(prompt1, assist1, prompt2)
    action_label = map_llm_action_to_label(action)
    
    observation = [
        vehicles_in_ego_lane,
        vehicles_in_left_lane,
        vehicles_in_right_lane,
        closest_in_ego_lane_dist,
        closest_left_lane_dist,
        closest_right_lane_dist,
        relative_velocity_ego_lane,
        relative_velocity_left_lane,
        relative_velocity_right_lane,
        prev_action
    ]
    observations.append(observation)
    actions.append(action_label)

save_and_go(observations, actions, "prev_action.csv")

