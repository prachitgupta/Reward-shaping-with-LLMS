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

def claude_query(env,obs):
    # Generate prompt for LLM
    prompt1, assist1, prompt2 = env.prompt_design_safe(obs)
    ##ask for claude response
    llm_act = claude_action(prompt1, assist1, prompt2, env.prev_action).strip().split('.')[0]
    ##int action
    action = map_llm_action_to_label(llm_act)
    return action

##setup env , "cosd","sind", " longoff", " latoff", " angoff" 
## cosd = {cosd}, sind = {sind}, longoff = {longoff}, latoff = {latoff}, angoff = {angoff}
class MyRoundaboutEnvLLM(gym.Env):
    """
    Custom Gym environment for highway driving with LLM prompts.
    """
    def __init__(self):
        super(MyRoundaboutEnvLLM, self).__init__()
        self.prev_action = 'FASTER'

        self.config = {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "see_behind": True,
                "duration": 11,
                # "vehicles_count": 4,
                "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-15, 15],
                        "vy": [-15, 15],
                    },
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(0, 32, 9),
            },
            "duration": 40,
            "vehicles_density": 1.5,
            "show_trajectories": True,
            "render_agent": True,
        }
        #
        self.env = gym.make("roundabout-v0",render_mode='rgb_array', config= self.config)
        self.action_space = self.env.action_space

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, 5), dtype=np.float32)

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
                    You, the \'ego\' car, are now driving on a highway and about to encounter a roundabout. You have already driven for 0 seconds.\n\
                    The decision made by the agent LAST time step was \'FASTER\' (accelerate the vehicle).'
                    
        env_description = "ego-vehicle if approaching a roundabout with flowing traffic. \
                           ego_vehicle has to handle lane changes and longitudinal control to pass the roundabout as fast as possible while avoiding collisions.\n \
                           roundabout has flowing traffic so ego should check for the the collision before merging and aslo on the roundabout especially when other vehicles are exiting the roundabout in middle"

        # Part 2: Driving rules that must be followed
        rules = "There are several rules you need to follow when you drive on a highway:\n\
                1. Try to keep a safe distance from the car in front of you.\n\
                2. DON’T change your lane frequently. If you want to change lanes, double-check the safety of vehicles in the target lane.\n \
                3. On encountering the roundabout, you must complete half a circle and then continue moving forward."
        
        prompt1 += env_description + rules

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
        There are 2 lanes on the roundabout: Lane-1, Lane-2.\n\n'
        
        x, y, vx, vy = obs_[:, 1], obs_[:, 2], obs_[:, 3], obs_[:, 4]

        # Step 1: Get ego vehicle's current position (x, y coordinates)
        ego_position = self.env.unwrapped.vehicle.position  # This gives you the (x, y) tuple
        ego_heading = self.env.unwrapped.vehicle.heading
        # Step 2: Get the current lane object using its start and end node names
        ego_lane = self.env.unwrapped.road.network.get_closest_lane_index(ego_position, ego_heading)[2]
        ego_x, ego_y = ego_position
        ego_vx, ego_vy = vx[0] , vy[0]
        
        # Other vehicles' relative positions
        # Exclude the ego vehicle and get only other vehicles
        all_vehicles = self.env.unwrapped.road.vehicles
        other_vehicles = [v for v in all_vehicles if v is not self.env.unwrapped.vehicle]

        # Access position and heading of non-ego vehicles
        veh_lanes = []
        for vehicle in other_vehicles:
            vehicle_position = vehicle.position
            vehicle_heading = vehicle.heading
            veh_lane = self.env.unwrapped.road.network.get_closest_lane_index(vehicle_position, vehicle_heading)[2]
            veh_lanes.append(veh_lane)
        
        veh_x, veh_y = x[1:] - ego_x, y[1:] - ego_y
        veh_vx, veh_vy = vx[1:], vy[1:]
        veh_lanes = np.array(veh_lanes)
        
        # lane availability based on the ego vehicle's current lane
        if ego_lane == 1:
            ego_left_lane = 'Left lane: Not available\n'
            ego_right_lane = 'Right lane: Lane-' + str(ego_lane - 1) + '\n'
        else:
            ego_left_lane = 'Left lane: Lane-' + str(ego_lane + 1) + '\n'
            ego_right_lane = 'Right lane: Not available\n'

        # Append ego vehicle information to prompt2
        prompt2 += 'Ego vehicle:\n\
        \tCurrent lane: Lane-' + str(ego_lane) + '\n' + '\t' + ego_left_lane + '\t' + ego_right_lane + '\tCurrent speed: ' + str(ego_vy) + ' m/s\n\n'

        # Lane information including vehicles ahead in each lane
        lane_info = 'Lane info:\n'
        for i in range(2):
            inds = np.where(veh_lanes == i)[0]
            num_v = len(inds)
            if num_v > 0:
                # Find the closest vehicle in the current lane
                val, ind = self.find_smallest_positive(veh_y[inds])
                true_ind = inds[ind]
                lane_info += '\tLane-' + str(i) + ': There are ' + str(num_v) + ' vehicle(s) in this lane ahead of ego vehicle, closest being ' + str(abs(veh_y[true_ind])) + ' m ahead traveling at ' + str(abs(veh_vy[true_ind])) + ' m/s.\n'
            else:
                lane_info += '\tLane-' + str(i) + ': No other vehicle ahead of ego vehicle.\n'
        
        # Append lane information to prompt2
        prompt2 += lane_info

        # Part 6: Adding additional attention points and the final decision instruction
        safety_verification = '\nAttention points:\n\
        \t1.Safety is the main priority, You can stay IDLE or even Go slower but in no circumstance you should collide with lead vehicle.\n\
        \t2.before merging in roundabout i.e when you see vehicles in your lane you must avoid collision at all cost even if it requires to go slow\n\
        \t3. Safety is a priority, but do not forget efficiency.so avoid decesions leading to excessive idling that is maintaining the same speed or deaccelerating even when situation is favourable\n\
        \t4. you should only make a decesion once you have verified safety with other vehicles otherwise make a new decesion and verify its safety from scratch\n \
        \t5. Your suggested action has to be one from the five listed actions - IDLE, SLOWER, FASTER, LANE_LEFT, LANE_RIGHT.\n\
        Your last action was ' + self.prev_action + '.Please recommend action for the current scenario only in this format and DONT propound anything else other than \'Final decision: <final decision>\'.\n'

        # Append the attention information to prompt2
        prompt2 += safety_verification

        # Return the three prompts
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


# def record_videos(env, video_folder="videos"):
#     wrapped = RecordVideo(
#         env, video_folder=video_folder, episode_trigger=lambda e: True
#     )

#     # Capture intermediate frames
#     env.unwrapped.set_record_video_wrapper(wrapped)

#     return wrapped


# def show_videos(path="videos"):
#     html = []
#     for mp4 in Path(path).glob("*.mp4"):
#         video_b64 = base64.b64encode(mp4.read_bytes())
#         html.append(
#             """<video alt="{}" autoplay
#                       loop controls style="height: 400px;">
#                       <source src="data:video/mp4;base64,{}" type="video/mp4" />
#                  </video>""".format(
#                 mp4, video_b64.decode("ascii")
#             )
#         )
#     ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
    


if __name__ == "__main__":


    # Environment configuration
    config = {
                "observation": {
                    "type": "Kinematics",
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "absolute": True,
                    "normalize": False,
                    "see_behind": True,
                    "duration": 11,
                    # "vehicles_count": 4,
                    "features_range": {
                            "x": [-100, 100],
                            "y": [-100, 100],
                            "vx": [-15, 15],
                            "vy": [-15, 15],
                        },
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": np.linspace(0, 32, 9),
                },
                "duration": 40,
                "vehicles_density": 1.25,
                "show_trajectories": True,
                "render_agent": True,
            }


        # Create directory for action predictions storage if it doesn't exist
        # Create directory for action predictions storage if it doesn't exist

    predictions_dir = 'predictions/roundabout_analysis'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # Create and configure environment
    env = gym.make("roundabout-v0",render_mode='rgb_array', config= config)
    # Wrap environment with video recording
    #env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda episode_id: True)

    prev_action = 4
    num_collisions = 0
    total_score = 0
    num_episodes = 50

    # Run test episodes and record them
    for episode in trange(num_episodes, desc='Evaluating episodes'):
        collision_occurred = False
        episode_score = 0
        
        actions_file_path = os.path.join(predictions_dir, f'episode_{episode}_actions_normal.txt')
        
        with open(actions_file_path, 'w') as action_file:
            (obs, info), done, truncated = env.reset(), False, False
            while not (done or truncated):
                prompt1, assist1, prompt2 =  MyRoundaboutEnvLLM().prompt_design_safe(obs)
                
                # Get action from current model
                llm_act = claude_action(prompt1, assist1, prompt2).strip().split('.')[0]
                action = map_llm_action_to_label(llm_act)
               
                obs, reward, done, truncated, info = env.step(int(action))
                episode_score += reward  # Accumulate episode score
                
                # Log the action and probabilities
                action_file.write(f"Action: {action}\n")
                
                prev_action = action
                
                # Check for collision
                if "crashed" in info and info["crashed"]:
                    collision_occurred = True
                    print(f"Collision occurred in episode {episode}")

        if collision_occurred:
            num_collisions += 1
        total_score += episode_score

    # Close the environment
    env.close()

    # Print results
    print(f"Total Collisions: {num_collisions}/{num_episodes}")
    print(f"Total Score: {total_score}")
    print(f"Average Score per Episode: {total_score / num_episodes:.2f}")
    print(f"Collision Rate: {num_collisions / num_episodes:.2%}")
