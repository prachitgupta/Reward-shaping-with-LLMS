import random
import numpy as np

def generate_random_prompt():
    ego_lane = random.randint(1, 4)
    ego_speed = random.uniform(10, 30)  # Speed between 10 and 30 m/s
    veh_count = random.randint(0, 5)  # Random number of vehicles in a lane
    lanes = [random.randint(1, 4) for _ in range(veh_count)]
    vehicle_data = {
        "positions": np.random.uniform(-500, 500, size=veh_count),  # Random vehicle positions
        "speeds": np.random.uniform(10, 25, size=veh_count)  # Random vehicle speeds
    }
    
    # Lane info for the ego vehicle
    ego_left_lane = f"Left lane: {'Not available' if ego_lane == 1 else 'Lane-' + str(ego_lane - 1)}\n"
    ego_right_lane = f"Right lane: {'Not available' if ego_lane == 4 else 'Lane-' + str(ego_lane + 1)}\n"

    lane_info = "Lane info:\n"
    for i in range(4):
        inds = [j for j, l in enumerate(lanes) if l == i + 1]
        num_v = len(inds)
        if num_v > 0:
            closest_vehicle = min(vehicle_data["positions"][inds])
            closest_speed = vehicle_data["speeds"][inds][0]  # Assuming first vehicle is the closest
            lane_info += f"\tLane-{i + 1}: There are {num_v} vehicle(s) in this lane ahead of ego vehicle, closest being {closest_vehicle} m ahead traveling at {closest_speed} m/s.\n"
        else:
            lane_info += f"\tLane-{i + 1}: No other vehicle ahead of ego vehicle.\n"

    prev_action = 'SLOWER'
    
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
    return prompt1, assist1, prompt2

# Generate 10 random prompts
for _ in range(10):
    prompt1, assist1, prompt2 = generate_random_prompt()
    print("Prompt 1:\n", prompt1)
    print("\nAssist 1:\n", assist1)
    print("\nPrompt 2:\n", prompt2)
    print("\n====================================\n")
