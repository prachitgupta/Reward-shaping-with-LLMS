import numpy as np
import pandas as pd

# Define a function to extract the desired features
def extract_features_from_dataset(data):
    """
    Extract features from the dataset based on the given criteria.
    """
    processed_data = []

    for i, row in enumerate(data):
        # Ego vehicle features
        ego_features = row[:5]
        ego_lane = ego_features[2] // 4 + 1  # Lane ID of ego vehicle
        ego_speed = ego_features[3]  # Speed of ego vehicle

        # Other vehicles' features
        other_vehicles = row[5:50].reshape(9, 5)  # 9 vehicles, 5 features each
        actions = row[50]
        
        # Separate features of other vehicles
        other_lanes = other_vehicles[:, 2] // 4 + 1  # Lane IDs of other vehicles
        distances = np.abs(other_vehicles[:, 1] - ego_features[1])  # Distances from ego vehicle
        relative_velocities = other_vehicles[:, 3] - ego_speed  # Relative velocities

        # Number of vehicles in ego lane and adjacent lanes
        vehicles_in_ego_lane = np.sum(other_lanes == ego_lane)
        vehicles_in_left_lane = np.sum(other_lanes == ego_lane - 1)
        vehicles_in_right_lane = np.sum(other_lanes == ego_lane + 1)

        ## Closest vehicles
        closest_ego_index = np.where(other_lanes == ego_lane, distances, np.inf).argmin() if vehicles_in_ego_lane != 0 else np.nan
        closest_left_index = np.where(other_lanes == ego_lane - 1, distances, np.inf).argmin() if vehicles_in_left_lane != 0 else np.nan
        closest_right_index = np.where(other_lanes == ego_lane + 1, distances, np.inf).argmin() if vehicles_in_right_lane != 0 else np.nan

        # Distances of other vehicles
        ## Ego lane
        if np.isnan(closest_ego_index):
            closest_in_ego_lane_dist = 10000  # Assign large value for no vehicle
            relative_velocity_ego_lane = 10000
        else:
            closest_in_ego_lane_dist = distances[closest_ego_index]
            relative_velocity_ego_lane = relative_velocities[closest_ego_index]
        
        ## Left lane
        if np.isnan(closest_left_index):
            # Check if the left lane is non-existent (i.e., topmost lane)
            if ego_lane == 1:
                closest_left_lane_dist = 0  # No lane to the left of topmost lane
                relative_velocity_left_lane = 0  
            else:
                closest_left_lane_dist = 10000  # No vehicle in left lane
                relative_velocity_left_lane = 10000  # No vehicle in left lane
        else:
            closest_left_lane_dist = distances[closest_left_index]
            relative_velocity_left_lane = relative_velocities[closest_left_index]
        
        ## Right lane
        if np.isnan(closest_right_index):
            # Check if the right lane is non-existent (i.e., bottommost lane)
            if ego_lane == 4:
                closest_right_lane_dist = 0  # No lane to the right of bottommost lane
                relative_velocity_right_lane = 0  # No vehicle in non-existent lane
            else:
                closest_right_lane_dist = 10000  # No vehicle in right lane
                relative_velocity_right_lane = 10000  # No vehicle in right lane
        else:
            closest_right_lane_dist = distances[closest_right_index]
            relative_velocity_right_lane = relative_velocities[closest_right_index]

        previous_action_1 = data[i - 1, 50] if i > 0 else 0
        #previous_action_2 = data[i - 2, 50] if i > 1 else 0
            
        # Append computed features
        processed_data.append([
            vehicles_in_ego_lane,
            vehicles_in_left_lane,
            vehicles_in_right_lane,
            closest_in_ego_lane_dist,
            closest_left_lane_dist,
            closest_right_lane_dist,
            relative_velocity_ego_lane,
            relative_velocity_left_lane,
            relative_velocity_right_lane,
            previous_action_1,
            #previous_action_2,
            actions
        ])

    return np.array(processed_data)

# Load the dataset
file_path = 'datasets/claude_5k.csv'
data = pd.read_csv(file_path).values

# Extract features from the dataset
processed_features = extract_features_from_dataset(data)

# Convert processed features to a DataFrame and save it
processed_df = pd.DataFrame(processed_features, columns=[
    'vehicles_in_ego_lane',
    'vehicles_in_left_lane',
    'vehicles_in_right_lane',
    'closest_in_ego_lane_dist',
    'closest_left_lane_dist',
    'closest_right_lane_dist',
    'relative_velocity_ego_lane',
    'relative_velocity_left_lane',
    'relative_velocity_right_lane',
    'previous_action_1',
    #'previous_action_2',
    'action'
])

# Save processed dataset
processed_df.to_csv('datasets_try/processed_features5.csv', index=False)
print("Processed dataset saved as 'processed_features5.csv'")
