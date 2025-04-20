# import os
# from huggingface_hub import InferenceClient

# # Load API token from environment variable
# api_key = os.getenv("HF_API_TOKEN")

# # Initialize the client
# client = InferenceClient(
#     model="Qwen/QwQ-32B-AWQ",
#     api_key=api_key
# )

# # Define the prompt
# prompt = "Explain reinforcement learning in simple terms."

# # Generate response
# output = client.text_generation(prompt, max_new_tokens=512)

# # Print the response
# print(output)

import pandas as pd

# Define the comparison data
data = {
    "Criteria": [
        "Funding Opportunities",
        "Core & Elective Courses",
        "Expenditure / ROI",
        "Labs in HRI & Safe Control",
        "Labs in Robotics",
        "Chances of Success (Career)",
        "Competition & Peer Group"
    ],
    "CMU - Mechanical Engineering": [
        "High availability of RA/TA; strong interdisciplinary funding through RI and SCS",
        "Advanced robotics, HRI, safe control, ML, access to RI and SCS courses",
        "High tuition and cost of living (~$90K/year); excellent placement in big tech and academia",
        "Access to HRT Lab, Safe AI Lab, and strong work on control under uncertainty",
        "Biorobotics Lab, AirLab, robust links with RI; autonomous systems and perception",
        "High success rate in robotics careers, direct link to top companies and startups",
        "Very high competition; elite peer group from around the world"
    ],
    "UIUC - Mechanical Engineering": [
        "Moderate RA/TA availability; strong research assistantship support from CSL",
        "Solid robotics curriculum with electives in control, mechatronics, and ML",
        "Lower tuition (~$60K/year) and cost of living; strong ROI with research opportunities",
        "CoHRR and Safe Autonomy research in CSL; fewer HRI-dedicated labs than CMU",
        "CSL Robotics, mechanical control groups; wide range but fewer cutting-edge labs",
        "Good academic and industry placement; fewer direct robotics-focused roles than CMU",
        "Competitive but more balanced peer group; strong Midwest research community"
    ],
    "Rating (1-5)": [
        "CMU: 5 | UIUC: 4",
        "CMU: 5 | UIUC: 4",
        "CMU: 3 | UIUC: 5",
        "CMU: 5 | UIUC: 3",
        "CMU: 5 | UIUC: 4",
        "CMU: 5 | UIUC: 4",
        "CMU: 5 | UIUC: 4"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)
df
