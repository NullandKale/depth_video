import requests
import json

base_url = "http://localhost:5000"

# Test /shows endpoint
response = requests.get(f"{base_url}/shows")
print("Shows:")
print(json.dumps(response.json(), indent=2))
print(response.text)

# Test /seasons endpoint
show_name = "Jake 2.0"
response = requests.post(f"{base_url}/seasons", json={"show_name": show_name})
print(f"Seasons of {show_name}:")
print(json.dumps(response.json(), indent=2))
print(response.text)

# Test /episodes endpoint
show_name = "Jake 2.0"
season = "Season 1"
response = requests.post(f"{base_url}/episodes", json={"show_name": show_name, "season": season})
print(f"Episodes of {show_name}, {season}:")
print(json.dumps(response.json(), indent=2))
print(response.text)

# Test /episode_path endpoint
show_name = "Jake 2.0"
season = "Season 1"
episode = "Jake 2.0 - 1x01 - The Tech.avi"
response = requests.post(f"{base_url}/episode_path", json={"show_name": show_name, "season": season, "episode": episode})
print(f"Path of {show_name}, {season}, {episode}:")
print(json.dumps(response.json(), indent=2))
print(response.text)

# Test /process_video endpoint
video_path = "C:\\Users\\alec\\Desktop\\output_test.mp4"
response = requests.post(f"{base_url}/process_video", json={"video_path": video_path})
print(f"Processing {video_path}:")
print(json.dumps(response.json(), indent=2))
print(response.text)

# Test /process_video endpoint
video_path = "C:\\Users\\alec\\Desktop\\output_test_2.mp4"
response = requests.post(f"{base_url}/process_video", json={"video_path": video_path})
print(f"Processing {video_path}:")
print(json.dumps(response.json(), indent=2))
print(response.text)
