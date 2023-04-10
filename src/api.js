// Define the API endpoint URLs
const BASE_URL = "http://localhost:5000";
const SHOWS_URL = `${BASE_URL}/shows`;
const SEASONS_URL = `${BASE_URL}/seasons`;
const EPISODES_URL = `${BASE_URL}/episodes`;
const EPISODE_PATH_URL = `${BASE_URL}/episode_path`;
const PROCESS_VIDEO_URL = `${BASE_URL}/process_video`;
const VIDEO_STATE_URL = `${BASE_URL}/video_state`;

// Function to make GET requests to the API and return the response JSON
async function fetchJSON(url) {
  const response = await fetch(url);
  const data = await response.json();
  return data;
}

// Function to make POST requests to the API with JSON data and return the response JSON
async function postJSON(url, data) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
  const responseData = await response.json();
  return responseData;
}

// Function to get a list of shows from the API
async function getShows() {
  const shows = await fetchJSON(SHOWS_URL);
  return shows;
}

// Function to get a list of seasons for a given show from the API
async function getSeasons(showName) {
  const data = { show_name: showName };
  const seasons = await postJSON(SEASONS_URL, data);
  return seasons;
}

// Function to get a list of episodes for a given show and season from the API
async function getEpisodes(showName, season) {
  const data = { show_name: showName, season: season };
  const episodes = await postJSON(EPISODES_URL, data);
  return episodes;
}

// Function to get the file path for a given episode from the API
async function getEpisodePath(showName, season, episode) {
  const data = { show_name: showName, season: season, episode: episode };
  const episodePath = await postJSON(EPISODE_PATH_URL, data);
  return episodePath;
}

// Function to initiate video processing for a given video path
async function processVideo(videoPath) {
  const data = { video_path: videoPath };
  const response = await postJSON(PROCESS_VIDEO_URL, data);
  return response;
}

// get video cache state
async function videoState(videoPath) {
    const data = { video_path: videoPath };
    const response = await postJSON(VIDEO_STATE_URL, data);
    return response;
  }
  