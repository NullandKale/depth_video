// DOM elements
const showSelect = document.getElementById("show-select");
const seasonSelect = document.getElementById("season-select");
const episodeSelect = document.getElementById("episode-select");
const processVideoButton = document.getElementById("process-video");
const episodePathParagraph = document.getElementById("episode-path");

// Load shows when the page is loaded
document.addEventListener("DOMContentLoaded", async () => {
  const shows = await getShows();
  for (const show of shows) {
    const option = document.createElement("option");
    option.value = show;
    option.textContent = show;
    showSelect.appendChild(option);
  }
});

// Load seasons when a show is selected
showSelect.addEventListener("change", async () => {
  seasonSelect.style.display = "none";
  episodeSelect.style.display = "none";
  processVideoButton.style.display = "none";
  episodePathParagraph.style.display = "none";

  const showName = showSelect.value;
  if (!showName) return;

  const seasons = await getSeasons(showName);
  seasonSelect.innerHTML = '<option value="">Select a season</option>';

  for (const season of seasons) {
    const option = document.createElement("option");
    option.value = season;
    option.textContent = season;
    seasonSelect.appendChild(option);
  }

  seasonSelect.style.display = "block";
});

// Load episodes when a season is selected
seasonSelect.addEventListener("change", async () => {
  episodeSelect.style.display = "none";
  processVideoButton.style.display = "none";
  episodePathParagraph.style.display = "none";

  const showName = showSelect.value;
  const season = seasonSelect.value;
  if (!showName || !season) return;

  const episodes = await getEpisodes(showName, season);
  episodeSelect.innerHTML = '<option value="">Select an episode</option>';

  for (const episode of episodes) {
    const option = document.createElement("option");
    option.value = episode;
    option.textContent = episode;
    episodeSelect.appendChild(option);
  }

  episodeSelect.style.display = "block";
});

// Show process video button when an episode is selected
episodeSelect.addEventListener("change", () => {
  processVideoButton.style.display = episodeSelect.value ? "block" : "none";
});

// Process video and show episode path when the button is clicked
processVideoButton.addEventListener("click", async () => {
  const showName = showSelect.value;
  const season = seasonSelect.value;
  const episode = episodeSelect.value;
  if (!showName || !season || !episode) return;

  const episodePath = await getEpisodePath(showName, season, episode);
  const videoStateResponse = await videoState(episodePath);

  if (!videoStateResponse.processed) {
    const processVideoResponse = await processVideo(episodePath);
    if (processVideoResponse.processed) {
      episodePathParagraph.textContent = `Output video path: ${processVideoResponse.output_path}`;
    } else {
      episodePathParagraph.textContent = "Processing video...";
    }
  } else {
    episodePathParagraph.textContent = `Output video path: ${videoStateResponse.output_path}`;
  }

  episodePathParagraph.style.display = "block";
});
