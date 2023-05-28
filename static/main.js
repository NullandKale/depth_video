let frameCount = 0;
let lastUpdateTime = Date.now();
let fps = 0;

function getResolutionFromUI() {
    // Get desired resolution from the UI
    const resolution = document.getElementById('resolutionSelection').value.split('x');
    const desiredWidth = parseInt(resolution[0]);
    const desiredHeight = parseInt(resolution[1]);

    return {desiredWidth, desiredHeight};
}

function computeFPS() {
    let now = Date.now();
    let deltaTime = now - lastUpdateTime;
    fps = frameCount / (deltaTime / 1000);
    frameCount = 0;
    lastUpdateTime = now;

    // Update FPS display
    document.getElementById('fpsCounter').textContent = `FPS: ${fps.toFixed(2)}`;
}

async function updateServerStats() {
    const response = await fetch('http://localhost:5000/stats');
    if (!response.ok) {
        console.error('Failed to fetch server stats: ', response.statusText);
        return;
    }

    const stats = await response.json();

    // Update the UI with the fetched stats
    document.getElementById('totalFrames').textContent = `Total Frames: ${stats.frame_count}`;
    document.getElementById('averageProcessingTime').textContent = `Average Processing Time: ${stats.average_processing_time_ms.toFixed(2)} ms`;
    document.getElementById('framesPerSecond').textContent = `Frames per Second: ${stats.frames_per_second.toFixed(2)}`;
}

async function startProcessing() {
    // Enumerate cameras and choose one
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    
    // You might want to choose a different index or provide a UI for the user to choose
    let desiredDevice = videoDevices[0];  // Picks the first video device by default

    const video = document.getElementById('video');
    
    const {desiredWidth, desiredHeight} = getResolutionFromUI();

    // Request video stream
    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            deviceId: desiredDevice.deviceId,
            width: { ideal: desiredWidth },
            height: { ideal: desiredHeight }
        }
    });

    // Play the video
    video.srcObject = stream;
    await video.play();

    // Process each frame
    // Replace your processFrame function with captureFrame
    setInterval(() => captureFrame(video), 1000 / 30);
    
    // Start the FPS counter
    setInterval(computeFPS, 250);

    // Update the server stats every second
    setInterval(updateServerStats, 250);
}

function processFrame(video) {
    const canvas = drawImageToCanvas(video);
    sendCanvasToServer(canvas);
}

function drawImageToCanvas(image) {
    const {desiredWidth, desiredHeight} = getResolutionFromUI();

    const canvas = document.createElement('canvas');
    canvas.width = desiredWidth;
    canvas.height = desiredHeight;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(image, 0, 0, desiredWidth, desiredHeight);

    return canvas;
}

function sendCanvasToServer(canvas) {
    canvas.toBlob(sendBlobToServer, 'image/jpeg');
}

function sendBlobToServer(blob) {
    const reader = new FileReader();
    reader.onload = () => {
        sendArrayBufferToServer(new Uint8Array(reader.result));
    };
    reader.readAsArrayBuffer(blob);
}

async function sendArrayBufferToServer(arrayBuffer) {
    const streamId = 1;
    const {desiredWidth, desiredHeight} = getResolutionFromUI();
    const timestamp = Date.now();

    const headerBin = createHeader(streamId, desiredWidth, desiredHeight, timestamp);

    const message = new Uint8Array([...headerBin, ...arrayBuffer]);

    const response = await fetch('http://localhost:5000/process', {
        method: 'POST',
        body: message.buffer
    })

    if (!response.ok) {
        console.error('An error occurred: ', response.statusText);
        return;
    }

    const jsonResponse = await response.json();
    // Decode base64 image data and create a Blob
    const byteCharacters = atob(jsonResponse.image);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], {type: 'image/jpeg'});

    const imageBitmap = await createImageBitmap(blob);

    const outputCanvas = document.getElementById('outputCanvas');
    const ctx = outputCanvas.getContext('2d');
    ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    ctx.drawImage(imageBitmap, 0, 0, outputCanvas.width, outputCanvas.height);
    frameCount++;
}

function createHeader(streamId, width, height, timestamp) {
    const streamIdBin = new Uint8Array([streamId]);
    const widthBin = new Uint16Array([width]);
    const heightBin = new Uint16Array([height]);
    const timestampBin = new BigInt64Array([BigInt(timestamp)]);

    return new Uint8Array([
        ...streamIdBin,
        ...new Uint8Array(widthBin.buffer),
        ...new Uint8Array(heightBin.buffer),
        ...new Uint8Array(timestampBin.buffer)
    ]);
}

let frameQueue = [];
let isProcessing = false;

async function processFrameQueue() {
    if (isProcessing || frameQueue.length === 0) return;

    isProcessing = true;

    // Take the first frame from the queue
    let frame = frameQueue.shift();

    try {
        await sendCanvasToServer(frame);
    } catch (error) {
        console.error('An error occurred: ', error);
    }

    isProcessing = false;

    // Continue processing the next frame
    processFrameQueue();
}


function captureFrame(video) {
    const canvas = drawImageToCanvas(video);

    // Add the frame to the queue
    frameQueue.push(canvas);

    // Start processing the queue if not already doing so
    processFrameQueue();
}



window.onload = function() {
    document.getElementById('startButton').onclick = startProcessing;
}
