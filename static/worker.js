self.addEventListener('message', async (event) => {
    const arrayBuffer = event.data;
    
    const streamId = 1;
    const desiredWidth = event.data.width;
    const desiredHeight = event.data.height;
    const timestamp = Date.now();

    const headerBin = createHeader(streamId, desiredWidth, desiredHeight, timestamp);
    
    const message = new Uint8Array([...headerBin, ...new Uint8Array(arrayBuffer)]);

    const response = await fetch('http://localhost:5000/process', {
        method: 'POST',
        body: message.buffer
    })

    if (!response.ok) {
        throw new Error(`An error occurred: ${response.statusText}`);
    }

    const jsonResponse = await response.json();

    // Pass the jsonResponse back to the main thread
    self.postMessage(jsonResponse);
});

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
