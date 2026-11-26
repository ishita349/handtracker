
import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const status = document.getElementById("status");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");

const FINGERS = {
    Thumb:  [2, 3, 4],
    Index:  [5, 6, 7],
    Middle: [9, 10, 11],
    Ring:   [13, 14, 15],
    Pinky:  [17, 18, 19]
};

const CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
];

const alpha = 0.3;

let smoothed = null;
let detector = null;
let stream = null;
let running = false;
let animationId = null;
let lastVideoTime = -1;

const names = Object.keys(FINGERS);

// Create the finger percentage bars
for (const name of names) {

    document.getElementById("fingers").insertAdjacentHTML(
        "beforeend",
        `
        <div class="finger">
            <div class="finger-label">
                <span>${name}</span>
                <span class="percentage"
                      id="${name}-value">0%</span>
            </div>

            <div class="bar">
                <div class="fill"
                     id="${name}-bar"></div>
            </div>
        </div>
        `
    );
}

// Equivalent of your NumPy angle calculation
function calculateAngle(a, b, c) {

    const v1 = [
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    ];

    const v2 = [
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    ];

    const magnitude =
        Math.hypot(...v1) * Math.hypot(...v2);

    if (magnitude < 1e-8) {
        return 180;
    }

    const dot =
        v1[0] * v2[0] +
        v1[1] * v2[1] +
        v1[2] * v2[2];

    const cosine = Math.max(
        -1,
        Math.min(1, dot / magnitude)
    );

    return Math.acos(cosine) * 180 / Math.PI;
}

// Initialize the pretrained MediaPipe model
async function initializeDetector() {

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm"
    );

    detector = await HandLandmarker.createFromOptions(
        vision,
        {
            baseOptions: {
                modelAssetPath: "./hand_landmarker.task"
            },

            runningMode: "VIDEO",
            numHands: 1,

            minHandDetectionConfidence: 0.7,
            minHandPresenceConfidence: 0.7,
            minTrackingConfidence: 0.7
        }
    );
}

// Draw the green skeleton and red joints
function drawHand(landmarks) {

    const w = canvas.width;
    const h = canvas.height;

    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 2;

    for (const [a, b] of CONNECTIONS) {

        ctx.beginPath();

        ctx.moveTo(
            landmarks[a].x * w,
            landmarks[a].y * h
        );

        ctx.lineTo(
            landmarks[b].x * w,
            landmarks[b].y * h
        );

        ctx.stroke();
    }

    ctx.fillStyle = "#ff0000";

    for (const point of landmarks) {

        ctx.beginPath();

        ctx.arc(
            point.x * w,
            point.y * h,
            5,
            0,
            2 * Math.PI
        );

        ctx.fill();
    }
}

// Update the finger movement values
function updateDisplay(values) {

    const integers = values.map(
        value => Math.floor(value)
    );

    names.forEach((name, i) => {

        document.getElementById(
            `${name}-value`
        ).textContent = `${integers[i]}%`;

        document.getElementById(
            `${name}-bar`
        ).style.width = `${integers[i]}%`;
    });

    document.getElementById("values").textContent =
        `[${integers.join(", ")}]`;
}

// Real-time tracking loop
function trackHands() {

    if (!running) return;

    try {

        if (
            video.readyState >= 2 &&
            video.currentTime !== lastVideoTime
        ) {

            lastVideoTime = video.currentTime;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            ctx.clearRect(
                0, 0,
                canvas.width,
                canvas.height
            );

            const result = detector.detectForVideo(
                video,
                performance.now()
            );

            if (result.landmarks.length > 0) {

                status.textContent = "Hand detected";

                const landmarks = result.landmarks[0];
                const world = result.worldLandmarks[0];

                drawHand(landmarks);

                const curls = Object.values(FINGERS).map(
                    ([a, b, c]) => {

                        const angle = calculateAngle(
                            world[a],
                            world[b],
                            world[c]
                        );

                        return Math.max(
                            0,
                            Math.min(100, 180 - angle)
                        );
                    }
                );

                // Same exponential smoothing as Python
                if (smoothed === null) {

                    smoothed = curls;

                } else {

                    smoothed = curls.map(
                        (value, i) =>
                            alpha * value +
                            (1 - alpha) * smoothed[i]
                    );
                }

                updateDisplay(smoothed);

            } else {

                smoothed = null;
                status.textContent = "No hand detected";

                updateDisplay([0, 0, 0, 0, 0]);
            }
        }

        animationId = requestAnimationFrame(trackHands);

    } catch (error) {

        stopCamera();
        status.textContent = error.message;
        console.error(error);
    }
}

// Start webcam
startButton.addEventListener("click", async () => {

    startButton.disabled = true;

    try {

        status.textContent = "Loading hand tracking model...";

        if (!detector) {
            await initializeDetector();
        }

        status.textContent = "Requesting camera access...";

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = stream;

        await video.play();

        lastVideoTime = -1;
        running = true;
        stopButton.disabled = false;

        trackHands();

    } catch (error) {

        stopCamera();

        status.textContent =
            "Could not start: " + error.message;

        console.error(error);
    }
});

// Stop webcam and release camera resources
function stopCamera() {

    running = false;

    if (animationId !== null) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }

    if (stream) {
        stream.getTracks().forEach(
            track => track.stop()
        );
        stream = null;
    }

    video.srcObject = null;

    ctx.clearRect(
        0, 0,
        canvas.width,
        canvas.height
    );

    smoothed = null;
    updateDisplay([0, 0, 0, 0, 0]);

    startButton.disabled = false;
    stopButton.disabled = true;

    status.textContent = "Camera is disabled";
}

stopButton.addEventListener("click", stopCamera);

window.addEventListener("pagehide", stopCamera);
