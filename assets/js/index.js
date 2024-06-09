import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils, ImageEmbedder } = vision;
const demosSection = document.getElementById("demos");
const imageBlendShapes = document.getElementById("image-blend-shapes");
const videoBlendShapes = document.getElementById("video-blend-shapes");
const creditScoreLog = document.getElementById("credit-score-log");
let faceLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
let imageEmbedder;
const videoWidth = 480;
const similarityIndicator = indicators.find(
  (indicator) => indicator.categoryName === "faceSimilarity"
);

const createImageEmbedder = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  imageEmbedder = await ImageEmbedder.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite`,
    },
    runningMode: runningMode,
    // quantize: true
  });
  demosSection.classList.remove("invisible");
};
createImageEmbedder();

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU",
    },
    outputFaceBlendshapes: true,
    runningMode,
    numFaces: 10,
  });
  demosSection.classList.remove("invisible");
}
createFaceLandmarker();

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}
// Enable the live webcam view and start detection.
function enableCam(event) {
  if (!faceLandmarker) {
    console.log("Wait! faceLandmarker not loaded yet.");
    return;
  }
  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "START QUIZ";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "STOP QUIZ";
  }
  // getUsermedia parameters.
  const constraints = {
    video: true,
  };
  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}
let lastVideoTime = -1;
let results = undefined;
let uploadImageEmbedderResult;
const drawingUtils = new DrawingUtils(canvasCtx);
async function predictWebcam() {
  const radio = video.videoHeight / video.videoWidth;
  video.style.width = videoWidth + "px";
  video.style.height = videoWidth * radio + "px";
  canvasElement.style.width = videoWidth + "px";
  canvasElement.style.height = videoWidth * radio + "px";
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await faceLandmarker.setOptions({ runningMode: runningMode });
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = faceLandmarker.detectForVideo(video, startTimeMs);
  }

  if (results.faceLandmarks) {
    for (const landmarks of results.faceLandmarks) {
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
        { color: "#C0C0C070", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
        { color: "#FF3030" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
        { color: "#FF3030" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
        { color: "#30FF30" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
        { color: "#30FF30" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LIPS,
        { color: "#E0E0E0" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
        { color: "#FF3030" }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
        { color: "#30FF30" }
      );
    }
  }

  const isFaceDetected = isFacePresent(results.faceLandmarks);
  if (!isFaceDetected) {
    if (
      creditScoreLog.firstChild &&
      creditScoreLog.firstChild.innerText !==
        "No face detected. Please ensure your face is visible"
    ) {
      addLog("No face detected. Please ensure your face is visible");
    }
  }

  const isMultipleFacePresent = checkMultipleFacePresent(results.faceLandmarks);
  if (isMultipleFacePresent) {
    if (
      creditScoreLog.firstChild &&
      creditScoreLog.firstChild.innerText !==
        "Multiple faces detected. Please ensure only one face is visible"
    ) {
      addLog("Multiple faces detected. Please ensure only one face is visible");
    }
  }

  drawBlendShapes(videoBlendShapes, results.faceBlendshapes);

  // if image mode is initialized, create a new embedder with video runningMode

  await imageEmbedder.setOptions({ runningMode: "VIDEO" });
  const embedderResult = await imageEmbedder.embedForVideo(video, startTimeMs);
  if (uploadImageEmbedderResult != null) {
    const similarity = ImageEmbedder.cosineSimilarity(
      uploadImageEmbedderResult.embeddings[0],
      embedderResult.embeddings[0]
    );
    displaySimilarity(similarity);
  }

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
function drawBlendShapes(el, blendShapes) {
  if (!blendShapes.length) {
    return;
  }
  let htmlMaker = "";
  blendShapes[0].categories.map((shape) => {
    // check if indicators exists and compare score then show alert
    indicators.map((indicator) => {
      if (
        indicator.categoryName === shape.categoryName &&
        shape.score > indicator.score
      ) {
        if (
          creditScoreLog.firstChild &&
          creditScoreLog.firstChild.innerText !== indicator.message
        ) {
          addLog(indicator.message);
        }
      }
    });

    htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${
          shape.displayName || shape.categoryName
        }</span>
        <span class="blend-shapes-value" style="width: calc(${
          +shape.score * 100
        }% - 120px)">${(+shape.score).toFixed(4)}</span>
      </li>
    `;
  });
  el.innerHTML = htmlMaker;
}

function isFacePresent(detectionResults) {
  // Check if any faces are detected
  return detectionResults && detectionResults.length > 0;
}

function checkMultipleFacePresent(detectionResults) {
  // Check if any faces are detected
  return detectionResults && detectionResults.length > 1;
}

const fileInput = document.getElementById("getFile");

fileInput.addEventListener("change", async (event) => {
  let reader = new FileReader();
  const output = document.getElementById("embedded_image");
  reader.onload = function () {
    output.src = reader.result;
    output.classList.remove("hidden");
    setTimeout(async function () {
      if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await imageEmbedder.setOptions({ runningMode: runningMode });
      }
      uploadImageEmbedderResult = await imageEmbedder.embed(output);
    }, 100);
  };
  reader.readAsDataURL(event.target.files[0]);
});

function calculateSimilarity(embedding1, embedding2) {
  if (!embedding1 || !embedding2) return null;
  let sum = 0;
  for (let i = 0; i < embedding1.length; i++) {
    const diff = embedding1[i] - embedding2[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function displaySimilarity(similarity) {
  if (similarityIndicator) {
    if (similarity < similarityIndicator.score) {
      if (
        creditScoreLog.firstChild &&
        creditScoreLog.firstChild.innerText !== similarityIndicator.message
      ) {
        addLog(similarityIndicator.message);
      }
    } else {
      if (
        creditScoreLog.firstChild &&
        creditScoreLog.firstChild.innerText !== "validate image is completed"
      ) {
        addLog("validate image is completed");
      }
    }
  }
}

function addLog(message) {
  const li = document.createElement("li");
  li.innerText = message;
  creditScoreLog.prepend(li);
}
