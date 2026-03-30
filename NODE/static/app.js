const state = {
  fileName: "",
  originalDataUrl: "",
  bicubicDataUrl: "",
  srDataUrl: "",
  bicubicImage: null,
  srImage: null,
  boxSize: 96,
  zoomFactor: 3,
  selectionMode: "bicubic",
  roi: null,
  imageWidth: 0,
  imageHeight: 0,
};

const elements = {
  imageInput: document.getElementById("image-input"),
  predictBtn: document.getElementById("predict-btn"),
  resetBtn: document.getElementById("reset-btn"),
  boxSize: document.getElementById("box-size"),
  boxSizeLabel: document.getElementById("box-size-label"),
  zoomFactor: document.getElementById("zoom-factor"),
  selectionMode: document.getElementById("selection-mode"),
  statusTitle: document.getElementById("status-title"),
  statusBody: document.getElementById("status-body"),
  predictionMeta: document.getElementById("prediction-meta"),
  bicubicCanvas: document.getElementById("bicubic-canvas"),
  srCanvas: document.getElementById("sr-canvas"),
  bicubicOverlay: document.getElementById("bicubic-overlay"),
  srOverlay: document.getElementById("sr-overlay"),
  bicubicZoom: document.getElementById("bicubic-zoom"),
  srZoom: document.getElementById("sr-zoom"),
  diffZoom: document.getElementById("diff-zoom"),
};

function setStatus(title, body) {
  elements.statusTitle.textContent = title;
  elements.statusBody.textContent = body;
}

function updatePredictionMeta(text) {
  elements.predictionMeta.textContent = text;
}

function dataUrlToImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Unable to load image preview."));
    img.src = dataUrl;
  });
}

function get2dContext(canvas, willReadFrequently = false) {
  return canvas.getContext("2d", willReadFrequently ? { willReadFrequently: true } : undefined);
}

function getImageAtMode() {
  return state.selectionMode === "sr" ? state.srImage : state.bicubicImage;
}

function ensureCanvasSize(canvas, image) {
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
}

function drawImage(canvas, image) {
  if (!canvas || !image) {
    return;
  }
  ensureCanvasSize(canvas, image);
  const context = get2dContext(canvas, true);
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function computeRoiFromCenter(centerX, centerY, boxSize, width, height) {
  const size = Math.min(boxSize, width, height);
  const half = Math.floor(size / 2);
  let left = Math.round(centerX - half);
  let top = Math.round(centerY - half);
  left = clamp(left, 0, Math.max(0, width - size));
  top = clamp(top, 0, Math.max(0, height - size));
  return { left, top, size };
}

function updateOverlay(overlay, roi, canvas) {
  if (!overlay || !roi || !canvas) {
    overlay.style.display = "none";
    return;
  }

  overlay.style.display = "block";
  overlay.style.left = `${(roi.left / canvas.width) * 100}%`;
  overlay.style.top = `${(roi.top / canvas.height) * 100}%`;
  overlay.style.width = `${(roi.size / canvas.width) * 100}%`;
  overlay.style.height = `${(roi.size / canvas.height) * 100}%`;
}

function drawZoom(canvas, sourceCanvas, roi, zoomFactor, mode = "rgb") {
  const context = get2dContext(canvas, true);
  const targetSize = Math.max(roi.size * zoomFactor, 1);
  canvas.width = targetSize;
  canvas.height = targetSize;
  context.imageSmoothingEnabled = false;

  const imageData = get2dContext(sourceCanvas, true).getImageData(roi.left, roi.top, roi.size, roi.size);

  if (mode === "diff") {
    const output = context.createImageData(roi.size, roi.size);
    for (let index = 0; index < imageData.data.length; index += 4) {
      const r = imageData.data[index];
      const g = imageData.data[index + 1];
      const b = imageData.data[index + 2];
      const avg = Math.round((r + g + b) / 3);
      const boosted = clamp(avg * 4, 0, 255);
      output.data[index] = boosted;
      output.data[index + 1] = Math.round(boosted * 0.42);
      output.data[index + 2] = Math.round(boosted * 0.12);
      output.data[index + 3] = 255;
    }
    const temp = document.createElement("canvas");
    temp.width = roi.size;
    temp.height = roi.size;
    temp.getContext("2d").putImageData(output, 0, 0);
    context.drawImage(temp, 0, 0, roi.size, roi.size, 0, 0, targetSize, targetSize);
    return;
  }

  const temp = document.createElement("canvas");
  temp.width = roi.size;
  temp.height = roi.size;
  temp.getContext("2d").putImageData(imageData, 0, 0);
  context.drawImage(temp, 0, 0, roi.size, roi.size, 0, 0, targetSize, targetSize);
}

function renderAll() {
  const roi = state.roi;
  const zoomFactor = Number(elements.zoomFactor.value);
  state.zoomFactor = zoomFactor;
  state.boxSize = Number(elements.boxSize.value);
  elements.boxSizeLabel.textContent = `${state.boxSize} px`;

  if (!state.bicubicImage || !state.srImage) {
    return;
  }

  drawImage(elements.bicubicCanvas, state.bicubicImage);
  drawImage(elements.srCanvas, state.srImage);
  updateOverlay(elements.bicubicOverlay, roi, elements.bicubicCanvas);
  updateOverlay(elements.srOverlay, roi, elements.srCanvas);

  if (!roi) {
    return;
  }

  drawZoom(elements.bicubicZoom, elements.bicubicCanvas, roi, zoomFactor, "rgb");
  drawZoom(elements.srZoom, elements.srCanvas, roi, zoomFactor, "rgb");

  const bicubicContext = get2dContext(elements.bicubicCanvas, true);
  const srContext = get2dContext(elements.srCanvas, true);
  const bicubicCrop = bicubicContext.getImageData(roi.left, roi.top, roi.size, roi.size);
  const srCrop = srContext.getImageData(roi.left, roi.top, roi.size, roi.size);
  const diffCanvas = document.createElement("canvas");
  diffCanvas.width = roi.size;
  diffCanvas.height = roi.size;
  const diffContext = get2dContext(diffCanvas, true);
  const diffImage = diffContext.createImageData(roi.size, roi.size);

  for (let index = 0; index < diffImage.data.length; index += 4) {
    const r = Math.abs(bicubicCrop.data[index] - srCrop.data[index]);
    const g = Math.abs(bicubicCrop.data[index + 1] - srCrop.data[index + 1]);
    const b = Math.abs(bicubicCrop.data[index + 2] - srCrop.data[index + 2]);
    const energy = Math.min(255, Math.round((r + g + b) * 1.5));
    diffImage.data[index] = energy;
    diffImage.data[index + 1] = Math.round(energy * 0.35);
    diffImage.data[index + 2] = Math.round(energy * 0.12);
    diffImage.data[index + 3] = 255;
  }

  diffContext.putImageData(diffImage, 0, 0);
  const diffZoomContext = get2dContext(elements.diffZoom, true);
  elements.diffZoom.width = roi.size * zoomFactor;
  elements.diffZoom.height = roi.size * zoomFactor;
  diffZoomContext.imageSmoothingEnabled = false;
  diffZoomContext.drawImage(diffCanvas, 0, 0, roi.size, roi.size, 0, 0, elements.diffZoom.width, elements.diffZoom.height);
}

function setDefaultRoi() {
  if (!state.bicubicImage) {
    return;
  }
  const image = state.selectionMode === "sr" ? state.srImage : state.bicubicImage;
  const size = Number(elements.boxSize.value);
  const roi = computeRoiFromCenter(image.naturalWidth / 2, image.naturalHeight / 2, size, image.naturalWidth, image.naturalHeight);
  state.roi = roi;
  renderAll();
}

function pointerToImageCoordinates(event, canvas) {
  const image = getImageAtMode();
  if (!image) {
    return null;
  }

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;
  return { x, y };
}

async function handlePredict() {
  if (!state.originalDataUrl) {
    return;
  }

  elements.predictBtn.disabled = true;
  setStatus("Generating SR", "The uploaded image is being passed through the local checkpoint.");
  updatePredictionMeta("Running inference...");

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      imageDataUrl: state.originalDataUrl,
      filename: state.fileName,
    }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Prediction request failed.");
  }

  const result = await response.json();
  state.bicubicDataUrl = result.bicubicDataUrl;
  state.srDataUrl = result.srDataUrl;
  state.imageWidth = result.inputWidth * result.scale;
  state.imageHeight = result.inputHeight * result.scale;

  [state.bicubicImage, state.srImage] = await Promise.all([
    dataUrlToImage(state.bicubicDataUrl),
    dataUrlToImage(state.srDataUrl),
  ]);

  setStatus(
    `Ready: ${result.modelName}`,
    `${result.inputWidth}×${result.inputHeight} input -> ${state.imageWidth}×${state.imageHeight} preview. ${result.notice}`,
  );
  updatePredictionMeta(`${result.modelName} | ${result.processingMs} ms | ${result.checkpoint}`);

  elements.resetBtn.disabled = false;
  elements.predictBtn.disabled = false;
  setDefaultRoi();
}

elements.imageInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }

  state.fileName = file.name;
  state.originalDataUrl = await fileToDataUrl(file);
  state.bicubicDataUrl = "";
  state.srDataUrl = "";
  state.bicubicImage = null;
  state.srImage = null;
  state.roi = null;
  state.imageWidth = 0;
  state.imageHeight = 0;

  elements.predictBtn.disabled = false;
  elements.resetBtn.disabled = true;
  updatePredictionMeta("Upload ready");
  setStatus("Image loaded", `${file.name} is ready for local super-resolution.`);

  try {
    await handlePredict();
  } catch (error) {
    console.error(error);
    setStatus("Prediction failed", error.message || "Something went wrong while generating the preview.");
    updatePredictionMeta("Error");
  }
});

elements.predictBtn.addEventListener("click", async () => {
  try {
    await handlePredict();
  } catch (error) {
    console.error(error);
    setStatus("Prediction failed", error.message || "Something went wrong while generating the preview.");
    updatePredictionMeta("Error");
  }
});

elements.resetBtn.addEventListener("click", () => {
  setDefaultRoi();
});

elements.boxSize.addEventListener("input", () => {
  state.boxSize = Number(elements.boxSize.value);
  elements.boxSizeLabel.textContent = `${state.boxSize} px`;
  if (state.roi) {
    const image = getImageAtMode();
    if (image) {
      const size = Math.min(state.boxSize, image.naturalWidth, image.naturalHeight);
      let left = state.roi.left;
      let top = state.roi.top;
      left = clamp(left, 0, Math.max(0, image.naturalWidth - size));
      top = clamp(top, 0, Math.max(0, image.naturalHeight - size));
      state.roi = { left, top, size };
    }
    renderAll();
  }
});

elements.zoomFactor.addEventListener("change", () => {
  state.zoomFactor = Number(elements.zoomFactor.value);
  if (state.roi) {
    renderAll();
  }
});

elements.selectionMode.addEventListener("change", () => {
  state.selectionMode = elements.selectionMode.value;
  if (state.bicubicImage) {
    setDefaultRoi();
  }
});

for (const canvas of [elements.bicubicCanvas, elements.srCanvas]) {
  canvas.addEventListener("click", (event) => {
    if (!state.bicubicImage) return;
    const coords = pointerToImageCoordinates(event, canvas);
    if (!coords) return;

    const image = getImageAtMode();
    const size = Number(elements.boxSize.value);
    const roi = computeRoiFromCenter(
      coords.x,
      coords.y,
      size,
      image.naturalWidth,
      image.naturalHeight,
    );
    state.roi = roi;
    renderAll();
  });
}

async function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Unable to read the selected file."));
    reader.readAsDataURL(file);
  });
}

elements.boxSizeLabel.textContent = `${elements.boxSize.value} px`;
setStatus("Ready", "Upload an image to generate a local super-resolution preview.");
updatePredictionMeta("Waiting for input");