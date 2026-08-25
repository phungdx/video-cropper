import { clamp, type Box, type FrameSize } from './geometry';
import type { Detection } from './tracking';

export type AppearanceSignature = number[];

export type TargetProfile = {
  id: string;
  lastBox: Box;
  signature: AppearanceSignature | null;
};

export type TargetMatch = {
  detection: Detection;
  score: number;
};

export function computeAppearanceSignatureFromPixels(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
): AppearanceSignature {
  const signature: number[] = [];

  for (let cellY = 0; cellY < 4; cellY += 1) {
    for (let cellX = 0; cellX < 4; cellX += 1) {
      let totalR = 0;
      let totalG = 0;
      let totalB = 0;
      let totalWeight = 0;

      const startX = Math.floor((cellX * width) / 4);
      const endX = Math.max(startX + 1, Math.floor(((cellX + 1) * width) / 4));
      const startY = Math.floor((cellY * height) / 4);
      const endY = Math.max(startY + 1, Math.floor(((cellY + 1) * height) / 4));

      for (let y = startY; y < endY; y += 1) {
        for (let x = startX; x < endX; x += 1) {
          const index = (y * width + x) * 4;
          const alpha = pixels[index + 3] / 255;

          totalR += pixels[index] * alpha;
          totalG += pixels[index + 1] * alpha;
          totalB += pixels[index + 2] * alpha;
          totalWeight += alpha;
        }
      }

      if (totalWeight === 0) {
        signature.push(0, 0, 0);
      } else {
        signature.push(totalR / totalWeight, totalG / totalWeight, totalB / totalWeight);
      }
    }
  }

  return signature;
}

export function compareAppearanceSignatures(
  a: AppearanceSignature | null,
  b: AppearanceSignature | null,
): number {
  if (!a || !b || a.length !== b.length || a.length === 0) {
    return 0;
  }

  let dot = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let index = 0; index < a.length; index += 1) {
    dot += a[index] * b[index];
    magnitudeA += a[index] * a[index];
    magnitudeB += b[index] * b[index];
  }

  if (magnitudeA === 0 || magnitudeB === 0) {
    return 0;
  }

  return dot / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB));
}

export function captureAppearanceSignature(
  source: CanvasImageSource,
  box: Box,
  frame: FrameSize,
): AppearanceSignature | null {
  if (typeof document === 'undefined') {
    return null;
  }

  const sampleSize = 8;
  const canvas = document.createElement('canvas');
  canvas.width = sampleSize;
  canvas.height = sampleSize;

  const context = canvas.getContext('2d');
  if (!context) {
    return null;
  }

  const clampedBox = {
    x: clamp(box.x, 0, frame.width),
    y: clamp(box.y, 0, frame.height),
    width: clamp(box.width, 1, frame.width),
    height: clamp(box.height, 1, frame.height),
  };

  context.drawImage(
    source,
    clampedBox.x,
    clampedBox.y,
    clampedBox.width,
    clampedBox.height,
    0,
    0,
    sampleSize,
    sampleSize,
  );

  const pixels = context.getImageData(0, 0, sampleSize, sampleSize).data;
  return computeAppearanceSignatureFromPixels(pixels, sampleSize, sampleSize);
}

export function selectTargetDetection(
  detections: Detection[],
  target: TargetProfile,
  candidateSignatures: Array<AppearanceSignature | null>,
  frame: FrameSize,
): TargetMatch | null {
  const people = detections.filter((d) => d.label === 'person');

  if (people.length === 0) {
    return null;
  }

  const scored = people
    .map((detection, index) => {
      const signature = candidateSignatures[index];
      const appearance = compareAppearanceSignatures(target.signature, signature);
      const sizeSimilarity = scoreSizeSimilarity(target.lastBox, detection.box);
      const spatialProximity = scoreSpatialProximity(target.lastBox, detection.box);
      const score = appearance * 0.72 + sizeSimilarity * 0.18 + spatialProximity * 0.10;

      return { detection, score };
    })
    .sort((a, b) => b.score - a.score);

  return scored[0].score >= 0.55 ? scored[0] : null;
}

function scoreSpatialProximity(previous: Box, current: Box): number {
  const previousCenterX = previous.x + previous.width / 2;
  const previousCenterY = previous.y + previous.height / 2;
  const currentCenterX = current.x + current.width / 2;
  const currentCenterY = current.y + current.height / 2;
  const distance = Math.hypot(currentCenterX - previousCenterX, currentCenterY - previousCenterY);
  const maxDistance = Math.max(previous.width, previous.height) * 5;
  return clamp(1 - distance / maxDistance, 0, 1);
}

function scoreSizeSimilarity(previous: Box, current: Box): number {
  const widthDiff = Math.abs(previous.width - current.width) / Math.max(previous.width, current.width);
  const heightDiff = Math.abs(previous.height - current.height) / Math.max(previous.height, current.height);
  const averageDiff = (widthDiff + heightDiff) / 2;
  return clamp(1 - averageDiff, 0, 1);
}
