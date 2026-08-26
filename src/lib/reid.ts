import { clamp, type Box, type FrameSize } from './geometry';

export type AppearanceSignature = number[];

const GRID_COLUMNS = 3;
const GRID_ROWS = 5;
const SAMPLE_WIDTH = 12;
const SAMPLE_HEIGHT = 20;
const HORIZONTAL_INSET = 0.12;
const LUMA_WEIGHT = 0.5;

/**
 * Builds a small, brightness-tolerant appearance descriptor.
 *
 * Each grid cell contributes normalized chroma (r, g, b divided by their sum) plus a
 * down-weighted luma term. Normalizing per cell means the descriptor mostly encodes
 * "what colours is this person wearing, and where", so it survives exposure changes
 * while still separating two people in different clothing.
 */
export function computeAppearanceSignatureFromPixels(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
): AppearanceSignature {
  const signature: number[] = [];

  for (let cellY = 0; cellY < GRID_ROWS; cellY += 1) {
    for (let cellX = 0; cellX < GRID_COLUMNS; cellX += 1) {
      let totalR = 0;
      let totalG = 0;
      let totalB = 0;
      let totalWeight = 0;

      const startX = Math.floor((cellX * width) / GRID_COLUMNS);
      const endX = Math.max(startX + 1, Math.floor(((cellX + 1) * width) / GRID_COLUMNS));
      const startY = Math.floor((cellY * height) / GRID_ROWS);
      const endY = Math.max(startY + 1, Math.floor(((cellY + 1) * height) / GRID_ROWS));

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
        signature.push(0, 0, 0, 0);
        continue;
      }

      const r = totalR / totalWeight;
      const g = totalG / totalWeight;
      const b = totalB / totalWeight;
      const energy = r + g + b;
      const luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255;

      if (energy <= 0) {
        signature.push(0, 0, 0, luma * LUMA_WEIGHT);
      } else {
        signature.push(r / energy, g / energy, b / energy, luma * LUMA_WEIGHT);
      }
    }
  }

  return signature;
}

/**
 * Similarity in [0, 1] for two non-negative descriptors.
 *
 * This is 1 minus the Bray-Curtis dissimilarity. Unlike cosine similarity it stays
 * discriminative for all-positive vectors, where nearly every pair looks alike.
 */
export function compareAppearanceSignatures(
  a: AppearanceSignature | null,
  b: AppearanceSignature | null,
): number {
  if (!a || !b || a.length !== b.length || a.length === 0) {
    return 0;
  }

  let difference = 0;
  let magnitude = 0;

  for (let index = 0; index < a.length; index += 1) {
    difference += Math.abs(a[index] - b[index]);
    magnitude += Math.abs(a[index]) + Math.abs(b[index]);
  }

  if (magnitude === 0) {
    return 0;
  }

  return clamp(1 - difference / magnitude, 0, 1);
}

/** Moves `base` a fraction of the way towards `next` so the reference appearance can adapt. */
export function blendSignatures(
  base: AppearanceSignature | null,
  next: AppearanceSignature | null,
  alpha: number,
): AppearanceSignature | null {
  if (!next) {
    return base;
  }

  if (!base || base.length !== next.length) {
    return next;
  }

  const ratio = clamp(alpha, 0, 1);
  return base.map((value, index) => value + (next[index] - value) * ratio);
}

export function captureAppearanceSignature(
  source: CanvasImageSource,
  box: Box,
  frame: FrameSize,
): AppearanceSignature | null {
  if (typeof document === 'undefined') {
    return null;
  }

  const canvas = document.createElement('canvas');
  canvas.width = SAMPLE_WIDTH;
  canvas.height = SAMPLE_HEIGHT;

  const context = canvas.getContext('2d');
  if (!context) {
    return null;
  }

  const region = insetSampleRegion(box, frame);
  if (region.width < 1 || region.height < 1) {
    return null;
  }

  try {
    context.drawImage(
      source,
      region.x,
      region.y,
      region.width,
      region.height,
      0,
      0,
      SAMPLE_WIDTH,
      SAMPLE_HEIGHT,
    );
  } catch {
    return null;
  }

  const pixels = context.getImageData(0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT).data;
  return computeAppearanceSignatureFromPixels(pixels, SAMPLE_WIDTH, SAMPLE_HEIGHT);
}

/**
 * Trims the sides of a detection box before sampling so background pixels around the
 * person contribute less to the descriptor, then clips the result to the frame.
 */
function insetSampleRegion(box: Box, frame: FrameSize): Box {
  const inset = box.width * HORIZONTAL_INSET;
  const left = clamp(box.x + inset, 0, frame.width);
  const right = clamp(box.x + box.width - inset, left, frame.width);
  const top = clamp(box.y, 0, frame.height);
  const bottom = clamp(box.y + box.height, top, frame.height);

  return {
    x: left,
    y: top,
    width: right - left,
    height: bottom - top,
  };
}
