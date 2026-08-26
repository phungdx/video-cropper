export type Box = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type FrameSize = {
  width: number;
  height: number;
};

export type Point = {
  x: number;
  y: number;
};

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function boxCenter(box: Box): Point {
  return {
    x: box.x + box.width / 2,
    y: box.y + box.height / 2,
  };
}

export function pointInBox(point: Point, box: Box): boolean {
  return (
    point.x >= box.x &&
    point.x <= box.x + box.width &&
    point.y >= box.y &&
    point.y <= box.y + box.height
  );
}

export function iou(a: Box, b: Box): number {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);

  const intersectionWidth = Math.max(0, right - left);
  const intersectionHeight = Math.max(0, bottom - top);
  const intersectionArea = intersectionWidth * intersectionHeight;
  const unionArea = a.width * a.height + b.width * b.height - intersectionArea;

  return unionArea <= 0 ? 0 : intersectionArea / unionArea;
}

export function expandBox(box: Box, paddingRatio: number, bounds: FrameSize): Box {
  const padX = box.width * paddingRatio;
  const padY = box.height * paddingRatio;
  const left = clamp(box.x - padX, 0, bounds.width);
  const top = clamp(box.y - padY, 0, bounds.height);
  const right = clamp(box.x + box.width + padX, left, bounds.width);
  const bottom = clamp(box.y + box.height + padY, top, bounds.height);

  return {
    x: left,
    y: top,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top),
  };
}

export function projectBoxToRect(box: Box, source: Box, target: FrameSize): Box {
  return {
    x: ((box.x - source.x) / source.width) * target.width,
    y: ((box.y - source.y) / source.height) * target.height,
    width: (box.width / source.width) * target.width,
    height: (box.height / source.height) * target.height,
  };
}

/**
 * Slack added around the detected box. Detector boxes sit tight against the body and wobble a few
 * pixels between passes, so a little margin stops hair and hands from being shaved off.
 */
export const CROP_MARGIN = 0.04;
/** Guard rails on the derived aspect ratio, so one bad detection cannot pick a freak shape. */
export const MIN_CROP_ASPECT_RATIO = 0.2;
export const MAX_CROP_ASPECT_RATIO = 3;
/** Used before a person is selected, and if a box is too degenerate to measure. */
export const DEFAULT_CROP_ASPECT_RATIO = 9 / 16;

/**
 * The shape of the subject's own bounding box, which becomes the output shape so that cropping to
 * the box does not need to add margin on either axis.
 */
export function boxAspectRatio(box: Box, fallback = DEFAULT_CROP_ASPECT_RATIO): number {
  if (!Number.isFinite(box.width) || !Number.isFinite(box.height) || box.width <= 0 || box.height <= 0) {
    return fallback;
  }

  return clamp(box.width / box.height, MIN_CROP_ASPECT_RATIO, MAX_CROP_ASPECT_RATIO);
}

/**
 * Grows the requested size to the target aspect ratio around a fixed center, shrinks it if it no
 * longer fits, then slides it inside the frame.
 */
function fitRectToFrame(
  center: Point,
  width: number,
  height: number,
  frame: FrameSize,
  aspectRatio: number,
): Box {
  let cropWidth = width;
  let cropHeight = height;

  if (cropWidth / cropHeight < aspectRatio) {
    cropWidth = cropHeight * aspectRatio;
  } else {
    cropHeight = cropWidth / aspectRatio;
  }

  if (cropWidth > frame.width) {
    cropWidth = frame.width;
    cropHeight = cropWidth / aspectRatio;
  }

  if (cropHeight > frame.height) {
    cropHeight = frame.height;
    cropWidth = cropHeight * aspectRatio;
  }

  cropWidth = Math.min(cropWidth, frame.width);
  cropHeight = Math.min(cropHeight, frame.height);

  return {
    x: clamp(center.x - cropWidth / 2, 0, Math.max(0, frame.width - cropWidth)),
    y: clamp(center.y - cropHeight / 2, 0, Math.max(0, frame.height - cropHeight)),
    width: cropWidth,
    height: cropHeight,
  };
}

/** The neutral framing used before a person is picked: a centered slice of the whole frame. */
export function centerCropRect(frame: FrameSize, aspectRatio: number): Box {
  return fitRectToFrame(
    { x: frame.width / 2, y: frame.height / 2 },
    frame.width,
    frame.height,
    frame,
    aspectRatio,
  );
}

/**
 * Crops to the subject's bounding box: the box plus a small margin, grown on one axis only if the
 * output shape demands it. When the aspect ratio came from the same subject, as it does for an
 * export, that growth is negligible and the crop is the bounding box.
 */
export function computeCropRect(
  box: Box,
  frame: FrameSize,
  aspectRatio: number,
  margin = CROP_MARGIN,
): Box {
  const grow = 1 + Math.max(0, margin) * 2;

  return fitRectToFrame(
    boxCenter(box),
    box.width * grow,
    box.height * grow,
    frame,
    aspectRatio,
  );
}
