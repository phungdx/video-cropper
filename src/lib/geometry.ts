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

export function computeCropRect(
  box: Box,
  frame: FrameSize,
  aspectRatio: number,
  padding = 0.45,
): Box {
  const center = boxCenter(box);
  const paddedWidth = box.width * (1 + padding * 2);
  const paddedHeight = box.height * (1 + padding * 2);

  let cropWidth = paddedWidth;
  let cropHeight = paddedHeight;

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

  let x = center.x - cropWidth / 2;
  let y = center.y - cropHeight / 2;

  x = clamp(x, 0, Math.max(0, frame.width - cropWidth));
  y = clamp(y, 0, Math.max(0, frame.height - cropHeight));

  return {
    x,
    y,
    width: cropWidth,
    height: cropHeight,
  };
}
