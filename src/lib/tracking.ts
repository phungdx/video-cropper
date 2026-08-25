import { boxCenter, clamp, iou, pointInBox, type Box, type Point } from './geometry';

export type Detection = {
  label: string;
  score: number;
  box: Box;
};

export function choosePersonForClick(
  detections: Detection[],
  point: Point,
): Detection | null {
  const people = detections.filter((d) => d.label === 'person');

  const containing = people
    .filter((d) => pointInBox(point, d.box))
    .sort((a, b) => b.score - a.score);

  if (containing.length > 0) {
    return containing[0];
  }

  if (people.length === 0) {
    return null;
  }

  return people
    .map((d) => {
      const center = boxCenter(d.box);
      const distance = Math.hypot(center.x - point.x, center.y - point.y);
      return { detection: d, distance };
    })
    .sort((a, b) => a.distance - b.distance || b.detection.score - a.detection.score)[0]
    .detection;
}

export function matchDetectionToPrevious(
  detections: Detection[],
  previous: Box,
  minScore = 0.1,
): Detection | null {
  const people = detections.filter((d) => d.label === 'person' && d.score >= minScore);

  if (people.length === 0) {
    return null;
  }

  const scored = people
    .map((d) => {
      const overlap = iou(previous, d.box);
      const center = boxCenter(d.box);
      const prevCenter = boxCenter(previous);
      const distance = Math.hypot(center.x - prevCenter.x, center.y - prevCenter.y);
      const maxSearchDistance = Math.max(previous.width, previous.height) * 4;
      const proximity = clamp(1 - distance / maxSearchDistance, 0, 1);
      const score = overlap * 0.65 + proximity * 0.25 + d.score * 0.1;

      return { detection: d, score };
    })
    .sort((a, b) => b.score - a.score);

  return scored[0].score >= 0.3 ? scored[0].detection : null;
}

export function smoothBox(previous: Box, next: Box, alpha = 0.35): Box {
  return {
    x: previous.x + (next.x - previous.x) * alpha,
    y: previous.y + (next.y - previous.y) * alpha,
    width: previous.width + (next.width - previous.width) * alpha,
    height: previous.height + (next.height - previous.height) * alpha,
  };
}
