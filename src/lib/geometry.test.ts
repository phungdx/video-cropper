import { describe, expect, it } from 'vitest';
import {
  boxAspectRatio,
  boxCenter,
  centerCropRect,
  computeCropRect,
  expandBox,
  projectBoxToRect,
} from './geometry';

describe('computeCropRect', () => {
  it('centers a crop box around the tracked box and clamps to frame bounds', () => {
    const result = computeCropRect(
      { x: 300, y: 200, width: 120, height: 240 },
      { width: 1280, height: 720 },
      9 / 16,
    );

    expect(result.width / result.height).toBeCloseTo(9 / 16, 2);
    expect(result.x).toBeGreaterThanOrEqual(0);
    expect(result.y).toBeGreaterThanOrEqual(0);
  });

  it('crops to the subject bounding box when the ratio came from that subject', () => {
    const frame = { width: 1920, height: 1080 };
    const person = { x: 860, y: 300, width: 200, height: 650 };
    const crop = computeCropRect(person, frame, boxAspectRatio(person));

    // Only the 4% margin separates the crop from the box itself, on both axes.
    expect(crop.width).toBeCloseTo(person.width * 1.08, 5);
    expect(crop.height).toBeCloseTo(person.height * 1.08, 5);
    expect(boxCenter(crop).x).toBeCloseTo(boxCenter(person).x, 5);
    expect(boxCenter(crop).y).toBeCloseTo(boxCenter(person).y, 5);
  });

  it('holds the whole subject in shot when the ratio does not match them', () => {
    const frame = { width: 1920, height: 1080 };
    const person = { x: 860, y: 300, width: 200, height: 650 };
    const crop = computeCropRect(person, frame, 1);

    // The short axis grows, the long one is never cut.
    expect(crop.width).toBeGreaterThanOrEqual(person.width);
    expect(crop.height).toBeGreaterThanOrEqual(person.height);
    expect(crop.width / crop.height).toBeCloseTo(1, 5);
  });

  it('derives a portrait ratio from a standing subject and guards degenerate boxes', () => {
    expect(boxAspectRatio({ x: 0, y: 0, width: 200, height: 650 })).toBeCloseTo(200 / 650, 5);
    expect(boxAspectRatio({ x: 0, y: 0, width: 0, height: 0 })).toBe(9 / 16);
    expect(boxAspectRatio({ x: 0, y: 0, width: 1, height: 400 })).toBe(0.2);
  });

  it('gives a centered slice of the frame when nothing is tracked', () => {
    const crop = centerCropRect({ width: 1920, height: 1080 }, 9 / 16);

    expect(crop.height).toBe(1080);
    expect(crop.width).toBeCloseTo(607.5, 1);
    expect(crop.x).toBeCloseTo((1920 - 607.5) / 2, 1);
  });

  it('expands and projects the tracked box into preview space', () => {
    const focus = expandBox(
      { x: 300, y: 200, width: 120, height: 240 },
      0.12,
      { width: 1280, height: 720 },
    );

    const projected = projectBoxToRect(
      focus,
      { x: 200, y: 100, width: 320, height: 400 },
      { width: 200, height: 250 },
    );

    expect(focus.x).toBeLessThan(300);
    expect(projected.x).toBeGreaterThanOrEqual(0);
    expect(projected.y).toBeGreaterThanOrEqual(0);
    expect(projected.width).toBeGreaterThan(0);
    expect(projected.height).toBeGreaterThan(0);
  });
});
