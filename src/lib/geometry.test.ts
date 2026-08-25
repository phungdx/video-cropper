import { describe, expect, it } from 'vitest';
import { computeCropRect, expandBox, projectBoxToRect } from './geometry';

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
