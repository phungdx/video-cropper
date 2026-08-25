import { describe, expect, it } from 'vitest';
import { computeCropRect } from './geometry';

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
});
