import { describe, expect, it } from 'vitest';
import {
  blendSignatures,
  compareAppearanceSignatures,
  computeAppearanceSignatureFromPixels,
} from './reid';

const GRID_WIDTH = 3;
const GRID_HEIGHT = 5;

function signatureForColor(color: [number, number, number]): number[] {
  const pixels = new Uint8ClampedArray(GRID_WIDTH * GRID_HEIGHT * 4);

  for (let index = 0; index < GRID_WIDTH * GRID_HEIGHT; index += 1) {
    const offset = index * 4;
    pixels[offset] = color[0];
    pixels[offset + 1] = color[1];
    pixels[offset + 2] = color[2];
    pixels[offset + 3] = 255;
  }

  return computeAppearanceSignatureFromPixels(pixels, GRID_WIDTH, GRID_HEIGHT);
}

describe('compareAppearanceSignatures', () => {
  it('scores similar signatures higher than different ones', () => {
    const same = compareAppearanceSignatures([1, 2, 3, 4], [1, 2, 3, 4]);
    const different = compareAppearanceSignatures([1, 2, 3, 4], [9, 8, 7, 6]);

    expect(same).toBeGreaterThan(different);
  });

  it('stays high for the same colours under different brightness', () => {
    const bright = signatureForColor([200, 40, 40]);
    const dim = signatureForColor([100, 20, 20]);

    expect(compareAppearanceSignatures(bright, dim)).toBeGreaterThan(0.9);
  });

  it('separates people wearing different colours', () => {
    const red = signatureForColor([200, 40, 40]);
    const blue = signatureForColor([40, 40, 200]);

    expect(compareAppearanceSignatures(red, blue)).toBeLessThan(0.6);
  });

  it('returns zero when one signature is missing', () => {
    expect(compareAppearanceSignatures(null, [1, 2, 3])).toBe(0);
  });
});

describe('blendSignatures', () => {
  it('moves the reference partway towards the newest sample', () => {
    const blended = blendSignatures([0, 0, 0, 0], [1, 1, 1, 1], 0.25);

    expect(blended).toEqual([0.25, 0.25, 0.25, 0.25]);
  });

  it('keeps the existing reference when there is no new sample', () => {
    expect(blendSignatures([1, 2, 3], null, 0.5)).toEqual([1, 2, 3]);
  });
});
