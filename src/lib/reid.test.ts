import { describe, expect, it } from 'vitest';
import { compareAppearanceSignatures, selectTargetDetection } from './reid';

describe('compareAppearanceSignatures', () => {
  it('scores similar signatures higher than different ones', () => {
    const same = compareAppearanceSignatures([1, 2, 3, 4], [1, 2, 3, 4]);
    const different = compareAppearanceSignatures([1, 2, 3, 4], [9, 8, 7, 6]);

    expect(same).toBeGreaterThan(different);
  });
});

describe('selectTargetDetection', () => {
  it('selects the same person by appearance even when another person is closer', () => {
    const match = selectTargetDetection(
      [
        { label: 'person', score: 0.96, box: { x: 550, y: 90, width: 110, height: 220 } },
        { label: 'person', score: 0.88, box: { x: 130, y: 100, width: 115, height: 225 } },
      ],
      {
        id: 'person-1',
        lastBox: { x: 120, y: 90, width: 110, height: 220 },
        signature: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      },
      [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      ],
      { width: 1280, height: 720 },
    );

    expect(match?.detection.box.x).toBe(550);
  });

  it('keeps the locked target when a closer bystander has a weaker appearance match', () => {
    const match = selectTargetDetection(
      [
        {
          label: 'person',
          score: 0.94,
          box: { x: 1120, y: 80, width: 50, height: 100 },
        },
        {
          label: 'person',
          score: 0.89,
          box: { x: 126, y: 84, width: 100, height: 200 },
        },
      ],
      {
        id: 'person-1',
        lastBox: { x: 120, y: 80, width: 100, height: 200 },
        signature: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      },
      [
        [0.8, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.6, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      ],
      { width: 1280, height: 720 },
    );

    expect(match?.detection.box.x).toBe(1120);
  });

  it('returns null when the saved person is not present', () => {
    const match = selectTargetDetection(
      [{ label: 'person', score: 0.95, box: { x: 520, y: 90, width: 110, height: 220 } }],
      {
        id: 'person-1',
        lastBox: { x: 120, y: 90, width: 110, height: 220 },
        signature: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      },
      [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
      { width: 1280, height: 720 },
    );

    expect(match).toBeNull();
  });
});
