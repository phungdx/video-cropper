import { describe, expect, it } from 'vitest';
import { choosePersonForClick, matchDetectionToPrevious } from './tracking';

describe('choosePersonForClick', () => {
  it('returns the person box containing the click point', () => {
    const selection = choosePersonForClick(
      [
        { label: 'person', score: 0.93, box: { x: 10, y: 10, width: 80, height: 100 } },
        { label: 'person', score: 0.88, box: { x: 150, y: 20, width: 90, height: 110 } },
      ],
      { x: 35, y: 40 },
    );

    expect(selection?.box.x).toBe(10);
  });
});

describe('matchDetectionToPrevious', () => {
  it('does not hand the lock to a far away person when the selected person is gone', () => {
    const match = matchDetectionToPrevious(
      [
        { label: 'person', score: 0.96, box: { x: 620, y: 90, width: 110, height: 220 } },
      ],
      { x: 120, y: 80, width: 100, height: 200 },
    );

    expect(match).toBeNull();
  });

  it('keeps the lock on a nearby person', () => {
    const match = matchDetectionToPrevious(
      [
        { label: 'person', score: 0.9, box: { x: 126, y: 84, width: 100, height: 200 } },
      ],
      { x: 120, y: 80, width: 100, height: 200 },
    );

    expect(match?.box.x).toBe(126);
  });
});
