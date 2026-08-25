import { describe, expect, it } from 'vitest';
import { choosePersonForClick } from './tracking';

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
