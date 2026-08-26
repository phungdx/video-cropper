import { describe, expect, it } from 'vitest';
import { computeAppearanceSignatureFromPixels } from './reid';
import {
  advanceTrack,
  associateTarget,
  createTrack,
  predictBox,
  LOST_AFTER_MISSES,
  type TargetTrack,
  type TrackCandidate,
} from './tracker';
import type { Detection } from './tracking';

const FRAME = { width: 1280, height: 720 };
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

const RED_SHIRT = signatureForColor([200, 40, 40]);
const BLUE_SHIRT = signatureForColor([40, 40, 200]);

function person(x: number, y: number, score = 0.9): Detection {
  return { label: 'person', score, box: { x, y, width: 80, height: 180 } };
}

function candidate(detection: Detection, signature: number[]): TrackCandidate {
  return { detection, signature };
}

function lockedTrack(): TargetTrack {
  return createTrack('person-1', person(100, 100), RED_SHIRT);
}

describe('associateTarget', () => {
  it('follows the selected person as they move', () => {
    const association = associateTarget(
      lockedTrack(),
      [candidate(person(118, 104), RED_SHIRT)],
      { dt: 0.1, frame: FRAME },
    );

    expect(association.mode).toBe('gated');
    expect(association.index).toBe(0);
  });

  it('does not hand the lock to a bystander who walks through the target position', () => {
    const association = associateTarget(
      lockedTrack(),
      [candidate(person(104, 102), BLUE_SHIRT)],
      { dt: 0.1, frame: FRAME },
    );

    expect(association.mode).toBe('none');
    expect(association.index).toBeNull();
  });

  it('prefers the matching person when a bystander overlaps the target', () => {
    const association = associateTarget(
      lockedTrack(),
      [candidate(person(102, 100, 0.97), BLUE_SHIRT), candidate(person(126, 106), RED_SHIRT)],
      { dt: 0.1, frame: FRAME },
    );

    expect(association.index).toBe(1);
  });

  it('ignores a look-alike on the far side of the frame while the lock is healthy', () => {
    const association = associateTarget(
      lockedTrack(),
      [candidate(person(900, 120), RED_SHIRT)],
      { dt: 0.1, frame: FRAME },
    );

    expect(association.mode).toBe('none');
  });

  it('re-acquires the target anywhere in the frame once the track is lost', () => {
    const lost: TargetTrack = {
      ...lockedTrack(),
      status: 'lost',
      misses: LOST_AFTER_MISSES,
      secondsSinceMatch: 1.5,
    };

    const association = associateTarget(
      lost,
      [candidate(person(880, 130), BLUE_SHIRT), candidate(person(940, 120), RED_SHIRT)],
      { dt: 0.1, frame: FRAME },
    );

    expect(association.mode).toBe('reacquired');
    expect(association.index).toBe(1);
  });

  it('stays unlocked while lost if the only people on screen look wrong', () => {
    const lost: TargetTrack = {
      ...lockedTrack(),
      status: 'lost',
      misses: LOST_AFTER_MISSES,
      secondsSinceMatch: 2,
    };

    const association = associateTarget(
      lost,
      [candidate(person(880, 130), BLUE_SHIRT), candidate(person(500, 200), BLUE_SHIRT)],
      { dt: 0.1, frame: FRAME },
    );

    expect(association.mode).toBe('none');
  });

  it('reports a miss when nobody is detected', () => {
    const association = associateTarget(lockedTrack(), [], { dt: 0.1, frame: FRAME });

    expect(association.mode).toBe('none');
  });
});

describe('advanceTrack', () => {
  it('keeps the box moving through a short occlusion instead of freezing', () => {
    const moving: TargetTrack = { ...lockedTrack(), velocity: { x: 120, y: 0 } };

    const next = advanceTrack(moving, [], { index: null, score: 0, appearance: 0, mode: 'none' }, {
      dt: 0.2,
      frame: FRAME,
    });

    expect(next.box.x).toBeGreaterThan(moving.box.x);
    expect(next.status).toBe('coasting');
  });

  it('marks the track lost after repeated misses', () => {
    let track: TargetTrack = { ...lockedTrack(), velocity: { x: 60, y: 0 } };

    for (let pass = 0; pass < LOST_AFTER_MISSES; pass += 1) {
      track = advanceTrack(track, [], { index: null, score: 0, appearance: 0, mode: 'none' }, {
        dt: 0.1,
        frame: FRAME,
      });
    }

    expect(track.status).toBe('lost');
  });

  it('eases towards the matched detection and learns its motion', () => {
    const track = lockedTrack();
    const candidates = [candidate(person(140, 100), RED_SHIRT)];
    const association = associateTarget(track, candidates, { dt: 0.1, frame: FRAME });
    const next = advanceTrack(track, candidates, association, { dt: 0.1, frame: FRAME });

    expect(next.status).toBe('tracking');
    expect(next.box.x).toBeGreaterThan(track.box.x);
    expect(next.box.x).toBeLessThan(140);
    expect(next.velocity.x).toBeGreaterThan(0);
    expect(next.misses).toBe(0);
  });

  it('snaps straight onto the target after a re-acquisition', () => {
    const lost: TargetTrack = {
      ...lockedTrack(),
      status: 'lost',
      misses: LOST_AFTER_MISSES,
      secondsSinceMatch: 1.5,
    };
    const candidates = [candidate(person(940, 120), RED_SHIRT)];
    const association = associateTarget(lost, candidates, { dt: 0.1, frame: FRAME });
    const next = advanceTrack(lost, candidates, association, { dt: 0.1, frame: FRAME });

    expect(next.box.x).toBe(940);
    expect(next.status).toBe('tracking');
    expect(next.velocity).toEqual({ x: 0, y: 0 });
  });
});

describe('predictBox', () => {
  it('keeps the predicted box inside the frame', () => {
    const predicted = predictBox(
      { x: 1200, y: 600, width: 80, height: 180 },
      { x: 500, y: 500 },
      1,
      FRAME,
    );

    expect(predicted.x).toBe(FRAME.width - 80);
    expect(predicted.y).toBe(FRAME.height - 180);
  });
});

describe('following one person to the end of a clip', () => {
  it('keeps the lock through a crossing bystander and a full occlusion', () => {
    const PASSES = 30;
    const DT = 0.1;
    const OCCLUSION = { from: 15, to: 20 };
    const targetX = (pass: number) => 100 + pass * 24;
    const bystanderX = (pass: number) => 700 - pass * 30;

    let track = createTrack('person 1', person(targetX(0), 100), RED_SHIRT);
    let swaps = 0;

    for (let pass = 1; pass < PASSES; pass += 1) {
      const occluded = pass >= OCCLUSION.from && pass <= OCCLUSION.to;
      const candidates: TrackCandidate[] = [
        candidate(person(bystanderX(pass), 100, 0.95), BLUE_SHIRT),
      ];

      if (!occluded) {
        candidates.push(candidate(person(targetX(pass), 100), RED_SHIRT));
      }

      const association = associateTarget(track, candidates, { dt: DT, frame: FRAME });

      if (association.index === 0) {
        swaps += 1;
      }

      track = advanceTrack(track, candidates, association, { dt: DT, frame: FRAME });
    }

    const finalCenter = track.box.x + track.box.width / 2;
    const trueCenter = targetX(PASSES - 1) + 40;

    expect(swaps).toBe(0);
    expect(track.status).toBe('tracking');
    expect(Math.abs(finalCenter - trueCenter)).toBeLessThan(80);
  });
});
