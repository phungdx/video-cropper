import { boxCenter, clamp, iou, type Box, type FrameSize, type Point } from './geometry';
import {
  blendSignatures,
  compareAppearanceSignatures,
  type AppearanceSignature,
} from './reid';
import { smoothBox, type Detection } from './tracking';

/** Consecutive detection passes without a match before the track opens up a global search. */
export const LOST_AFTER_MISSES = 6;

/** Appearance floor for staying locked on a candidate inside the motion gate. */
const MIN_GATED_APPEARANCE = 0.6;
/** Combined score a gated candidate must reach to take over the lock. */
const MIN_GATED_SCORE = 0.55;
/** Appearance floor for grabbing the target again anywhere in the frame. */
const MIN_REACQUIRE_APPEARANCE = 0.62;
/** How far ahead of the runner-up a re-acquisition candidate must be. */
const MIN_REACQUIRE_MARGIN = 0.06;
/** Relaxed floor used when there is only one person on screen, so there is nobody to confuse. */
const MIN_SOLO_REACQUIRE_APPEARANCE = 0.45;

const APPEARANCE_WEIGHT = 0.5;
const MOTION_WEIGHT = 0.35;
const SIZE_WEIGHT = 0.15;

const ANCHOR_WEIGHT = 0.55;
const LIVE_WEIGHT = 0.45;

const TRACKING_SMOOTHING = 0.55;
const SIGNATURE_ADAPTATION = 0.18;
const VELOCITY_SMOOTHING = 0.5;
const COAST_VELOCITY_DECAY = 0.55;
const MAX_GATE_EXPANSION_SECONDS = 2.5;

export type TrackStatus = 'tracking' | 'coasting' | 'lost';

export type TargetTrack = {
  id: string;
  /** Smoothed box in video pixel space. */
  box: Box;
  /** Box-centre velocity in video pixels per second of playback. */
  velocity: Point;
  /** Appearance captured when the user picked this person; never drifts. */
  anchorSignature: AppearanceSignature | null;
  /** Slowly adapting appearance so lighting and pose changes do not break the lock. */
  liveSignature: AppearanceSignature | null;
  status: TrackStatus;
  misses: number;
  confidence: number;
  secondsSinceMatch: number;
};

export type TrackCandidate = {
  detection: Detection;
  signature: AppearanceSignature | null;
};

export type AssociationMode = 'gated' | 'reacquired' | 'none';

export type Association = {
  index: number | null;
  score: number;
  appearance: number;
  mode: AssociationMode;
};

export type TrackStepOptions = {
  /** Seconds of playback since the previous tracker update. */
  dt: number;
  frame: FrameSize;
};

export function createTrack(
  id: string,
  detection: Detection,
  signature: AppearanceSignature | null,
): TargetTrack {
  return {
    id,
    box: detection.box,
    velocity: { x: 0, y: 0 },
    anchorSignature: signature,
    liveSignature: signature,
    status: 'tracking',
    misses: 0,
    confidence: detection.score,
    secondsSinceMatch: 0,
  };
}

/** Advances a box along its velocity, keeping it inside the frame. */
export function predictBox(box: Box, velocity: Point, dt: number, frame: FrameSize): Box {
  const step = clamp(dt, 0, 1);

  return {
    x: clamp(box.x + velocity.x * step, 0, Math.max(0, frame.width - box.width)),
    y: clamp(box.y + velocity.y * step, 0, Math.max(0, frame.height - box.height)),
    width: box.width,
    height: box.height,
  };
}

export function appearanceScore(
  track: TargetTrack,
  signature: AppearanceSignature | null,
): number {
  const anchor = compareAppearanceSignatures(track.anchorSignature, signature);
  const live = compareAppearanceSignatures(track.liveSignature ?? track.anchorSignature, signature);

  return anchor * ANCHOR_WEIGHT + live * LIVE_WEIGHT;
}

/**
 * Picks the detection that belongs to the tracked person.
 *
 * While the lock is healthy the search is gated to where the person can plausibly be, so a
 * passer-by cannot steal the box. The gate widens the longer the target has been missing, and
 * once the track is `lost` the whole frame is searched by appearance alone.
 */
export function associateTarget(
  track: TargetTrack,
  candidates: TrackCandidate[],
  options: TrackStepOptions,
): Association {
  const miss: Association = { index: null, score: 0, appearance: 0, mode: 'none' };
  const people = candidates
    .map((candidate, index) => ({ candidate, index }))
    .filter((entry) => entry.candidate.detection.label === 'person');

  if (people.length === 0) {
    return miss;
  }

  const predicted = predictBox(track.box, track.velocity, options.dt, options.frame);
  const elapsed = track.secondsSinceMatch + Math.max(0, options.dt);
  const reach =
    Math.max(predicted.width, predicted.height) *
    (1.1 + Math.min(elapsed, MAX_GATE_EXPANSION_SECONDS) * 1.6);
  const predictedCenter = boxCenter(predicted);

  const scored = people.map(({ candidate, index }) => {
    const box = candidate.detection.box;
    const center = boxCenter(box);
    const distance = Math.hypot(center.x - predictedCenter.x, center.y - predictedCenter.y);
    const proximity = reach <= 0 ? 0 : clamp(1 - distance / reach, 0, 1);
    const motion = Math.max(iou(predicted, box), proximity);
    const appearance = appearanceScore(track, candidate.signature);
    const size = sizeSimilarity(predicted, box);
    const score = appearance * APPEARANCE_WEIGHT + motion * MOTION_WEIGHT + size * SIZE_WEIGHT;

    return { index, appearance, motion, score };
  });

  const gated = scored
    .filter((entry) => entry.motion > 0 && entry.appearance >= MIN_GATED_APPEARANCE)
    .sort((a, b) => b.score - a.score);

  if (gated.length > 0 && gated[0].score >= MIN_GATED_SCORE) {
    return {
      index: gated[0].index,
      score: gated[0].score,
      appearance: gated[0].appearance,
      mode: 'gated',
    };
  }

  if (track.status !== 'lost') {
    return miss;
  }

  const byAppearance = [...scored].sort((a, b) => b.appearance - a.appearance);
  const best = byAppearance[0];
  const runnerUp = byAppearance[1];
  const soloRecovery =
    byAppearance.length === 1 && best.appearance >= MIN_SOLO_REACQUIRE_APPEARANCE;
  const confidentRecovery =
    best.appearance >= MIN_REACQUIRE_APPEARANCE &&
    (!runnerUp || best.appearance - runnerUp.appearance >= MIN_REACQUIRE_MARGIN);

  if (soloRecovery || confidentRecovery) {
    return {
      index: best.index,
      score: best.appearance,
      appearance: best.appearance,
      mode: 'reacquired',
    };
  }

  return miss;
}

/**
 * Produces the next track state. On a match the box eases towards the detection; on a miss the
 * box keeps coasting along the last known velocity instead of freezing or jumping.
 */
export function advanceTrack(
  track: TargetTrack,
  candidates: TrackCandidate[],
  association: Association,
  options: TrackStepOptions,
): TargetTrack {
  const dt = Math.max(0, options.dt);

  if (association.index === null) {
    const velocity = {
      x: track.velocity.x * COAST_VELOCITY_DECAY,
      y: track.velocity.y * COAST_VELOCITY_DECAY,
    };
    const misses = track.misses + 1;

    return {
      ...track,
      box: predictBox(track.box, velocity, dt, options.frame),
      velocity,
      misses,
      status: misses >= LOST_AFTER_MISSES ? 'lost' : 'coasting',
      confidence: track.confidence * 0.6,
      secondsSinceMatch: track.secondsSinceMatch + dt,
    };
  }

  const candidate = candidates[association.index];
  const reacquired = association.mode === 'reacquired';
  const box = reacquired
    ? candidate.detection.box
    : smoothBox(track.box, candidate.detection.box, TRACKING_SMOOTHING);
  const previousCenter = boxCenter(track.box);
  const nextCenter = boxCenter(box);
  const measured =
    dt > 0
      ? { x: (nextCenter.x - previousCenter.x) / dt, y: (nextCenter.y - previousCenter.y) / dt }
      : { x: 0, y: 0 };
  const velocity = reacquired
    ? { x: 0, y: 0 }
    : {
        x: track.velocity.x * (1 - VELOCITY_SMOOTHING) + measured.x * VELOCITY_SMOOTHING,
        y: track.velocity.y * (1 - VELOCITY_SMOOTHING) + measured.y * VELOCITY_SMOOTHING,
      };

  return {
    ...track,
    box,
    velocity,
    liveSignature: reacquired
      ? candidate.signature ?? track.liveSignature
      : blendSignatures(
          track.liveSignature ?? track.anchorSignature,
          candidate.signature,
          SIGNATURE_ADAPTATION,
        ),
    status: 'tracking',
    misses: 0,
    confidence: association.score,
    secondsSinceMatch: 0,
  };
}

function sizeSimilarity(previous: Box, current: Box): number {
  const widthDiff =
    Math.abs(previous.width - current.width) / Math.max(previous.width, current.width, 1);
  const heightDiff =
    Math.abs(previous.height - current.height) / Math.max(previous.height, current.height, 1);

  return clamp(1 - (widthDiff + heightDiff) / 2, 0, 1);
}
