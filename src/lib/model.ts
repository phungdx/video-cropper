import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import type { Detection } from './tracking';

let detectorPromise: Promise<cocoSsd.ObjectDetection> | null = null;

export async function loadDetector(): Promise<cocoSsd.ObjectDetection> {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      await tf.ready();

      await tf.setBackend('webgl').catch(() => undefined);

      return cocoSsd.load({ base: 'lite_mobilenet_v2' });
    })();
  }

  return detectorPromise;
}

export async function detectPeople(
  detector: cocoSsd.ObjectDetection,
  source: HTMLVideoElement | HTMLCanvasElement,
): Promise<Detection[]> {
  const results = await detector.detect(source);

  return results
    .filter((result) => result.class === 'person')
    .map((result) => ({
      label: result.class,
      score: result.score,
      box: {
        x: result.bbox[0],
        y: result.bbox[1],
        width: result.bbox[2],
        height: result.bbox[3],
      },
    }));
}
