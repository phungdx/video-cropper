import { centerCropRect, clamp, computeCropRect, type Box, type FrameSize } from './geometry';

/** Frames per second requested from the canvas capture stream. */
export const EXPORT_FRAME_RATE = 30;
/** Tall side cap so a 4K source does not produce an unusable export. */
export const MAX_EXPORT_HEIGHT = 1920;
/** Tall side floor, so a crop taken from a distant subject still yields a watchable file. */
export const MIN_EXPORT_HEIGHT = 640;

export type RecordingFormat = {
  mimeType: string;
  extension: string;
};

/**
 * Ordered by preference: MP4 first because it plays everywhere, then WebM variants which are
 * what Chrome and Firefox actually hand back today.
 */
const RECORDING_FORMATS: readonly RecordingFormat[] = [
  { mimeType: 'video/mp4;codecs=avc1.42E01E,mp4a.40.2', extension: 'mp4' },
  { mimeType: 'video/mp4', extension: 'mp4' },
  { mimeType: 'video/webm;codecs=vp9,opus', extension: 'webm' },
  { mimeType: 'video/webm;codecs=vp8,opus', extension: 'webm' },
  { mimeType: 'video/webm', extension: 'webm' },
];

export function pickRecordingFormat(
  isSupported: (mimeType: string) => boolean,
): RecordingFormat | null {
  return RECORDING_FORMATS.find((format) => isSupported(format.mimeType)) ?? null;
}

/** Video encoders reject odd dimensions, so every exported side is rounded to an even number. */
function toEvenSize(value: number): number {
  return Math.max(2, Math.round(value / 2) * 2);
}

/**
 * Picks the output resolution from the crop itself rather than the source frame, so a tight crop
 * is written close to its native pixels instead of being blown up to full frame height.
 */
export function resolveExportSize(
  crop: FrameSize,
  aspectRatio: number,
  maxHeight = MAX_EXPORT_HEIGHT,
): FrameSize {
  const height = toEvenSize(
    clamp(crop.height, Math.min(MIN_EXPORT_HEIGHT, maxHeight), maxHeight),
  );

  return {
    width: toEvenSize(height * aspectRatio),
    height,
  };
}

export function buildDownloadFileName(sourceName: string | null, extension: string): string {
  const base = (sourceName ?? '')
    .replace(/\.[^./\\]+$/, '')
    .replace(/[\\/:*?"<>|]+/g, '-')
    .trim();

  return `${base.length > 0 ? base : 'video'}-cropped.${extension}`;
}

export function exportProgress(currentTime: number, duration: number): number {
  if (!Number.isFinite(duration) || duration <= 0) {
    return 0;
  }

  return Math.min(1, Math.max(0, currentTime / duration));
}

export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 MB';
  }

  const megabytes = bytes / (1024 * 1024);

  return megabytes < 1 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${megabytes.toFixed(1)} MB`;
}

/**
 * Draws one full-bleed cropped frame into the export canvas and returns the crop that was used,
 * so the caller can hold the last framing when the tracker has nothing new to offer.
 */
export function drawExportFrame(
  ctx: CanvasRenderingContext2D,
  source: CanvasImageSource,
  frame: FrameSize,
  box: Box | null,
  fallbackCrop: Box | null,
  aspectRatio: number,
  output: FrameSize,
): Box {
  const crop =
    box !== null
      ? computeCropRect(box, frame, aspectRatio)
      : fallbackCrop ?? centerCropRect(frame, aspectRatio);

  ctx.drawImage(
    source,
    crop.x,
    crop.y,
    crop.width,
    crop.height,
    0,
    0,
    output.width,
    output.height,
  );

  return crop;
}
