import { useEffect, useRef, useState } from 'react';
import {
  boxAspectRatio,
  clamp,
  centerCropRect,
  computeCropRect,
  pointInBox,
  DEFAULT_CROP_ASPECT_RATIO,
  type Box,
  type FrameSize,
  type Point,
} from './lib/geometry';
import {
  choosePersonForClick,
  smoothBox,
  type Detection,
} from './lib/tracking';
import { captureAppearanceSignature } from './lib/reid';
import {
  buildDownloadFileName,
  drawExportFrame,
  exportProgress,
  formatFileSize,
  pickRecordingFormat,
  resolveExportSize,
  EXPORT_FRAME_RATE,
} from './lib/export';
import {
  advanceTrack,
  associateTarget,
  createTrack,
  predictBox,
  LOST_AFTER_MISSES,
  type TargetTrack,
  type TrackCandidate,
} from './lib/tracker';
import type { ObjectDetection } from '@tensorflow-models/coco-ssd';
import type { FastExportFrame } from './lib/fast-export';

/** How far ahead of the last detection the box is allowed to coast on screen, in seconds. */
const MAX_RENDER_EXTRAPOLATION = 0.25;
/** Per-frame easing applied to the on-screen box so it glides instead of stepping. */
const RENDER_SMOOTHING = 0.35;

/**
 * Canvas overlays are drawn imperatively, so the brand palette is mirrored here to stay in
 * step with the CSS custom properties in styles.css.
 */
const PALETTE = {
  ink: '#141413',
  light: '#faf9f5',
  coral: '#d97757',
  blue: '#6a9bcc',
  green: '#788c5d',
  brick: '#93382a',
  displayFont: 'Lora, ui-serif, Georgia, serif',
  uiFont: 'Poppins, ui-sans-serif, system-ui, sans-serif',
} as const;

type ModelPhase = 'idle' | 'loading' | 'ready' | 'error';
type TrackingPhase = 'idle' | 'ready' | 'tracking' | 'coasting' | 'lost';
type ExportPhase = 'idle' | 'recording' | 'finishing' | 'ready';
type CropMode = 'follow' | 'fixed';
type BlurCanvases = {
  source: HTMLCanvasElement | null;
  blurred: HTMLCanvasElement | null;
};
type BlurRegion = {
  id: number;
  box: Box;
  startTime: number;
  endTime: number;
};

type ExportResult = {
  url: string;
  fileName: string;
  size: number;
};

type AudioTap = {
  source: MediaElementAudioSourceNode;
  destination: MediaStreamAudioDestinationNode;
};

type TrackingSnapshot = {
  phase: TrackingPhase;
  confidence: number;
  message: string;
  detections: number;
  targetId: string | null;
};

type VideoMeta = {
  width: number;
  height: number;
  duration: number;
};

type CanvasSize = {
  width: number;
  height: number;
};

const defaultTrackingSnapshot: TrackingSnapshot = {
  phase: 'idle',
  confidence: 0,
  message: 'Upload a video to begin.',
  detections: 0,
  targetId: null,
};

function describeTrack(track: TargetTrack, detections: number): TrackingSnapshot {
  if (track.status === 'tracking') {
    return {
      phase: 'tracking',
      confidence: track.confidence,
      message: `Locked on ${track.id}. The box follows them until the video ends.`,
      detections,
      targetId: track.id,
    };
  }

  if (track.status === 'coasting') {
    return {
      phase: 'coasting',
      confidence: track.confidence,
      message: `${track.id} is hidden. Holding their path and waiting for them to reappear.`,
      detections,
      targetId: track.id,
    };
  }

  return {
    phase: 'lost',
    confidence: 0,
    message: `Searching the whole frame for ${track.id}. The crop stays put until they return.`,
    detections,
    targetId: track.id,
  };
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return '0:00';
  }

  const totalSeconds = Math.floor(seconds);
  const minutes = Math.floor(totalSeconds / 60);
  const remainder = String(totalSeconds % 60).padStart(2, '0');
  return `${minutes}:${remainder}`;
}

function formatClockTime(seconds: number): string {
  const totalSeconds = Math.max(0, Math.ceil(seconds));
  const hours = String(Math.floor(totalSeconds / 3600)).padStart(2, '0');
  const minutes = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, '0');
  const remainder = String(totalSeconds % 60).padStart(2, '0');
  return `${hours}:${minutes}:${remainder}`;
}

function parseClockTime(value: string): number | null {
  const match = /^(\d{2,}):([0-5]\d):([0-5]\d)$/.exec(value.trim());
  if (!match) {
    return null;
  }

  return Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3]);
}

function useObservedCanvasSize<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState<CanvasSize>({ width: 1, height: 1 });

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    const updateSize = () => {
      const rect = element.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;

      setSize({
        width: Math.max(1, Math.round(rect.width * dpr)),
        height: Math.max(1, Math.round(rect.height * dpr)),
      });
    };

    updateSize();
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(updateSize);
    observer?.observe(element);
    window.addEventListener('resize', updateSize);

    return () => {
      observer?.disconnect();
      window.removeEventListener('resize', updateSize);
    };
  }, []);

  return { ref, size };
}

function resizeCanvas(canvas: HTMLCanvasElement, size: CanvasSize) {
  if (canvas.width !== size.width) {
    canvas.width = size.width;
  }

  if (canvas.height !== size.height) {
    canvas.height = size.height;
  }
}

function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const r = Math.min(radius, width / 2, height / 2);

  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function drawBanner(
  ctx: CanvasRenderingContext2D,
  message: string,
  canvasSize: CanvasSize,
  subtitle?: string,
) {
  ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

  const centerX = canvasSize.width / 2;
  const centerY = canvasSize.height / 2;

  ctx.save();
  ctx.fillStyle = 'rgba(250, 249, 245, 0.94)';
  drawRoundedRect(ctx, centerX - 230, centerY - 62, 460, 124, 14);
  ctx.fill();

  ctx.strokeStyle = 'rgba(20, 20, 19, 0.12)';
  ctx.lineWidth = Math.max(1, canvasSize.width / 640);
  ctx.stroke();

  ctx.fillStyle = PALETTE.ink;
  ctx.font = `500 27px ${PALETTE.displayFont}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(message, centerX, centerY - 10);

  if (subtitle) {
    ctx.fillStyle = 'rgba(20, 20, 19, 0.6)';
    ctx.font = `400 15px ${PALETTE.uiFont}`;
    ctx.fillText(subtitle, centerX, centerY + 24);
  }

  ctx.restore();
}

function drawBoxWithLabel(
  ctx: CanvasRenderingContext2D,
  box: Box,
  label: string,
  scaleX: number,
  scaleY: number,
  borderColor: string,
  fillColor: string,
  lineWidth: number,
) {
  const x = box.x * scaleX;
  const y = box.y * scaleY;
  const width = box.width * scaleX;
  const height = box.height * scaleY;

  ctx.save();
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = borderColor;
  ctx.fillStyle = fillColor;
  drawRoundedRect(ctx, x, y, width, height, 10);
  ctx.fill();
  ctx.stroke();

  ctx.font = `500 15px ${PALETTE.uiFont}`;
  ctx.textAlign = 'left';
  const labelWidth = ctx.measureText(label).width + 22;
  const labelHeight = 28;
  const labelY = Math.max(6, y - labelHeight - 8);

  ctx.fillStyle = borderColor;
  drawRoundedRect(ctx, x, labelY, labelWidth, labelHeight, 6);
  ctx.fill();

  ctx.fillStyle = PALETTE.light;
  ctx.textBaseline = 'middle';
  ctx.fillText(label, x + 11, labelY + labelHeight / 2 + 1);
  ctx.restore();
}

function drawCandidateBox(
  ctx: CanvasRenderingContext2D,
  detection: Detection,
  index: number,
  scaleX: number,
  scaleY: number,
) {
  drawBoxWithLabel(
    ctx,
    detection.box,
    `${index + 1} · ${Math.round(detection.score * 100)}%`,
    scaleX,
    scaleY,
    'rgba(106, 155, 204, 0.92)',
    'rgba(106, 155, 204, 0.1)',
    2,
  );
}

function drawSourceOverlay(
  ctx: CanvasRenderingContext2D,
  canvasSize: CanvasSize,
  videoMeta: VideoMeta | null,
  detections: Detection[],
  track: TargetTrack | null,
  trackedBox: Box | null,
) {
  ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

  if (!videoMeta) {
    drawBanner(ctx, 'Upload a video', canvasSize, 'The browser preview and tracking tools will appear here.');
    return;
  }

  const frame: FrameSize = { width: videoMeta.width, height: videoMeta.height };
  const scaleX = canvasSize.width / frame.width;
  const scaleY = canvasSize.height / frame.height;

  if (!track || !trackedBox) {
    detections.forEach((detection, index) => {
      drawCandidateBox(ctx, detection, index, scaleX, scaleY);
    });

    drawBanner(
      ctx,
      detections.length > 0 ? 'Click the person to follow' : 'Waiting for a person',
      canvasSize,
      detections.length > 0
        ? 'Whoever you click stays in the crop for the rest of the video.'
        : 'Pause on a clear frame, then click the person to lock on.',
    );
    return;
  }

  const trackedCenter = {
    x: trackedBox.x + trackedBox.width / 2,
    y: trackedBox.y + trackedBox.height / 2,
  };

  detections.forEach((detection, index) => {
    const center = {
      x: detection.box.x + detection.box.width / 2,
      y: detection.box.y + detection.box.height / 2,
    };
    const isTarget =
      Math.hypot(center.x - trackedCenter.x, center.y - trackedCenter.y) <
      Math.max(trackedBox.width, trackedBox.height) * 0.4;

    if (!isTarget) {
      drawCandidateBox(ctx, detection, index, scaleX, scaleY);
    }
  });

  const locked = track.status === 'tracking';
  const borderColor = locked
    ? PALETTE.green
    : track.status === 'coasting'
      ? PALETTE.coral
      : PALETTE.brick;
  const label = locked
    ? `${track.id} · ${Math.round(track.confidence * 100)}%`
    : track.status === 'coasting'
      ? `${track.id} · hidden`
      : `${track.id} · searching`;

  drawBoxWithLabel(
    ctx,
    trackedBox,
    label,
    scaleX,
    scaleY,
    borderColor,
    locked ? 'rgba(120, 140, 93, 0.18)' : 'rgba(217, 119, 87, 0.14)',
    3.5,
  );

  if (!locked) {
    ctx.save();
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([10, 8]);
    ctx.strokeRect(
      trackedBox.x * scaleX,
      trackedBox.y * scaleY,
      trackedBox.width * scaleX,
      trackedBox.height * scaleY,
    );
    ctx.restore();
  }
}

/**
 * Paints exactly what an export would record: the subject's bounding box, full bleed. No dimming
 * or highlight, because the crop is the box now, so there is no surrounding context to play down.
 */
function drawPreviewFrame(
  ctx: CanvasRenderingContext2D,
  canvasSize: CanvasSize,
  video: HTMLVideoElement | null,
  videoMeta: VideoMeta | null,
  trackedBox: Box | null,
  aspectRatio: number,
  cropMode: CropMode,
  fixedCrop: Box | null,
  blurRegions: BlurRegion[],
  blurCanvases: BlurCanvases,
  blurStrength: number,
) {
  ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

  if (
    !video ||
    !videoMeta ||
    (cropMode === 'follow' && !trackedBox) ||
    (cropMode === 'fixed' && !fixedCrop) ||
    video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    drawBanner(ctx, 'Preview crop', canvasSize, 'Select a person to see the adaptive crop follow them.');
    return;
  }

  const crop = cropMode === 'fixed' && fixedCrop
    ? fixedCrop
    : trackedBox
      ? computeCropRect(
          trackedBox,
          { width: videoMeta.width, height: videoMeta.height },
          aspectRatio,
        )
      : null;

  if (!crop) {
    drawBanner(ctx, 'Set your crop frame', canvasSize, 'Drag a rectangle on the source video.');
    return;
  }

  ctx.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, canvasSize.width, canvasSize.height);
  drawBlurRegions(
    ctx,
    video,
    crop,
    canvasSize,
    blurRegions,
    blurCanvases,
    blurStrength,
    video.currentTime,
  );
}

function drawBlurRegions(
  ctx: CanvasRenderingContext2D,
  video: CanvasImageSource,
  crop: Box,
  output: CanvasSize,
  regions: BlurRegion[],
  canvases: BlurCanvases,
  blurStrength: number,
  currentTime: number,
) {
  const visibleRegions = regions
    .filter((region) => currentTime >= region.startTime && currentTime <= region.endTime)
    .map(({ box: region }) => ({
      left: Math.max(region.x, crop.x),
      top: Math.max(region.y, crop.y),
      right: Math.min(region.x + region.width, crop.x + crop.width),
      bottom: Math.min(region.y + region.height, crop.y + crop.height),
    }))
    .filter((region) => region.right > region.left && region.bottom > region.top);

  if (visibleRegions.length === 0) {
    return;
  }

  const sourceCanvas = canvases.source ?? (canvases.source = document.createElement('canvas'));
  const blurredCanvas = canvases.blurred ?? (canvases.blurred = document.createElement('canvas'));
  for (const canvas of [sourceCanvas, blurredCanvas]) {
    if (canvas.width !== output.width) canvas.width = output.width;
    if (canvas.height !== output.height) canvas.height = output.height;
  }

  const sourceCtx = sourceCanvas.getContext('2d');
  const blurCtx = blurredCanvas.getContext('2d');
  if (!sourceCtx || !blurCtx) {
    return;
  }

  sourceCtx.clearRect(0, 0, output.width, output.height);
  sourceCtx.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, output.width, output.height);
  blurCtx.clearRect(0, 0, output.width, output.height);
  blurCtx.filter = `blur(${(blurStrength * output.height) / 1080}px)`;
  blurCtx.drawImage(sourceCanvas, 0, 0);
  blurCtx.filter = 'none';

  ctx.save();
  ctx.beginPath();
  for (const region of visibleRegions) {
    const x = ((region.left - crop.x) / crop.width) * output.width;
    const y = ((region.top - crop.y) / crop.height) * output.height;
    const width = ((region.right - region.left) / crop.width) * output.width;
    const height = ((region.bottom - region.top) / crop.height) * output.height;
    ctx.rect(x, y, width, height);
  }
  ctx.clip();
  ctx.drawImage(blurredCanvas, 0, 0);
  ctx.restore();
}

function drawBlurOverlay(
  ctx: CanvasRenderingContext2D,
  canvasSize: CanvasSize,
  frame: FrameSize,
  regions: BlurRegion[],
  draft: Box | null,
  currentTime: number,
) {
  const scaleX = canvasSize.width / frame.width;
  const scaleY = canvasSize.height / frame.height;
  ctx.save();
  ctx.lineWidth = Math.max(2, canvasSize.width / 500);
  ctx.setLineDash([8, 5]);

  regions.forEach((entry, index) => {
    const region = entry.box;
    const active = currentTime >= entry.startTime && currentTime <= entry.endTime;
    const x = region.x * scaleX;
    const y = region.y * scaleY;
    const width = region.width * scaleX;
    const height = region.height * scaleY;
    ctx.strokeStyle = active ? PALETTE.coral : 'rgba(250, 249, 245, 0.6)';
    ctx.fillStyle = active ? 'rgba(217, 119, 87, 0.18)' : 'rgba(250, 249, 245, 0.08)';
    ctx.fillRect(x, y, width, height);
    ctx.strokeRect(x, y, width, height);
    ctx.fillStyle = PALETTE.light;
    ctx.font = `500 13px ${PALETTE.uiFont}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`Blur ${index + 1}`, x + 6, y - 5);
  });

  if (draft) {
    ctx.strokeStyle = PALETTE.coral;
    ctx.fillStyle = 'rgba(217, 119, 87, 0.18)';
    ctx.setLineDash([8, 5]);
    ctx.fillRect(draft.x * scaleX, draft.y * scaleY, draft.width * scaleX, draft.height * scaleY);
    ctx.strokeRect(draft.x * scaleX, draft.y * scaleY, draft.width * scaleX, draft.height * scaleY);
  }

  ctx.restore();
}

function drawFixedCropOverlay(
  ctx: CanvasRenderingContext2D,
  canvasSize: CanvasSize,
  frame: FrameSize,
  crop: Box | null,
) {
  ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);
  const scaleX = canvasSize.width / frame.width;
  const scaleY = canvasSize.height / frame.height;

  if (!crop) {
    drawBanner(ctx, 'Set your crop frame', canvasSize, 'Drag across the video to choose the fixed crop.');
    return;
  }

  const x = crop.x * scaleX;
  const y = crop.y * scaleY;
  const width = crop.width * scaleX;
  const height = crop.height * scaleY;

  ctx.save();
  ctx.fillStyle = 'rgba(20, 20, 19, 0.58)';
  ctx.fillRect(0, 0, canvasSize.width, canvasSize.height);
  ctx.clearRect(x, y, width, height);
  ctx.strokeStyle = PALETTE.coral;
  ctx.lineWidth = Math.max(2, canvasSize.width / 500);
  ctx.strokeRect(x, y, width, height);
  ctx.fillStyle = PALETTE.light;
  ctx.font = `500 14px ${PALETTE.uiFont}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText('Fixed crop', x + 8, y - 7);
  ctx.restore();
}

function cropFromPoints(start: Point, end: Point, frame: FrameSize): Box | null {
  const left = clamp(Math.min(start.x, end.x), 0, frame.width);
  const top = clamp(Math.min(start.y, end.y), 0, frame.height);
  const right = clamp(Math.max(start.x, end.x), 0, frame.width);
  const bottom = clamp(Math.max(start.y, end.y), 0, frame.height);
  const width = right - left;
  const height = bottom - top;

  return width >= 24 && height >= 24 ? { x: left, y: top, width, height } : null;
}

/** Resolves once the video has actually landed on the requested timestamp. */
function seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    if (Math.abs(video.currentTime - time) < 0.05) {
      resolve();
      return;
    }

    const handleSettled = () => {
      video.removeEventListener('seeked', handleSettled);
      resolve();
    };

    video.addEventListener('seeked', handleSettled);
    video.currentTime = time;
  });
}

function getVideoPointFromClick(
  event: React.MouseEvent<HTMLCanvasElement>,
  canvas: HTMLCanvasElement,
  videoMeta: VideoMeta,
): Point {
  const rect = canvas.getBoundingClientRect();

  return {
    x: ((event.clientX - rect.left) / rect.width) * videoMeta.width,
    y: ((event.clientY - rect.top) / rect.height) * videoMeta.height,
  };
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sourceStage = useObservedCanvasSize<HTMLDivElement>();
  const previewStage = useObservedCanvasSize<HTMLDivElement>();
  const detectorRef = useRef<ObjectDetection | null>(null);
  const detectionsRef = useRef<Detection[]>([]);
  const trackRef = useRef<TargetTrack | null>(null);
  const renderBoxRef = useRef<Box | null>(null);
  // Locked in when the subject is picked. The crop shape has to hold still for the whole clip:
  // the recorder cannot resize its canvas mid-take, and a shape that breathed with every
  // detection would make the subject pulse.
  const cropAspectRef = useRef(DEFAULT_CROP_ASPECT_RATIO);
  const cropModeRef = useRef<CropMode>('follow');
  const fixedCropRef = useRef<Box | null>(null);
  const cropDragStartRef = useRef<Point | null>(null);
  const blurEditingRef = useRef(false);
  const blurRegionsRef = useRef<BlurRegion[]>([]);
  const blurDraftRef = useRef<Box | null>(null);
  const nextBlurRegionIdRef = useRef(1);
  const blurStrengthRef = useRef(24);
  const lastTrackTimeRef = useRef(0);
  const detectionInFlightRef = useRef(false);
  const nextTargetIdRef = useRef(1);
  const sourceFileRef = useRef<File | null>(null);
  const fastExportActiveRef = useRef(false);
  const cancelFastExportRef = useRef(false);
  const videoUrlRef = useRef<string | null>(null);
  const exportCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const blurCanvasesRef = useRef<BlurCanvases>({ source: null, blurred: null });
  const exportCropRef = useRef<Box | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordingRef = useRef(false);
  const chunksRef = useRef<Blob[]>([]);
  const discardExportRef = useRef(false);
  const exportUrlRef = useRef<string | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioTapRef = useRef<AudioTap | null>(null);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [videoMeta, setVideoMeta] = useState<VideoMeta | null>(null);
  const [cropAspect, setCropAspect] = useState(DEFAULT_CROP_ASPECT_RATIO);
  const [cropMode, setCropMode] = useState<CropMode>('follow');
  const [fixedCrop, setFixedCrop] = useState<Box | null>(null);
  const [blurEditing, setBlurEditing] = useState(false);
  const [blurRegions, setBlurRegions] = useState<BlurRegion[]>([]);
  const [blurTimeDrafts, setBlurTimeDrafts] = useState<Record<string, string>>({});
  const [blurStrength, setBlurStrength] = useState(24);
  const [modelPhase, setModelPhase] = useState<ModelPhase>('idle');
  const [trackingSnapshot, setTrackingSnapshot] = useState<TrackingSnapshot>(defaultTrackingSnapshot);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exportPhase, setExportPhase] = useState<ExportPhase>('idle');
  const [fastExportActive, setFastExportActive] = useState(false);
  const [exportedFraction, setExportedFraction] = useState(0);
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);

  useEffect(() => {
    return () => {
      cancelFastExportRef.current = true;
      if (videoUrlRef.current) {
        URL.revokeObjectURL(videoUrlRef.current);
      }

      if (exportUrlRef.current) {
        URL.revokeObjectURL(exportUrlRef.current);
      }

      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        recorderRef.current.stop();
      }

      void audioContextRef.current?.close().catch(() => undefined);
    };
  }, []);

  useEffect(() => {
    if (!videoUrl) {
      detectorRef.current = null;
      setModelPhase('idle');
      return;
    }

    let cancelled = false;
    setModelPhase('loading');
    setError(null);

    import('./lib/model')
      .then(async (modelModule) => {
        const detector = await modelModule.loadDetector();

        if (cancelled) {
          return undefined;
        }

        detectorRef.current = detector;
        setModelPhase('ready');
        return undefined;
      })
      .catch(() => {
        if (!cancelled) {
          setModelPhase('error');
          setError('Could not load the browser detector.');
        }
      });

    return () => {
      cancelled = true;
    };
  }, [videoUrl]);

  async function runDetection() {
    const video = videoRef.current;
    const detector = detectorRef.current;

    if (!video || !detector || !videoMeta || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return false;
    }

    if (detectionInFlightRef.current) {
      return false;
    }

    detectionInFlightRef.current = true;

    try {
      const { detectPeople } = await import('./lib/model');
      const detections = await detectPeople(detector, video);
      detectionsRef.current = detections;

      const frame: FrameSize = { width: videoMeta.width, height: videoMeta.height };
      const track = trackRef.current;

      if (!track) {
        setTrackingSnapshot({
          phase: detections.length > 0 ? 'ready' : 'idle',
          confidence: 0,
          message:
            detections.length > 0
              ? 'Click a person in the frame to lock on.'
              : 'No person detected yet. Try a clearer frame.',
          detections: detections.length,
          targetId: null,
        });
        return true;
      }

      const candidates: TrackCandidate[] = detections.map((detection) => ({
        detection,
        signature: captureAppearanceSignature(video, detection.box, frame),
      }));

      // Playback time, not wall-clock time, so pausing or scrubbing cannot fake motion.
      const dt = clamp(video.currentTime - lastTrackTimeRef.current, 0, 1);
      lastTrackTimeRef.current = video.currentTime;

      const association = associateTarget(track, candidates, { dt, frame });
      const nextTrack = advanceTrack(track, candidates, association, { dt, frame });
      trackRef.current = nextTrack;
      setTrackingSnapshot(describeTrack(nextTrack, detections.length));

      return true;
    } catch {
      setModelPhase('error');
      setError('Person detection failed on this video.');
      return false;
    } finally {
      detectionInFlightRef.current = false;
    }
  }

  useEffect(() => {
    if (!videoMeta) {
      return;
    }

    let raf = 0;
    const frame: FrameSize = { width: videoMeta.width, height: videoMeta.height };

    const render = () => {
      const sourceCanvas = overlayCanvasRef.current;
      const previewCanvas = previewCanvasRef.current;
      const video = videoRef.current;
      const track = trackRef.current;

      // Detections land a few times a second, so the on-screen box coasts along the tracked
      // velocity between passes and eases towards each new measurement.
      if (!track) {
        renderBoxRef.current = null;
      } else {
        const elapsed = video
          ? clamp(video.currentTime - lastTrackTimeRef.current, 0, MAX_RENDER_EXTRAPOLATION)
          : 0;
        const aim =
          track.status === 'tracking'
            ? predictBox(track.box, track.velocity, elapsed, frame)
            : track.box;

        renderBoxRef.current = renderBoxRef.current
          ? smoothBox(renderBoxRef.current, aim, RENDER_SMOOTHING)
          : aim;
      }

      if (sourceCanvas) {
        resizeCanvas(sourceCanvas, sourceStage.size);
        const ctx = sourceCanvas.getContext('2d');
        if (ctx) {
          if (cropModeRef.current === 'fixed') {
            drawFixedCropOverlay(ctx, sourceStage.size, frame, fixedCropRef.current);
          } else {
            drawSourceOverlay(
              ctx,
              sourceStage.size,
              videoMeta,
              detectionsRef.current,
              track,
              renderBoxRef.current,
            );
          }
          drawBlurOverlay(
            ctx,
            sourceStage.size,
            frame,
            blurRegionsRef.current,
            blurDraftRef.current,
            video?.currentTime ?? 0,
          );
        }
      }

      if (previewCanvas) {
        resizeCanvas(previewCanvas, previewStage.size);
        const ctx = previewCanvas.getContext('2d');
        if (ctx) {
          drawPreviewFrame(
            ctx,
            previewStage.size,
            video,
            videoMeta,
            cropModeRef.current === 'fixed' ? null : renderBoxRef.current,
            cropAspectRef.current,
            cropModeRef.current,
            fixedCropRef.current,
            blurRegionsRef.current,
            blurCanvasesRef.current,
            blurStrengthRef.current,
          );
        }
      }

      // The recorder pulls frames straight off this canvas, so it has to be painted on every
      // animation frame even when the tracker has nothing new to report.
      const exportCanvas = exportCanvasRef.current;
      if (recordingRef.current && exportCanvas && video && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        const ctx = exportCanvas.getContext('2d');
        if (ctx) {
          exportCropRef.current = drawExportFrame(
            ctx,
            video,
            frame,
            renderBoxRef.current,
            exportCropRef.current,
            cropAspectRef.current,
            { width: exportCanvas.width, height: exportCanvas.height },
            cropModeRef.current === 'fixed' ? fixedCropRef.current : null,
          );
          drawBlurRegions(
            ctx,
            video,
            exportCropRef.current,
            { width: exportCanvas.width, height: exportCanvas.height },
            blurRegionsRef.current,
            blurCanvasesRef.current,
            blurStrengthRef.current,
            video.currentTime,
          );
        }
      }

      raf = window.requestAnimationFrame(render);
    };

    raf = window.requestAnimationFrame(render);
    return () => window.cancelAnimationFrame(raf);
  }, [previewStage.size, sourceStage.size, videoMeta]);

  useEffect(() => {
    if (!videoMeta || modelPhase !== 'ready') {
      return;
    }

    void runDetection();
  }, [modelPhase, videoMeta]);

  useEffect(() => {
    if (!videoMeta || modelPhase !== 'ready' || !isPlaying) {
      return;
    }

    // Run detection back to back while the video plays instead of on a fixed interval, so the
    // box keeps up with the person for the whole clip on whatever hardware is available.
    let active = true;
    let raf = 0;

    const pump = async () => {
      if (!active) {
        return;
      }

      await runDetection();

      if (!active) {
        return;
      }

      raf = window.requestAnimationFrame(() => {
        void pump();
      });
    };

    void pump();

    return () => {
      active = false;
      window.cancelAnimationFrame(raf);
    };
  }, [isPlaying, modelPhase, videoMeta]);

  const sourceAspectRatio = videoMeta ? `${videoMeta.width} / ${videoMeta.height}` : '16 / 9';
  const isExporting = exportPhase === 'recording' || exportPhase === 'finishing';
  const canExport = Boolean(videoMeta) && (
    cropMode === 'fixed' ? Boolean(fixedCrop) : Boolean(trackingSnapshot.targetId) && modelPhase === 'ready'
  );
  const exportPercent = Math.round(exportedFraction * 100);

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (videoUrlRef.current) {
      URL.revokeObjectURL(videoUrlRef.current);
    }

    cancelFastExportRef.current = true;
    sourceFileRef.current = file;

    finishExport(true);
    releaseExportResult();

    const nextUrl = URL.createObjectURL(file);
    videoUrlRef.current = nextUrl;
    setVideoUrl(nextUrl);
    setFileName(file.name);
    setVideoMeta(null);
    setCurrentTime(0);
    setIsPlaying(false);
    setError(null);
    trackRef.current = null;
    renderBoxRef.current = null;
    detectionsRef.current = [];
    lastTrackTimeRef.current = 0;
    nextTargetIdRef.current = 1;
    cropAspectRef.current = DEFAULT_CROP_ASPECT_RATIO;
    setCropAspect(DEFAULT_CROP_ASPECT_RATIO);
    cropModeRef.current = 'follow';
    setCropMode('follow');
    fixedCropRef.current = null;
    setFixedCrop(null);
    blurEditingRef.current = false;
    setBlurEditing(false);
    blurRegionsRef.current = [];
    setBlurRegions([]);
    blurDraftRef.current = null;
    nextBlurRegionIdRef.current = 1;
    setBlurTimeDrafts({});
    blurStrengthRef.current = 24;
    setBlurStrength(24);
    setTrackingSnapshot(defaultTrackingSnapshot);
  }

  function handleLoadedMetadata() {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    setVideoMeta({
      width: video.videoWidth,
      height: video.videoHeight,
      duration: video.duration,
    });
    setCurrentTime(video.currentTime);
  }

  function handleSourceClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if (cropModeRef.current === 'fixed' || blurEditingRef.current) {
      return;
    }
    const video = videoRef.current;
    if (!videoMeta || !video) {
      return;
    }

    const point = getVideoPointFromClick(event, event.currentTarget, videoMeta);
    const candidate = choosePersonForClick(detectionsRef.current, point);

    // Only trust the cached boxes when the click actually landed on one. Otherwise the list may
    // be from an older frame, and locking on it would capture the wrong person.
    if (candidate && pointInBox(point, candidate.box)) {
      activateTarget(candidate);
      return;
    }

    void runDetection().then(() => {
      const refreshed = choosePersonForClick(detectionsRef.current, point);

      if (refreshed) {
        activateTarget(refreshed);
        return;
      }

      setTrackingSnapshot({
        phase: 'idle',
        confidence: 0,
        message: 'No person found at that point. Try a different frame or person.',
        detections: detectionsRef.current.length,
        targetId: null,
      });
    });
  }

  function handleCropModeChange(mode: CropMode) {
    cropModeRef.current = mode;
    setCropMode(mode);

    if (mode === 'fixed' && videoMeta) {
      const crop = fixedCropRef.current ?? centerCropRect(videoMeta, cropAspectRef.current);
      fixedCropRef.current = crop;
      setFixedCrop(crop);
      cropAspectRef.current = crop.width / crop.height;
      setCropAspect(cropAspectRef.current);
    }
  }

  function getPointerVideoPoint(event: React.PointerEvent<HTMLCanvasElement>): Point | null {
    if (!videoMeta) {
      return null;
    }

    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * videoMeta.width,
      y: ((event.clientY - rect.top) / rect.height) * videoMeta.height,
    };
  }

  function handleCropPointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if ((!blurEditingRef.current && cropModeRef.current !== 'fixed') || !videoMeta || isExporting) {
      return;
    }

    const point = getPointerVideoPoint(event);
    if (!point) {
      return;
    }

    cropDragStartRef.current = point;
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handleCropPointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    const start = cropDragStartRef.current;
    if ((!blurEditingRef.current && cropModeRef.current !== 'fixed') || !start || !videoMeta) {
      return;
    }

    const point = getPointerVideoPoint(event);
    if (point) {
      const crop = cropFromPoints(start, point, videoMeta);
      if (crop) {
        if (blurEditingRef.current) {
          blurDraftRef.current = crop;
        } else {
          fixedCropRef.current = crop;
        }
      }
    }
  }

  function handleCropPointerUp(event: React.PointerEvent<HTMLCanvasElement>) {
    const start = cropDragStartRef.current;
    cropDragStartRef.current = null;
    if ((!blurEditingRef.current && cropModeRef.current !== 'fixed') || !start || !videoMeta) {
      return;
    }

    const point = getPointerVideoPoint(event);
    const crop = point ? cropFromPoints(start, point, videoMeta) : null;
    if (!crop) {
      blurDraftRef.current = null;
      return;
    }

    if (blurEditingRef.current) {
      const duration = videoMeta.duration;
      const startTime = Math.min(Math.floor(currentTime), Math.max(0, duration - Math.min(1, duration)));
      const nextRegions = [
        ...blurRegionsRef.current,
        { id: nextBlurRegionIdRef.current, box: crop, startTime, endTime: duration },
      ];
      nextBlurRegionIdRef.current += 1;
      blurRegionsRef.current = nextRegions;
      setBlurRegions(nextRegions);
      blurDraftRef.current = null;
      return;
    }

    fixedCropRef.current = crop;
    setFixedCrop(crop);
    cropAspectRef.current = crop.width / crop.height;
    setCropAspect(cropAspectRef.current);
  }

  function toggleBlurEditing() {
    const nextEditing = !blurEditingRef.current;
    blurEditingRef.current = nextEditing;
    setBlurEditing(nextEditing);
    blurDraftRef.current = null;
    cropDragStartRef.current = null;
  }

  function clearBlurRegions() {
    blurRegionsRef.current = [];
    blurDraftRef.current = null;
    setBlurRegions([]);
    setBlurTimeDrafts({});
  }

  function handleBlurStrengthChange(event: React.ChangeEvent<HTMLInputElement>) {
    const strength = Number(event.target.value);
    blurStrengthRef.current = strength;
    setBlurStrength(strength);
  }

  function handleBlurRegionTimeChange(
    id: number,
    edge: 'startTime' | 'endTime',
    requestedValue: number,
  ) {
    const duration = videoMeta?.duration ?? 0;
    const minimumGap = Math.min(1, duration);
    if (duration <= 0) {
      return;
    }

    const requested = clamp(requestedValue, 0, duration);
    const nextRegions = blurRegionsRef.current.map((region, regionIndex) => {
      if (region.id !== id) {
        return region;
      }

      if (edge === 'startTime') {
        return { ...region, startTime: Math.min(requested, Math.max(0, region.endTime - minimumGap)) };
      }

      return { ...region, endTime: Math.max(requested, Math.min(duration, region.startTime + minimumGap)) };
    });
    blurRegionsRef.current = nextRegions;
    setBlurRegions(nextRegions);
  }

  function removeBlurRegion(index: number) {
    const removed = blurRegionsRef.current[index];
    const nextRegions = blurRegionsRef.current.filter((_, regionIndex) => regionIndex !== index);
    blurRegionsRef.current = nextRegions;
    setBlurRegions(nextRegions);
    if (removed) {
      setBlurTimeDrafts((drafts) => {
        const remaining = { ...drafts };
        delete remaining[blurTimeDraftKey(removed.id, 'startTime')];
        delete remaining[blurTimeDraftKey(removed.id, 'endTime')];
        return remaining;
      });
    }
  }

  function blurTimeDraftKey(id: number, edge: 'startTime' | 'endTime') {
    return `${id}:${edge}`;
  }

  function handleBlurTimeTextChange(id: number, edge: 'startTime' | 'endTime', value: string) {
    const key = blurTimeDraftKey(id, edge);
    setBlurTimeDrafts((drafts) => ({ ...drafts, [key]: value }));
    const seconds = parseClockTime(value);
    const duration = videoMeta?.duration ?? 0;
    const maxTime = edge === 'endTime' ? Math.ceil(duration) : duration;
    if (seconds !== null && seconds <= maxTime) {
      if (blurRegionsRef.current.some((region) => region.id === id)) {
        handleBlurRegionTimeChange(id, edge, seconds);
      }
    }
  }

  function finishBlurTimeTextEdit(id: number, edge: 'startTime' | 'endTime') {
    const key = blurTimeDraftKey(id, edge);
    setBlurTimeDrafts((drafts) => {
      const { [key]: _discarded, ...remaining } = drafts;
      return remaining;
    });
  }

  function activateTarget(candidate: Detection) {
    const video = videoRef.current;

    if (!videoMeta || !video) {
      return;
    }

    const targetId = `person ${nextTargetIdRef.current}`;
    nextTargetIdRef.current += 1;
    const frame: FrameSize = { width: videoMeta.width, height: videoMeta.height };
    const signature = captureAppearanceSignature(video, candidate.box, frame);
    const track = createTrack(targetId, candidate, signature);
    const aspectRatio = boxAspectRatio(candidate.box);

    cropAspectRef.current = aspectRatio;
    setCropAspect(aspectRatio);
    trackRef.current = track;
    renderBoxRef.current = candidate.box;
    lastTrackTimeRef.current = video.currentTime;
    setTrackingSnapshot(describeTrack(track, detectionsRef.current.length));
    setError(null);
  }

  async function handlePlayPause() {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    if (video.paused) {
      await video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  }

  function handleSeek(event: React.ChangeEvent<HTMLInputElement>) {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    const nextTime = Number(event.target.value);
    video.currentTime = nextTime;
    setCurrentTime(nextTime);
  }

  function releaseExportResult() {
    if (exportUrlRef.current) {
      URL.revokeObjectURL(exportUrlRef.current);
      exportUrlRef.current = null;
    }

    setExportResult(null);
    setExportedFraction(0);
    setExportPhase('idle');
  }

  /**
   * Taps the element's audio into a stream destination so the recorded file keeps the original
   * soundtrack. Best effort: a browser without Web Audio simply gets a silent export.
   */
  function captureAudioTracks(video: HTMLVideoElement): MediaStreamTrack[] {
    try {
      const AudioCtor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;

      if (!AudioCtor) {
        return [];
      }

      const context = audioContextRef.current ?? new AudioCtor();
      audioContextRef.current = context;

      let tap = audioTapRef.current;

      if (!tap) {
        // A media element can only ever have one source node, so the tap is created once and
        // reused for every later export.
        const source = context.createMediaElementSource(video);
        const destination = context.createMediaStreamDestination();
        source.connect(context.destination);
        source.connect(destination);
        tap = { source, destination };
        audioTapRef.current = tap;
      }

      void context.resume().catch(() => undefined);

      return tap.destination.stream.getAudioTracks();
    } catch {
      return [];
    }
  }

  /**
   * Closes out the recorder. `discard` throws the take away, which is what a video swap wants,
   * while the default keeps whatever was captured so an early stop still yields a file.
   */
  function finishExport(discard = false) {
    recordingRef.current = false;
    const recorder = recorderRef.current;
    recorderRef.current = null;

    if (recorder && recorder.state !== 'inactive') {
      discardExportRef.current = discard;

      if (!discard) {
        setExportPhase('finishing');
      }

      recorder.stop();
    }
  }

  async function tryFastExport(): Promise<boolean> {
    const file = sourceFileRef.current;
    const video = videoRef.current;
    const track = trackRef.current;

    if (
      !file ||
      !videoMeta ||
      !video ||
      (cropModeRef.current === 'follow' && (!track || !detectorRef.current))
    ) {
      return false;
    }

    const mode = cropModeRef.current;
    const fixedCrop = fixedCropRef.current;
    const frame: FrameSize = { width: videoMeta.width, height: videoMeta.height };
    const openingCrop = mode === 'fixed'
      ? fixedCrop ?? centerCropRect(frame, cropAspectRef.current)
      : computeCropRect(track!.box, frame, cropAspectRef.current);
    const outputSize = resolveExportSize(openingCrop, cropAspectRef.current);
    const canvas = document.createElement('canvas');
    canvas.width = outputSize.width;
    canvas.height = outputSize.height;
    const ctx = canvas.getContext('2d');
    const trackingCanvas = document.createElement('canvas');
    const trackingCtx = trackingCanvas.getContext('2d');

    if (!ctx || (mode === 'follow' && !trackingCtx)) {
      return false;
    }

    let workingTrack: TargetTrack | null = mode === 'follow' && track
      ? {
          ...track,
          status: 'lost',
          misses: LOST_AFTER_MISSES,
          secondsSinceMatch: 1.5,
          velocity: { x: 0, y: 0 },
        }
      : null;
    let lastDetectionTime: number | null = null;
    let renderedBox: Box | null = null;
    let fallbackCrop: Box | null = openingCrop;
    const fastBlurCanvases: BlurCanvases = { source: null, blurred: null };
    cancelFastExportRef.current = false;
    fastExportActiveRef.current = true;
    setFastExportActive(true);

    try {
      const { exportWholeVideo } = await import('./lib/fast-export');
      const blob = await exportWholeVideo(
        file,
        canvas,
        async (frameData: FastExportFrame) => {
          const sourceFrame: FrameSize = { width: frameData.width, height: frameData.height };

          const shouldDetect =
            mode === 'follow' &&
            workingTrack !== null &&
            (lastDetectionTime === null || frameData.timestamp - lastDetectionTime >= 0.2);
          if (shouldDetect) {
            const detectionSource = frameData.source instanceof HTMLCanvasElement
              ? frameData.source
              : trackingCanvas;

            if (detectionSource !== frameData.source && trackingCtx) {
              trackingCanvas.width = frameData.width;
              trackingCanvas.height = frameData.height;
              trackingCtx.drawImage(frameData.source, 0, 0, frameData.width, frameData.height);
            }

            const detector = detectorRef.current;
            if (!detector) {
              throw new Error('Person detector is unavailable.');
            }

            const { detectPeople } = await import('./lib/model');
            const detections = await detectPeople(detector, detectionSource);
            const candidates: TrackCandidate[] = detections.map((detection) => ({
              detection,
              signature: captureAppearanceSignature(frameData.source, detection.box, sourceFrame),
            }));
            const dt = lastDetectionTime === null ? 0 : Math.max(0, frameData.timestamp - lastDetectionTime);
            const activeTrack = workingTrack;
            if (!activeTrack) {
              return;
            }
            const association = associateTarget(activeTrack, candidates, { dt, frame: sourceFrame });
            workingTrack = advanceTrack(activeTrack, candidates, association, { dt, frame: sourceFrame });
            lastDetectionTime = frameData.timestamp;
          }

          const elapsed = lastDetectionTime === null
            ? 0
            : clamp(frameData.timestamp - lastDetectionTime, 0, MAX_RENDER_EXTRAPOLATION);
          const aim = workingTrack
            ? workingTrack.status === 'tracking'
              ? predictBox(workingTrack.box, workingTrack.velocity, elapsed, sourceFrame)
              : workingTrack.box
            : null;
          renderedBox = aim
            ? renderedBox
              ? smoothBox(renderedBox, aim, RENDER_SMOOTHING)
              : aim
            : null;

          fallbackCrop = drawExportFrame(
            ctx,
            frameData.source,
            sourceFrame,
            mode === 'fixed' ? null : renderedBox,
            fallbackCrop,
            cropAspectRef.current,
            outputSize,
            mode === 'fixed' ? fixedCrop : null,
          );
          drawBlurRegions(
            ctx,
            frameData.source,
            fallbackCrop,
            outputSize,
            blurRegionsRef.current,
            fastBlurCanvases,
            blurStrengthRef.current,
            frameData.timestamp,
          );
        },
        setExportedFraction,
        () => cancelFastExportRef.current,
      );

      fastExportActiveRef.current = false;
      setFastExportActive(false);

      if (!blob) {
        setExportPhase('idle');
        setExportedFraction(0);
        return true;
      }

      const url = URL.createObjectURL(blob);
      exportUrlRef.current = url;
      setExportResult({
        url,
        fileName: buildDownloadFileName(fileName, 'mp4'),
        size: blob.size,
      });
      setExportedFraction(1);
      setExportPhase('ready');
      return true;
    } catch {
      fastExportActiveRef.current = false;
      setFastExportActive(false);
      if (cancelFastExportRef.current) {
        cancelFastExportRef.current = false;
        setExportPhase('idle');
        setExportedFraction(0);
        return true;
      }
      cancelFastExportRef.current = false;
      setExportedFraction(0);
      return false;
    }
  }

  /**
   * Replays the clip from the start and records the crop canvas in real time, so the file the
   * user downloads is exactly what the preview shows.
   */
  async function handleExport() {
    const video = videoRef.current;
    const track = trackRef.current;

    if (!video || !videoMeta || (cropModeRef.current === 'follow' && !track)) {
      return;
    }

    releaseExportResult();
    discardExportRef.current = false;
    cancelFastExportRef.current = false;
    setError(null);
    setExportPhase('recording');
    video.pause();
    setIsPlaying(false);

    if (await tryFastExport()) {
      return;
    }

    if (typeof MediaRecorder === 'undefined') {
      setExportPhase('idle');
      setError('This browser cannot record video. Try a recent Chrome, Edge, or Safari.');
      return;
    }

    const format = pickRecordingFormat((mimeType) => MediaRecorder.isTypeSupported(mimeType));

    if (!format) {
      setExportPhase('idle');
      setError('This browser has no video format the recorder can write.');
      return;
    }

    setError(null);

    await seekVideo(video, 0);
    // Detect once before the tape rolls so the very first frames are already framed, and so the
    // canvas is sized from where the subject actually is at the start of the clip.
    if (cropModeRef.current === 'follow') {
      await runDetection();
    }

    const frame: FrameSize = { width: videoMeta.width, height: videoMeta.height };
    const openingCrop = cropModeRef.current === 'fixed'
      ? fixedCropRef.current ?? centerCropRect(frame, cropAspectRef.current)
      : computeCropRect(trackRef.current?.box ?? track!.box, frame, cropAspectRef.current);
    const size = resolveExportSize(openingCrop, cropAspectRef.current);
    const canvas = exportCanvasRef.current ?? document.createElement('canvas');
    canvas.width = size.width;
    canvas.height = size.height;
    exportCanvasRef.current = canvas;
    exportCropRef.current = null;

    try {
      const stream = canvas.captureStream(EXPORT_FRAME_RATE);
      captureAudioTracks(video).forEach((audioTrack) => {
        stream.addTrack(audioTrack);
      });

      const recorder = new MediaRecorder(stream, { mimeType: format.mimeType });
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: format.mimeType });
        chunksRef.current = [];

        if (discardExportRef.current) {
          discardExportRef.current = false;
          return;
        }

        if (blob.size === 0) {
          setExportPhase('idle');
          setError('Nothing was recorded. Try the export again.');
          return;
        }

        const url = URL.createObjectURL(blob);
        exportUrlRef.current = url;
        setExportResult({
          url,
          fileName: buildDownloadFileName(fileName, format.extension),
          size: blob.size,
        });
        setExportedFraction(1);
        setExportPhase('ready');
      };

      recorder.onerror = () => {
        recordingRef.current = false;
        recorderRef.current = null;
        setExportPhase('idle');
        setError('Recording stopped unexpectedly partway through the export.');
      };

      recorderRef.current = recorder;
      recordingRef.current = true;
      recorder.start(1000);

      await video.play();
      setIsPlaying(true);
    } catch {
      recordingRef.current = false;
      recorderRef.current = null;
      setExportPhase('idle');
      setError('Could not start recording for this video.');
    }
  }

  function handleStopExport() {
    videoRef.current?.pause();
    setIsPlaying(false);
    if (fastExportActiveRef.current) {
      cancelFastExportRef.current = true;
      setExportPhase('finishing');
      return;
    }
    finishExport();
  }

  function clearSelection() {
    trackRef.current = null;
    renderBoxRef.current = null;
    if (cropModeRef.current === 'follow') {
      cropAspectRef.current = DEFAULT_CROP_ASPECT_RATIO;
      setCropAspect(DEFAULT_CROP_ASPECT_RATIO);
    }
    const detections = detectionsRef.current;

    setTrackingSnapshot({
      phase: detections.length > 0 ? 'ready' : 'idle',
      confidence: 0,
      message:
        detections.length > 0
          ? 'Click a person in the frame to lock on.'
          : 'No person detected yet. Try a clearer frame.',
      detections: detections.length,
      targetId: null,
    });
  }

  function handleTimeUpdate() {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    setCurrentTime(video.currentTime);

    if (recordingRef.current) {
      setExportedFraction(exportProgress(video.currentTime, video.duration));
    }
  }

  function handleEnded() {
    setIsPlaying(false);
    finishExport();
  }

  function handleSeeked() {
    const video = videoRef.current;
    const track = trackRef.current;

    // A scrub is not motion the tracker can follow, so let it search the whole frame for the
    // same person at the new timestamp instead of trusting the old position.
    if (video && track && Math.abs(video.currentTime - lastTrackTimeRef.current) > 1) {
      trackRef.current = {
        ...track,
        status: 'lost',
        misses: LOST_AFTER_MISSES,
        secondsSinceMatch: 1.5,
        velocity: { x: 0, y: 0 },
      };
      renderBoxRef.current = null;
      lastTrackTimeRef.current = video.currentTime;
    }

    void runDetection();
  }

  return (
    <div className="app-shell">
      <header className="site-nav">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">Video Cropper</span>
        </div>

        {fileName ? (
          <label className="upload-button ghost">
            <span>Replace video</span>
            <input type="file" accept="video/*" onChange={handleFileChange} disabled={isExporting} />
          </label>
        ) : null}
      </header>

      <main className="workspace">
        <section className="hero">
          <p className="eyebrow">Video crop preview</p>
          <h1>Select a person and let the crop follow them.</h1>
          <p className="lede">
            Upload a video, click the person you care about, and the box stays on them for the rest
            of the clip: through crossing bystanders, and through short disappearances.
          </p>

          {fileName ? null : (
            <div className="hero-actions">
              <label className="upload-button">
                <span>Choose a video</span>
                <input type="file" accept="video/*" onChange={handleFileChange} />
              </label>
              <p className="hero-note">MP4, WebM, or MOV. Nothing leaves your browser.</p>
            </div>
          )}
        </section>

        {error ? <div className="error-banner">{error}</div> : null}

        <div className="content-grid">
          <article className="panel">
            <div className="panel-head">
              <div>
                <p className="panel-label">Source</p>
                <h2>{fileName ?? 'No video loaded'}</h2>
              </div>

              <div className={`status-pill status-${trackingSnapshot.phase}`}>
                {modelPhase === 'loading'
                  ? 'Loading model'
                  : modelPhase === 'error'
                    ? 'Model error'
                    : trackingSnapshot.phase === 'tracking'
                      ? `Following ${trackingSnapshot.targetId}`
                      : trackingSnapshot.phase === 'coasting'
                        ? 'Holding through occlusion'
                        : trackingSnapshot.phase === 'lost'
                          ? 'Searching for target'
                          : 'Ready'}
              </div>
            </div>

            <div
              className="video-stage"
              ref={sourceStage.ref}
              style={{ aspectRatio: sourceAspectRatio, minHeight: videoMeta ? 0 : undefined }}
            >
              {videoUrl ? (
                <>
                  <video
                    ref={videoRef}
                    className="source-video"
                    src={videoUrl}
                    playsInline
                    preload="metadata"
                    onLoadedMetadata={handleLoadedMetadata}
                    onPlay={() => {
                      setIsPlaying(true);
                    }}
                    onPause={() => {
                      setIsPlaying(false);
                    }}
                    onTimeUpdate={handleTimeUpdate}
                    onSeeked={handleSeeked}
                    onEnded={handleEnded}
                  />
                  <canvas
                    ref={overlayCanvasRef}
                    className={`overlay-canvas${cropMode === 'fixed' || blurEditing ? ' fixed-crop-canvas' : ''}`}
                    onClick={handleSourceClick}
                    onPointerDown={handleCropPointerDown}
                    onPointerMove={handleCropPointerMove}
                    onPointerUp={handleCropPointerUp}
                    onPointerCancel={handleCropPointerUp}
                  />
                </>
              ) : (
                <div className="empty-state">
                  <p>Upload a video to see detection boxes and click-to-select tracking.</p>
                </div>
              )}
            </div>

            <div className="blur-tools">
              <button
                type="button"
                className={`control-button secondary${blurEditing ? ' selected' : ''}`}
                onClick={toggleBlurEditing}
                aria-pressed={blurEditing}
                disabled={!videoMeta || isExporting}
              >
                {blurEditing ? 'Done adding blur' : 'Add blur area'}
              </button>
              <p className="blur-help">
                {blurEditing
                  ? 'Drag over an area. It starts at the playhead; set its HH:MM:SS range below.'
                  : `${blurRegions.length} blur area${blurRegions.length === 1 ? '' : 's'} · set each HH:MM:SS range`}
              </p>
              <label className="blur-strength-control">
                <span>Blur strength</span>
                <input
                  type="range"
                  min="4"
                  max="64"
                  step="1"
                  value={blurStrength}
                  onChange={handleBlurStrengthChange}
                  disabled={blurRegions.length === 0 || isExporting}
                  aria-label="Blur strength"
                />
                <span className="blur-strength-value">{blurStrength}</span>
              </label>
              <button
                type="button"
                className="text-button"
                onClick={clearBlurRegions}
                disabled={blurRegions.length === 0 || isExporting}
              >
                Clear areas
              </button>
            </div>

            {blurRegions.length > 0 ? (
              <div className="blur-region-list" aria-label="Blur area time ranges">
                {blurRegions.map((region, index) => (
                  <div className="blur-region-row" key={region.id}>
                    <span className="blur-region-name">Area {index + 1}</span>
                    <label>
                      From
                      <input
                        type="text"
                        inputMode="numeric"
                        placeholder="HH:MM:SS"
                        pattern="[0-9]{2,}:[0-5][0-9]:[0-5][0-9]"
                        value={blurTimeDrafts[blurTimeDraftKey(region.id, 'startTime')] ?? formatClockTime(region.startTime)}
                        onChange={(event) => handleBlurTimeTextChange(region.id, 'startTime', event.target.value)}
                        onBlur={() => finishBlurTimeTextEdit(region.id, 'startTime')}
                        disabled={isExporting}
                        aria-label={`Area ${index + 1} start time, HH:MM:SS`}
                      />
                    </label>
                    <label>
                      To
                      <input
                        type="text"
                        inputMode="numeric"
                        placeholder="HH:MM:SS"
                        pattern="[0-9]{2,}:[0-5][0-9]:[0-5][0-9]"
                        value={blurTimeDrafts[blurTimeDraftKey(region.id, 'endTime')] ?? formatClockTime(region.endTime)}
                        onChange={(event) => handleBlurTimeTextChange(region.id, 'endTime', event.target.value)}
                        onBlur={() => finishBlurTimeTextEdit(region.id, 'endTime')}
                        disabled={isExporting}
                        aria-label={`Area ${index + 1} end time, HH:MM:SS`}
                      />
                    </label>
                    <button
                      type="button"
                      className="text-button"
                      onClick={() => removeBlurRegion(index)}
                      disabled={isExporting}
                      aria-label={`Remove blur area ${index + 1}`}
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            ) : null}

            <div className="controls">
              <button
                type="button"
                className="control-button"
                onClick={handlePlayPause}
                disabled={!videoUrl || isExporting}
              >
                {isPlaying ? 'Pause' : 'Play'}
              </button>
              <button
                type="button"
                className="control-button secondary"
                onClick={clearSelection}
                disabled={!videoUrl || !trackingSnapshot.targetId || isExporting}
              >
                Pick someone else
              </button>
              <label className="seek-row">
                <span>Seek</span>
                <input
                  type="range"
                  min="0"
                  max={videoMeta?.duration ?? 0}
                  step="0.01"
                  value={currentTime}
                  onChange={handleSeek}
                  disabled={!videoMeta || isExporting}
                />
              </label>
              <div className="time-readout">
                {formatTime(currentTime)} / {formatTime(videoMeta?.duration ?? 0)}
              </div>
            </div>
          </article>

          <article className="panel preview-panel">
            <div className="panel-head">
              <div>
                <p className="panel-label">Crop preview</p>
                <h2>
                  {cropMode === 'fixed'
                    ? 'Fixed frame'
                    : trackingSnapshot.targetId
                      ? `Crop locked to ${trackingSnapshot.targetId}`
                      : 'Crop follows the selected person'}
                </h2>
              </div>
              <div className="confidence-chip">
                {trackingSnapshot.confidence > 0
                  ? `${Math.round(trackingSnapshot.confidence * 100)}%`
                  : 'Preview'}
              </div>
            </div>

            <div className="crop-mode-switch" role="group" aria-label="Crop mode">
              <button
                type="button"
                className={`mode-button${cropMode === 'follow' ? ' active' : ''}`}
                aria-pressed={cropMode === 'follow'}
                onClick={() => handleCropModeChange('follow')}
                disabled={isExporting}
              >
                Follow person
              </button>
              <button
                type="button"
                className={`mode-button${cropMode === 'fixed' ? ' active' : ''}`}
                aria-pressed={cropMode === 'fixed'}
                onClick={() => handleCropModeChange('fixed')}
                disabled={!videoMeta || isExporting}
              >
                Fixed frame
              </button>
            </div>

            <div className="preview-stage" ref={previewStage.ref} style={{ aspectRatio: `${cropAspect}` }}>
              <canvas ref={previewCanvasRef} className="preview-canvas" />
            </div>

            <p className="status-copy">
              {cropMode === 'fixed'
                ? 'Drag on the source video to set the crop. This frame stays in the same place throughout the clip.'
                : trackingSnapshot.message}
            </p>
            <p className="status-meta">
              {trackingSnapshot.detections} person
              {trackingSnapshot.detections === 1 ? '' : 's'} seen by the last model pass
            </p>

            <div className="export-block">
              {isExporting ? (
                <>
                  <div className="export-actions">
                    <button type="button" className="control-button secondary" onClick={handleStopExport}>
                      {fastExportActive ? 'Cancel export' : 'Stop and keep'}
                    </button>
                    <span className="export-count">{exportPercent}%</span>
                  </div>
                  <div
                    className="export-progress"
                    role="progressbar"
                    aria-label="Export progress"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={exportPercent}
                  >
                    <span style={{ width: `${exportPercent}%` }} />
                  </div>
                  <p className="export-note">
                    {exportPhase === 'finishing'
                      ? fastExportActive ? 'Canceling the render.' : 'Wrapping up the file.'
                      : fastExportActive
                        ? 'Rendering the complete video with the current crop and blur edits.'
                        : 'Recording the crop while the clip plays. Keep this tab in front.'}
                  </p>
                </>
              ) : (
                <>
                  <div className="export-actions">
                    <button
                      type="button"
                      className="control-button"
                      onClick={() => {
                        void handleExport();
                      }}
                      disabled={!canExport}
                    >
                      {exportResult ? 'Export again' : 'Export cropped video'}
                    </button>

                    {exportResult ? (
                      <a
                        className="control-button secondary download-link"
                        href={exportResult.url}
                        download={exportResult.fileName}
                      >
                        Download · {formatFileSize(exportResult.size)}
                      </a>
                    ) : null}
                  </div>

                  <p className="export-note">
                    {exportResult
                      ? `${exportResult.fileName} is ready to save.`
                      : canExport
                        ? 'Renders the complete clip with its audio, crop, and blur edits.'
                        : 'Pick a person first, then the cropped video can be exported.'}
                  </p>
                </>
              )}
            </div>
          </article>
        </div>
      </main>

      <footer className="site-foot">
        Detection and tracking run locally in your browser with TensorFlow.js. Videos are never
        uploaded.
      </footer>
    </div>
  );
}
