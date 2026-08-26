import { useEffect, useRef, useState } from 'react';
import {
  boxAspectRatio,
  clamp,
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
) {
  ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

  if (!video || !videoMeta || !trackedBox || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    drawBanner(ctx, 'Preview crop', canvasSize, 'Select a person to see the adaptive crop follow them.');
    return;
  }

  const crop = computeCropRect(
    trackedBox,
    { width: videoMeta.width, height: videoMeta.height },
    aspectRatio,
  );

  ctx.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, canvasSize.width, canvasSize.height);
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
  const lastTrackTimeRef = useRef(0);
  const detectionInFlightRef = useRef(false);
  const nextTargetIdRef = useRef(1);
  const videoUrlRef = useRef<string | null>(null);
  const exportCanvasRef = useRef<HTMLCanvasElement | null>(null);
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
  const [modelPhase, setModelPhase] = useState<ModelPhase>('idle');
  const [trackingSnapshot, setTrackingSnapshot] = useState<TrackingSnapshot>(defaultTrackingSnapshot);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exportPhase, setExportPhase] = useState<ExportPhase>('idle');
  const [exportedFraction, setExportedFraction] = useState(0);
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);

  useEffect(() => {
    return () => {
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
          drawSourceOverlay(
            ctx,
            sourceStage.size,
            videoMeta,
            detectionsRef.current,
            track,
            renderBoxRef.current,
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
            renderBoxRef.current,
            cropAspectRef.current,
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
  const canExport = Boolean(videoMeta) && Boolean(trackingSnapshot.targetId) && modelPhase === 'ready';
  const exportPercent = Math.round(exportedFraction * 100);

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (videoUrlRef.current) {
      URL.revokeObjectURL(videoUrlRef.current);
    }

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

  /**
   * Replays the clip from the start and records the crop canvas in real time, so the file the
   * user downloads is exactly what the preview shows.
   */
  async function handleExport() {
    const video = videoRef.current;
    const track = trackRef.current;

    if (!video || !videoMeta || !track) {
      return;
    }

    if (typeof MediaRecorder === 'undefined') {
      setError('This browser cannot record video. Try a recent Chrome, Edge, or Safari.');
      return;
    }

    const format = pickRecordingFormat((mimeType) => MediaRecorder.isTypeSupported(mimeType));

    if (!format) {
      setError('This browser has no video format the recorder can write.');
      return;
    }

    releaseExportResult();
    discardExportRef.current = false;
    setError(null);
    setExportPhase('recording');

    video.pause();
    setIsPlaying(false);
    await seekVideo(video, 0);
    // Detect once before the tape rolls so the very first frames are already framed, and so the
    // canvas is sized from where the subject actually is at the start of the clip.
    await runDetection();

    const frame: FrameSize = { width: videoMeta.width, height: videoMeta.height };
    const openingCrop = computeCropRect(
      trackRef.current?.box ?? track.box,
      frame,
      cropAspectRef.current,
    );
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
    finishExport();
  }

  function clearSelection() {
    trackRef.current = null;
    renderBoxRef.current = null;
    cropAspectRef.current = DEFAULT_CROP_ASPECT_RATIO;
    setCropAspect(DEFAULT_CROP_ASPECT_RATIO);
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

            <div className="video-stage" ref={sourceStage.ref} style={{ aspectRatio: sourceAspectRatio }}>
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
                    className="overlay-canvas"
                    onClick={handleSourceClick}
                  />
                </>
              ) : (
                <div className="empty-state">
                  <p>Upload a video to see detection boxes and click-to-select tracking.</p>
                </div>
              )}
            </div>

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
                <p className="panel-label">Adaptive preview</p>
                <h2>
                  {trackingSnapshot.targetId
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

            <div className="preview-stage" ref={previewStage.ref} style={{ aspectRatio: `${cropAspect}` }}>
              <canvas ref={previewCanvasRef} className="preview-canvas" />
            </div>

            <p className="status-copy">{trackingSnapshot.message}</p>
            <p className="status-meta">
              {trackingSnapshot.detections} person
              {trackingSnapshot.detections === 1 ? '' : 's'} seen by the last model pass
            </p>

            <div className="export-block">
              {isExporting ? (
                <>
                  <div className="export-actions">
                    <button type="button" className="control-button secondary" onClick={handleStopExport}>
                      Stop and keep
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
                      ? 'Wrapping up the file.'
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
                        ? 'Replays the clip once from the start and records the crop with its audio.'
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
