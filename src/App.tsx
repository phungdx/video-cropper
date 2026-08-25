import { useEffect, useRef, useState } from 'react';
import { computeCropRect, type Box, type FrameSize, type Point } from './lib/geometry';
import {
  choosePersonForClick,
  matchDetectionToPrevious,
  smoothBox,
  type Detection,
} from './lib/tracking';
import type { ObjectDetection } from '@tensorflow-models/coco-ssd';

const PREVIEW_ASPECT_RATIO = 9 / 16;
const DETECTION_INTERVAL_MS = 260;
const LOST_THRESHOLD = 3;

type ModelPhase = 'idle' | 'loading' | 'ready' | 'error';
type TrackingPhase = 'idle' | 'ready' | 'tracking' | 'reacquiring' | 'lost';

type TrackingSnapshot = {
  phase: TrackingPhase;
  confidence: number;
  message: string;
  detections: number;
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
};

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
  ctx.fillStyle = 'rgba(7, 10, 20, 0.76)';
  drawRoundedRect(ctx, centerX - 220, centerY - 60, 440, 120, 24);
  ctx.fill();

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = Math.max(1.5, canvasSize.width / 320);
  ctx.stroke();

  ctx.fillStyle = '#eef4ff';
  ctx.font = '700 26px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(message, centerX, centerY - 8);

  if (subtitle) {
    ctx.fillStyle = 'rgba(238, 244, 255, 0.72)';
    ctx.font = '500 16px Inter, system-ui, sans-serif';
    ctx.fillText(subtitle, centerX, centerY + 24);
  }

  ctx.restore();
}

function drawDetectionBox(
  ctx: CanvasRenderingContext2D,
  detection: Detection,
  scaleX: number,
  scaleY: number,
  highlight: boolean,
) {
  const { box, score } = detection;
  const x = box.x * scaleX;
  const y = box.y * scaleY;
  const width = box.width * scaleX;
  const height = box.height * scaleY;
  const borderColor = highlight ? '#61f0d2' : 'rgba(140, 176, 255, 0.88)';

  ctx.save();
  ctx.lineWidth = highlight ? 4 : 2.5;
  ctx.strokeStyle = borderColor;
  ctx.fillStyle = highlight ? 'rgba(97, 240, 210, 0.16)' : 'rgba(140, 176, 255, 0.08)';
  drawRoundedRect(ctx, x, y, width, height, 16);
  ctx.fill();
  ctx.stroke();

  const label = `${detection.label} ${Math.round(score * 100)}%`;
  ctx.font = '700 15px Inter, system-ui, sans-serif';
  const labelWidth = ctx.measureText(label).width + 20;
  const labelHeight = 30;
  const labelX = x;
  const labelY = Math.max(8, y - labelHeight - 10);

  ctx.fillStyle = borderColor;
  drawRoundedRect(ctx, labelX, labelY, labelWidth, labelHeight, 12);
  ctx.fill();

  ctx.fillStyle = '#07111c';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, labelX + 10, labelY + labelHeight / 2);
  ctx.restore();
}

function drawSourceOverlay(
  ctx: CanvasRenderingContext2D,
  canvasSize: CanvasSize,
  videoMeta: VideoMeta | null,
  detections: Detection[],
  trackedBox: Box | null,
  tracking: TrackingSnapshot,
) {
  ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

  if (!videoMeta) {
    drawBanner(ctx, 'Upload a video', canvasSize, 'The browser preview and tracking tools will appear here.');
    return;
  }

  const frame: FrameSize = { width: videoMeta.width, height: videoMeta.height };
  const scaleX = canvasSize.width / frame.width;
  const scaleY = canvasSize.height / frame.height;

  if (trackedBox) {
    const selectedDetection = detections.reduce<Detection | null>((best, detection) => {
      if (!best) {
        return detection;
      }

      const bestScore = Math.abs(best.box.x - trackedBox.x) + Math.abs(best.box.y - trackedBox.y);
      const candidateScore =
        Math.abs(detection.box.x - trackedBox.x) + Math.abs(detection.box.y - trackedBox.y);

      return candidateScore < bestScore ? detection : best;
    }, null);

    if (selectedDetection) {
      drawDetectionBox(ctx, selectedDetection, scaleX, scaleY, true);
    } else {
      drawDetectionBox(
        ctx,
        {
          label: 'person',
          score: 1,
          box: trackedBox,
        },
        scaleX,
        scaleY,
        true,
      );
    }

    ctx.save();
    ctx.strokeStyle = '#61f0d2';
    ctx.lineWidth = 5;
    ctx.setLineDash([10, 8]);
    ctx.strokeRect(
      trackedBox.x * scaleX,
      trackedBox.y * scaleY,
      trackedBox.width * scaleX,
      trackedBox.height * scaleY,
    );
    ctx.restore();
  } else {
    drawBanner(
      ctx,
      detections.length > 0 ? 'Click one person' : 'Waiting for a person',
      canvasSize,
      detections.length > 0
        ? 'Click the target you want to follow. The other people stay out of the preview.'
        : 'Pause on a clear frame, then click the person to lock on.',
    );
  }

  if (tracking.phase === 'reacquiring' || tracking.phase === 'lost') {
    ctx.save();
    ctx.fillStyle = tracking.phase === 'lost' ? 'rgba(255, 103, 103, 0.14)' : 'rgba(255, 196, 87, 0.14)';
    drawRoundedRect(ctx, 24, 24, 220, 48, 18);
    ctx.fill();
    ctx.strokeStyle = tracking.phase === 'lost' ? '#ff7979' : '#ffc457';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#eef4ff';
    ctx.font = '700 16px Inter, system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.fillText(tracking.phase === 'lost' ? 'Target lost' : 'Reacquiring', 44, 48);
    ctx.restore();
  }
}

function drawPreviewFrame(
  ctx: CanvasRenderingContext2D,
  canvasSize: CanvasSize,
  video: HTMLVideoElement | null,
  videoMeta: VideoMeta | null,
  trackedBox: Box | null,
) {
  ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

  if (!video || !videoMeta || !trackedBox || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    drawBanner(ctx, 'Preview crop', canvasSize, 'Select a person to see the adaptive crop follow them.');
    return;
  }

  const crop = computeCropRect(trackedBox, { width: videoMeta.width, height: videoMeta.height }, PREVIEW_ASPECT_RATIO);
  ctx.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, canvasSize.width, canvasSize.height);

  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
  ctx.lineWidth = Math.max(2, canvasSize.width / 280);
  ctx.strokeRect(12, 12, canvasSize.width - 24, canvasSize.height - 24);
  ctx.strokeStyle = 'rgba(97, 240, 210, 0.85)';
  ctx.lineWidth = Math.max(1.5, canvasSize.width / 420);
  ctx.beginPath();
  ctx.moveTo(canvasSize.width / 2, canvasSize.height * 0.5 - 22);
  ctx.lineTo(canvasSize.width / 2, canvasSize.height * 0.5 + 22);
  ctx.moveTo(canvasSize.width / 2 - 22, canvasSize.height * 0.5);
  ctx.lineTo(canvasSize.width / 2 + 22, canvasSize.height * 0.5);
  ctx.stroke();
  ctx.restore();
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
  const trackedBoxRef = useRef<Box | null>(null);
  const trackingSnapshotRef = useRef<TrackingSnapshot>(defaultTrackingSnapshot);
  const detectionInFlightRef = useRef(false);
  const lostCountRef = useRef(0);
  const videoUrlRef = useRef<string | null>(null);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [videoMeta, setVideoMeta] = useState<VideoMeta | null>(null);
  const [modelPhase, setModelPhase] = useState<ModelPhase>('idle');
  const [trackingSnapshot, setTrackingSnapshot] = useState<TrackingSnapshot>(defaultTrackingSnapshot);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    trackingSnapshotRef.current = trackingSnapshot;
  }, [trackingSnapshot]);

  useEffect(() => {
    return () => {
      if (videoUrlRef.current) {
        URL.revokeObjectURL(videoUrlRef.current);
      }
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

    if (!video || !detector || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
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

      if (!trackedBoxRef.current) {
        const nextTracking: TrackingSnapshot = {
          phase: detections.length > 0 ? 'ready' : 'idle',
          confidence: 0,
          message:
            detections.length > 0
              ? 'Click a person in the frame to lock on.'
              : 'No person detected yet. Try a clearer frame.',
          detections: detections.length,
        };
        setTrackingSnapshot(nextTracking);
        return true;
      }

      const matched = matchDetectionToPrevious(detections, trackedBoxRef.current);

      if (matched) {
        trackedBoxRef.current = smoothBox(trackedBoxRef.current, matched.box, 0.35);
        lostCountRef.current = 0;
        setTrackingSnapshot({
          phase: 'tracking',
          confidence: matched.score,
          message: `Tracking person (${Math.round(matched.score * 100)}%)`,
          detections: detections.length,
        });
      } else {
        lostCountRef.current += 1;
        const phase = lostCountRef.current >= LOST_THRESHOLD ? 'lost' : 'reacquiring';

        setTrackingSnapshot({
          phase,
          confidence: 0,
          message:
            phase === 'lost'
              ? 'Lost the person. Click again to reselect.'
              : 'Reacquiring the person…',
          detections: detections.length,
        });
      }

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

    const render = () => {
      const sourceCanvas = overlayCanvasRef.current;
      const previewCanvas = previewCanvasRef.current;
      const video = videoRef.current;

      if (sourceCanvas) {
        resizeCanvas(sourceCanvas, sourceStage.size);
        const ctx = sourceCanvas.getContext('2d');
        if (ctx) {
          drawSourceOverlay(
            ctx,
            sourceStage.size,
            videoMeta,
            detectionsRef.current,
            trackedBoxRef.current,
            trackingSnapshotRef.current,
          );
        }
      }

      if (previewCanvas) {
        resizeCanvas(previewCanvas, previewStage.size);
        const ctx = previewCanvas.getContext('2d');
        if (ctx) {
          drawPreviewFrame(ctx, previewStage.size, video, videoMeta, trackedBoxRef.current);
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

    const timer = window.setInterval(() => {
      void runDetection();
    }, DETECTION_INTERVAL_MS);

    return () => window.clearInterval(timer);
  }, [isPlaying, modelPhase, videoMeta]);

  const sourceAspectRatio = videoMeta ? `${videoMeta.width} / ${videoMeta.height}` : '16 / 9';

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (videoUrlRef.current) {
      URL.revokeObjectURL(videoUrlRef.current);
    }

    const nextUrl = URL.createObjectURL(file);
    videoUrlRef.current = nextUrl;
    setVideoUrl(nextUrl);
    setFileName(file.name);
    setVideoMeta(null);
    setCurrentTime(0);
    setIsPlaying(false);
    setError(null);
    trackedBoxRef.current = null;
    detectionsRef.current = [];
    lostCountRef.current = 0;
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
    const detections = detectionsRef.current;
    let candidate = choosePersonForClick(detections, point);

    if (!candidate) {
      void runDetection().then(() => {
        const refreshedCandidate = choosePersonForClick(detectionsRef.current, point);

        if (refreshedCandidate) {
          trackedBoxRef.current = refreshedCandidate.box;
          lostCountRef.current = 0;
          setTrackingSnapshot({
            phase: 'tracking',
            confidence: refreshedCandidate.score,
            message: 'Tracking selected person.',
            detections: detectionsRef.current.length,
          });
          setError(null);
        } else {
          setTrackingSnapshot({
            phase: 'idle',
            confidence: 0,
            message: 'No person found at that point. Try a different frame or person.',
            detections: detectionsRef.current.length,
          });
        }
      });
      return;
    }

    trackedBoxRef.current = candidate.box;
    lostCountRef.current = 0;
    setTrackingSnapshot({
      phase: 'tracking',
      confidence: candidate.score,
      message: 'Tracking selected person.',
      detections: detectionsRef.current.length,
    });
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

  function clearSelection() {
    trackedBoxRef.current = null;
    lostCountRef.current = 0;
    const detections = detectionsRef.current;

    setTrackingSnapshot({
      phase: detections.length > 0 ? 'ready' : 'idle',
      confidence: 0,
      message:
        detections.length > 0
          ? 'Click a person in the frame to lock on.'
          : 'No person detected yet. Try a clearer frame.',
      detections: detections.length,
    });
  }

  function handleTimeUpdate() {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    setCurrentTime(video.currentTime);
  }

  function handleSeeked() {
    void runDetection();
  }

  return (
    <main className="app-shell">
      <section className="hero-card workspace">
        <header className="hero-header">
          <div>
            <p className="eyebrow">Video crop preview</p>
            <h1>Select a person and let the crop follow them.</h1>
            <p className="lede">
              Upload a video, click the person you care about, and watch the preview crop adapt as
              they move.
            </p>
          </div>

          <label className="upload-button">
            <span>{fileName ? 'Replace video' : 'Choose a video'}</span>
            <input type="file" accept="video/*" onChange={handleFileChange} />
          </label>
        </header>

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
                      ? 'Tracking'
                      : trackingSnapshot.phase === 'reacquiring'
                        ? 'Reacquiring'
                        : trackingSnapshot.phase === 'lost'
                          ? 'Lost'
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
                    onEnded={() => {
                      setIsPlaying(false);
                    }}
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
              <button type="button" className="control-button" onClick={handlePlayPause} disabled={!videoUrl}>
                {isPlaying ? 'Pause' : 'Play'}
              </button>
              <button type="button" className="control-button secondary" onClick={clearSelection} disabled={!videoUrl}>
                Clear selection
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
                  disabled={!videoMeta}
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
                <h2>Crop follows the selected person</h2>
              </div>
              <div className="confidence-chip">
                {trackingSnapshot.confidence > 0
                  ? `${Math.round(trackingSnapshot.confidence * 100)}%`
                  : 'Preview'}
              </div>
            </div>

            <div className="preview-stage" ref={previewStage.ref} style={{ aspectRatio: `${9 / 16}` }}>
              <canvas ref={previewCanvasRef} className="preview-canvas" />
            </div>

            <p className="status-copy">{trackingSnapshot.message}</p>
            <p className="status-meta">
              {trackingSnapshot.detections} person
              {trackingSnapshot.detections === 1 ? '' : 's'} seen by the last model pass
            </p>
          </article>
        </div>
      </section>
    </main>
  );
}
