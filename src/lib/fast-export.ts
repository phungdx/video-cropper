import {
  ALL_FORMATS,
  AudioSampleSink,
  AudioSampleSource,
  BlobSource,
  BufferTarget,
  CanvasSink,
  CanvasSource,
  Input,
  Mp4OutputFormat,
  Output,
  Quality,
} from 'mediabunny';

export type FastExportFrame = {
  source: CanvasImageSource;
  timestamp: number;
  duration: number;
  width: number;
  height: number;
};

/**
 * Decodes, renders, and encodes the source frames directly. The output timeline is advanced by
 * frame timestamps, so it does not have to wait for the source video to play in real time.
 */
export async function exportWholeVideo(
  file: Blob,
  canvas: HTMLCanvasElement,
  onFrame: (frame: FastExportFrame) => Promise<void>,
  onProgress: (fraction: number) => void,
  shouldCancel: () => boolean,
): Promise<Blob | null> {
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  let output: Output | null = null;

  try {
    const videoTrack = await input.getPrimaryVideoTrack();
    if (!videoTrack) {
      throw new Error('The source has no video track.');
    }

    const audioTrack = await input.getPrimaryAudioTrack();
    const target = new BufferTarget();
    output = new Output({ format: new Mp4OutputFormat(), target });

    const videoSource = new CanvasSource(canvas, {
      codec: 'avc',
      quality: new Quality('high'),
    });
    output.addVideoTrack(videoSource);

    const audioSource = audioTrack
      ? new AudioSampleSource({ codec: 'aac', quality: new Quality('high') })
      : null;
    if (audioSource) {
      output.addAudioTrack(audioSource);
    }

    await output.start();

    const videoStart = await videoTrack.getFirstTimestamp();
    const videoDuration = await videoTrack.computeDuration();
    const frameSink = new CanvasSink(videoTrack, { poolSize: 1 });
    const frames = frameSink.canvases();
    const audioSamples = audioTrack ? new AudioSampleSink(audioTrack).samples() : null;
    let nextAudio = audioSamples ? await audioSamples.next() : null;
    let lastReportedProgress = 0;

    async function addAudioThrough(timestamp: number) {
      if (!audioSource || !audioSamples || !nextAudio) {
        return;
      }

      while (!nextAudio.done && nextAudio.value.timestamp <= timestamp) {
        const sample = nextAudio.value;
        await audioSource.add(sample);
        sample.close();
        nextAudio = await audioSamples.next();
      }
    }

    for await (const frame of frames) {
      if (shouldCancel()) {
        await output.cancel();
        return null;
      }

      const timestamp = Math.max(0, frame.timestamp - videoStart);
      const duration = Math.max(frame.duration, 1 / 120);
      await onFrame({
        source: frame.canvas,
        timestamp,
        duration,
        width: frame.canvas.width,
        height: frame.canvas.height,
      });

      if (audioTrack) {
        await addAudioThrough(frame.timestamp);
      }

      await videoSource.add(timestamp, duration);
      const progress = videoDuration > 0 ? Math.min(1, timestamp / videoDuration) : 0;
      if (progress - lastReportedProgress >= 0.01) {
        lastReportedProgress = progress;
        onProgress(progress);
      }
    }

    if (audioSource && audioSamples && nextAudio) {
      while (!nextAudio.done) {
        const sample = nextAudio.value;
        await audioSource.add(sample);
        sample.close();
        nextAudio = await audioSamples.next();
      }
    }

    if (shouldCancel()) {
      await output.cancel();
      return null;
    }

    await output.finalize();
    onProgress(1);

    if (!target.buffer) {
      throw new Error('The video encoder returned an empty file.');
    }

    return new Blob([target.buffer], { type: 'video/mp4' });
  } catch (error) {
    if (output && output.state !== 'finalized' && output.state !== 'canceled') {
      await output.cancel().catch(() => undefined);
    }
    throw error;
  } finally {
    input.dispose();
  }
}
