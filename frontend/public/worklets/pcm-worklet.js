/**
 * Downsamples mic audio to 16 kHz mono int16 and posts it to the main thread.
 *
 * An AudioWorklet rather than MediaRecorder: MediaRecorder hands back
 * webm/opus, which would need decoding server-side and adds latency you cannot
 * predict. Here the raw Float32 is already in hand, so the conversion is a few
 * lines and the timing is yours.
 *
 * Runs on the audio render thread. Allocating inside process() causes audible
 * glitches -- keep buffers preallocated.
 */
class PCMWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = options?.processorOptions?.targetRate ?? 16000;
    // TODO: preallocate the resample accumulator sized to one output chunk.
  }

  /**
   * Linear-interpolation resample from `sampleRate` to `this.targetRate`,
   * then float -> int16. Good enough at these ratios; a polyphase filter
   * would be better and is not worth it for the MVP.
   */
  process(inputs) {
    // TODO: resample inputs[0][0], convert to Int16, this.port.postMessage(buf)
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
