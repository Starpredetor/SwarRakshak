/**
 * Downsamples mic audio to 16 kHz mono int16 and posts it to the main thread.
 *
 * An AudioWorklet rather than MediaRecorder: MediaRecorder hands back
 * webm/opus, which would need decoding server-side and adds latency you cannot
 * predict. Here the raw Float32 is already in hand, so the conversion is a few
 * lines and the timing is yours.
 *
 * Runs on the audio render thread. Allocating inside process() causes audible
 * glitches and dropped quanta, so every buffer is preallocated.
 */
const CHUNK_MS = 100;

class PCMWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = options?.processorOptions?.targetRate ?? 16000;
    // `sampleRate` is a global in the AudioWorklet scope -- the context's real
    // rate, typically 44100 or 48000.
    this.ratio = sampleRate / this.targetRate;

    this.chunkSamples = Math.round((this.targetRate * CHUNK_MS) / 1000);
    this.out = new Int16Array(this.chunkSamples);
    this.outIndex = 0;

    // Fractional read position, carried across render quanta. Resetting it
    //every quantum would drift the pitch; carrying the remainder does not.
    this.position = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel || channel.length === 0) return true;

    // Linear interpolation. Good enough at 48k -> 16k; a polyphase filter
    // would be better and is not worth it for the MVP.
    while (this.position < channel.length) {
      const i = Math.floor(this.position);
      const frac = this.position - i;
      const a = channel[i];
      const b = i + 1 < channel.length ? channel[i + 1] : a;
      const sample = Math.max(-1, Math.min(1, a + (b - a) * frac));

      this.out[this.outIndex++] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;

      if (this.outIndex === this.chunkSamples) {
        // slice() copies -- the buffer is reused on the next chunk, so it
        // cannot be transferred.
        this.port.postMessage(this.out.buffer.slice(0));
        this.outIndex = 0;
      }

      this.position += this.ratio;
    }

    this.position -= channel.length;
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
