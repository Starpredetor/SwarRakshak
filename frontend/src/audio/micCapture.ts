/**
 * Microphone capture: getUserMedia -> AudioWorklet -> int16 PCM callback.
 *
 * Note on browser audio processing: echoCancellation and noiseSuppression are
 * requested OFF. They are DSP that rewrites the signal, and the artifacts this
 * system looks for live in exactly the bands they touch -- leaving them on
 * means the browser partially launders the clone before we ever see it.
 */
export interface MicHandle {
  stop: () => void;
}

export async function startMicCapture(
  _onPCM: (buf: ArrayBuffer) => void,
  _targetRate = 16000,
): Promise<MicHandle> {
  // TODO:
  //   1. getUserMedia({ audio: { echoCancellation: false,
  //                              noiseSuppression: false,
  //                              autoGainControl: false } })
  //   2. new AudioContext(), addModule("/worklets/pcm-worklet.js")
  //   3. wire source -> worklet, port.onmessage -> onPCM
  //   4. return a stop() that tears down the track and the context
  throw new Error("not implemented");
}
