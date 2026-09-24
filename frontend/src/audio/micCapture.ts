/**
 * Microphone capture: getUserMedia -> AudioWorklet -> int16 PCM callback.
 *
 * Note on browser audio processing: echoCancellation, noiseSuppression and
 * autoGainControl are all requested OFF. They are DSP that rewrites the
 * signal, and the artifacts this system looks for live in exactly the bands
 * they touch -- leaving them on means the browser partially launders the clone
 * before we ever see it.
 */
export interface MicHandle {
  stop: () => void;
  contextRate: number;
}

export async function startMicCapture(
  onPCM: (buf: ArrayBuffer) => void,
  targetRate = 16000,
): Promise<MicHandle> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  });

  const context = new AudioContext();
  // Chrome starts contexts suspended until a user gesture. The caller is a
  // click handler, so this resolves immediately -- but without it the worklet
  // silently never runs.
  if (context.state === "suspended") await context.resume();

  await context.audioWorklet.addModule("/worklets/pcm-worklet.js");

  const source = context.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(context, "pcm-worklet", {
    processorOptions: { targetRate },
  });
  worklet.port.onmessage = (event) => onPCM(event.data as ArrayBuffer);

  // A worklet is only pulled if it reaches the destination, but routing the
  // mic to the speakers would feed back through the room. Zero gain keeps the
  // graph alive and silent.
  const mute = context.createGain();
  mute.gain.value = 0;

  source.connect(worklet);
  worklet.connect(mute);
  mute.connect(context.destination);

  return {
    contextRate: context.sampleRate,
    stop: () => {
      worklet.port.onmessage = null;
      worklet.disconnect();
      source.disconnect();
      mute.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      void context.close();
    },
  };
}
