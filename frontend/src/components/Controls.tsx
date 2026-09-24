import type { ClipInfo } from "../types";

/**
 * Source selection: live microphone, or replay a bundled clip.
 *
 * Both feed the same socket. The replay control is the reliable take -- if the
 * room is noisy or the mic misbehaves while recording, you still have a
 * reproducible run to cut to.
 */
export function Controls(_props: {
  micActive: boolean;
  onToggleMic: () => void;
  clips: ClipInfo[];
  onReplay: (filename: string) => void;
  status: "connecting" | "open" | "closed";
  detectorIsFallback: boolean;
}) {
  throw new Error("not implemented");
}
