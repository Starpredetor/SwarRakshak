import type { Verdict } from "../types";

/**
 * Append-only log of band changes, with timestamps.
 *
 * Only band *changes*, not every window -- a line per second scrolls too fast
 * to read back on video.
 */
export function EventLog(_props: { verdicts: Verdict[] }) {
  throw new Error("not implemented");
}
