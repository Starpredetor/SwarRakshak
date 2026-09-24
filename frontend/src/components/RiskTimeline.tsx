import type { Verdict } from "../types";

/**
 * Rolling risk-over-time chart with the band thresholds drawn in.
 *
 * This is what makes a mid-call voice swap legible: a single number cannot
 * show that the line was flat for thirty seconds and then climbed.
 */
export function RiskTimeline(_props: { verdicts: Verdict[]; windowSeconds?: number }) {
  throw new Error("not implemented");
}
