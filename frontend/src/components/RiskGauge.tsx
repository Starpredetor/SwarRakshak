import type { Band } from "../types";

/**
 * The hero element: a 0-100 arc that is the whole story in one glance.
 *
 * Colour comes from the band, not from the raw number, so the gauge never
 * shows an amber tint while the label reads LOW. Transitions are eased over
 * roughly the hop interval -- an instant jump reads as a glitch on video.
 */
export function RiskGauge(_props: { risk: number; band: Band; action: string }) {
  throw new Error("not implemented");
}
