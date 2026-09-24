import type { LayerScore } from "../types";

/**
 * One card per detection layer.
 *
 * Layers not in the MVP (L2 voiceprint, L3 diarisation) are rendered greyed
 * out and labelled "roadmap" rather than hidden. Showing the intended
 * architecture honestly is stronger than implying three layers are running
 * when one is -- and a judge who asks will get a straight answer.
 *
 * An abstaining layer shows "abstained", never a number. A layer that cannot
 * tell must not look confident.
 */
export function LayerCards(_props: { layers: LayerScore[]; detectorName: string }) {
  throw new Error("not implemented");
}
