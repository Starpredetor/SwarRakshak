import type { LayerScore } from "../types";

interface LayerMeta {
  key: string;
  title: string;
  blurb: string;
}

/** Layers in the MVP, in the order the pitch describes them. */
const ACTIVE: LayerMeta[] = [
  {
    key: "l1_spoof",
    title: "L1 · Vocoder artifacts",
    blurb: "Trained anti-spoof model. Language-agnostic.",
  },
  {
    key: "l1_fallback",
    title: "L1 · Offline fallback",
    blurb: "LFCC baseline. Materially weaker than the trained layer.",
  },
  {
    key: "l1b_dsp",
    title: "L1b · DSP evidence",
    blurb: "Heuristic, low weight. Explains the score.",
  },
];

/**
 * Layers on the roadmap. Rendered greyed out rather than hidden.
 *
 * Showing the intended architecture honestly is stronger than implying three
 * layers run when one does. A judge who asks gets a straight answer, and the
 * roadmap reads as a plan rather than as a gap.
 */
const ROADMAP: LayerMeta[] = [
  {
    key: "l2_speaker",
    title: "L2 · Voice biometrics",
    blurb: "ECAPA-TDNN against an enrolled voiceprint. Not in the MVP.",
  },
  {
    key: "l3_diarisation",
    title: "L3 · Speaker change",
    blurb: "Catches a voice swap after authentication. Not in the MVP.",
  },
];

export function LayerCards({
  layers,
  detectorName,
}: {
  layers: LayerScore[];
  detectorName: string;
}) {
  const byKey = new Map(layers.map((layer) => [layer.layer, layer]));

  // Before the first verdict there are no scores, but the active layers must
  // still be visible -- a panel showing only the greyed-out roadmap reads as
  // though nothing is running at all.
  const activeKeys = layers.length
    ? ACTIVE.filter((meta) => byKey.has(meta.key))
    : ACTIVE.filter((meta) => meta.key === detectorName || meta.key === "l1b_dsp");

  return (
    <div className="panel">
      <h2 className="panel__title">Detection layers</h2>
      <div className="layers">
        {activeKeys.map((meta) => {
          const layer = byKey.get(meta.key);
          return (
            <div key={meta.key} className="layer-card">
              <div className="layer-card__title">{meta.title}</div>
              <div className="layer-card__score">
                {/* An abstaining layer shows "abstained", never a number. A
                    layer that cannot tell must not look confident. */}
                {!layer ? (
                  <span className="layer-card__abstain">idle</span>
                ) : layer.abstain ? (
                  <span className="layer-card__abstain">abstained</span>
                ) : (
                  (layer.score * 100).toFixed(0)
                )}
              </div>
              <div className="layer-card__blurb">{meta.blurb}</div>
              {layer && !layer.abstain && (
                <div className="layer-card__confidence">
                  confidence {(layer.confidence * 100).toFixed(0)}%
                </div>
              )}
            </div>
          );
        })}

        {ROADMAP.map((meta) => (
          <div key={meta.key} className="layer-card layer-card--roadmap">
            <div className="layer-card__title">{meta.title}</div>
            <div className="layer-card__score">—</div>
            <div className="layer-card__blurb">{meta.blurb}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
