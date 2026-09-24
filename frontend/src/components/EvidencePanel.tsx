interface FeatureMeta {
  key: string;
  label: string;
  natural: [number, number];
  note: string;
}

/**
 * Typical natural-speech ranges for 16 kHz telephone-band audio.
 *
 * These are hand-picked reference bands, not fitted thresholds, and the panel
 * labels them as such. Presenting a heuristic as a measurement is the kind of
 * thing that survives a demo and not a question.
 */
const FEATURES: FeatureMeta[] = [
  { key: "flatness", label: "Spectral flatness", natural: [0.02, 0.35],
    note: "Vocoders sit flatter — fewer sharp harmonic peaks." },
  { key: "hf_ratio", label: "HF energy ratio", natural: [0.02, 0.30],
    note: "Energy above 6 kHz. Vocoders over- or under-generate it." },
  { key: "jitter", label: "Jitter (F0)", natural: [0.005, 0.04],
    note: "Real phonation is irregular. Too steady is itself a tell." },
  { key: "shimmer", label: "Shimmer (amplitude)", natural: [0.01, 0.20],
    note: "Cycle-to-cycle amplitude variation." },
];

/**
 * The explainability panel: the DSP features behind the current window.
 *
 * This is what turns "the box said 82" into "the box said 82 because the
 * spectrum is flat and the pitch is too steady".
 */
export function EvidencePanel({ detail }: { detail: Record<string, number> }) {
  const hasData = FEATURES.some((f) => detail[f.key] !== undefined);

  return (
    <div className="panel">
      <h2 className="panel__title">Evidence</h2>
      {!hasData ? (
        <p className="muted">Waiting for a voiced window.</p>
      ) : (
        <table className="evidence">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Value</th>
              <th>Typical natural range</th>
            </tr>
          </thead>
          <tbody>
            {FEATURES.map((feature) => {
              const value = detail[feature.key];
              if (value === undefined) return null;
              const [lo, hi] = feature.natural;
              const outside = value < lo || value > hi;
              return (
                <tr key={feature.key} title={feature.note}>
                  <td>{feature.label}</td>
                  <td className={outside ? "evidence__outside" : undefined}>
                    {value.toFixed(3)}
                  </td>
                  <td className="muted">
                    {lo} – {hi}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <p className="evidence__caveat">
        Reference ranges are hand-picked, not fitted. This layer explains the
        score; it does not drive it.
      </p>
    </div>
  );
}
