import type { Band, RunInfo } from "../types";

/**
 * Says what is being scored, right now, and what it is supposed to be.
 *
 * Without this the dashboard shows a number with no provenance: play two
 * clips in a row and there is nothing on screen tying the score to the audio
 * that produced it.
 *
 * The ground-truth chip comes from the clip's folder name and is marked as
 * such. The detector never receives it -- it exists so you can check results
 * by hand instead of trusting your memory of which file you clicked. Keep it
 * out of frame when recording, or state plainly that it is a debug aid.
 */
export function SourceBar({
  run,
  band,
  windows,
}: {
  run: RunInfo | null;
  band: Band;
  windows: number;
}) {
  if (!run) {
    return (
      <div className="panel sourcebar sourcebar--idle">
        <span className="muted">No active source — start the mic or pick a clip.</span>
      </div>
    );
  }

  const isMic = run.source === "mic";
  const truth = run.source_label;

  // Only meaningful once a verdict exists and the clip is labelled.
  const agrees =
    truth === null || windows === 0
      ? null
      : (truth === "fake" || truth === "spoof") === (band === "HIGH" || band === "MEDIUM");

  return (
    <div className="panel sourcebar">
      <span className="sourcebar__kind">{isMic ? "microphone" : "replay"}</span>
      <span className="sourcebar__name">{isMic ? "live input" : run.source}</span>

      {truth && (
        <span className={`chip chip--${truth}`}>
          ground truth: {truth}
          <span className="chip__note"> (debug — not seen by the detector)</span>
        </span>
      )}

      {agrees !== null && (
        <span className={agrees ? "sourcebar__ok" : "sourcebar__bad"}>
          {agrees ? "agrees" : "disagrees"}
        </span>
      )}

      <span className="sourcebar__windows">{windows} window{windows === 1 ? "" : "s"}</span>
    </div>
  );
}
