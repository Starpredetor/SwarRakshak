import type { ClipInfo } from "../types";
import type { SocketStatus } from "../api/ws";

/**
 * Source selection: live microphone, or replay a bundled clip.
 *
 * Both feed the same socket. The replay control is the reliable take -- if the
 * room is noisy or the mic misbehaves while recording, there is still a
 * reproducible run to cut to.
 */
export function Controls({
  micActive,
  onToggleMic,
  clips,
  onReplay,
  onStopReplay,
  replaying,
  status,
  detectorName,
  detectorIsFallback,
}: {
  micActive: boolean;
  onToggleMic: () => void;
  clips: ClipInfo[];
  onReplay: (filename: string) => void;
  onStopReplay: () => void;
  replaying: boolean;
  status: SocketStatus;
  detectorName: string;
  detectorIsFallback: boolean;
}) {
  return (
    <div className="panel controls">
      <div className="controls__row">
        <span className={`status status--${status}`}>{status}</span>
        <span className="controls__detector">
          {detectorName}
          {detectorIsFallback && (
            /* Recording the fallback by mistake is the failure mode that
               wastes a whole take. It gets a warning, not a footnote. */
            <strong className="controls__fallback"> · FALLBACK MODEL</strong>
          )}
        </span>
      </div>

      <div className="controls__row">
        <button
          className={micActive ? "btn btn--active" : "btn"}
          onClick={onToggleMic}
          disabled={status !== "open" || replaying}
        >
          {micActive ? "Stop microphone" : "Start microphone"}
        </button>

        {replaying && (
          <button className="btn" onClick={onStopReplay}>
            Stop replay
          </button>
        )}
      </div>

      <div className="controls__row">
        <label className="controls__label" htmlFor="clip">
          Replay clip
        </label>
        <select
          id="clip"
          className="select"
          defaultValue=""
          disabled={status !== "open" || micActive || clips.length === 0}
          onChange={(event) => {
            if (event.target.value) onReplay(event.target.value);
          }}
        >
          <option value="" disabled>
            {clips.length ? "choose a clip…" : "no clips in data/demo_clips/"}
          </option>
          {clips.map((clip) => (
            <option key={clip.filename} value={clip.filename}>
              {clip.filename}
              {clip.label ? ` · ${clip.label}` : ""} · {clip.duration_seconds}s
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
