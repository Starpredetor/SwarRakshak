import type { Verdict } from "../types";
import { BAND_ACTION } from "../types";

/**
 * Append-only log of band changes, with timestamps.
 *
 * Only band *changes*, not every window -- a line per second scrolls too fast
 * to read back on video, and the moment the band moved is the only moment
 * anyone replays.
 */
export function EventLog({ verdicts }: { verdicts: Verdict[] }) {
  const events = verdicts.filter((v) => v.band_changed).slice(-12).reverse();

  return (
    <div className="panel">
      <h2 className="panel__title">Events</h2>
      {events.length === 0 ? (
        <p className="muted">No band changes yet.</p>
      ) : (
        <ul className="events">
          {events.map((event) => (
            <li key={`${event.t}-${event.band}`} className="events__row">
              <span className="events__time">{event.t.toFixed(1)}s</span>
              <span className={`events__band band-${event.band}`}>{event.band}</span>
              <span className="events__action">{BAND_ACTION[event.band]}</span>
              <span className="events__risk">risk {event.risk}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
