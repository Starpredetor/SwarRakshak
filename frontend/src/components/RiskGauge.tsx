import type { Band } from "../types";

const RADIUS = 120;
const ARC_LENGTH = Math.PI * RADIUS; // semicircle

/**
 * The hero element: a 0-100 arc that is the whole story in one glance.
 *
 * Colour comes from the band, not from the raw number, so the gauge can never
 * show an amber tint while the label reads LOW. The sweep transitions over
 * roughly the hop interval -- an instant jump reads as a rendering glitch on
 * video rather than as a detection.
 */
export function RiskGauge({
  risk,
  band,
  action,
  stale,
}: {
  risk: number;
  band: Band;
  action: string;
  stale?: boolean;
}) {
  const filled = (Math.max(0, Math.min(100, risk)) / 100) * ARC_LENGTH;

  return (
    <div className="panel gauge" aria-live="polite">
      <svg viewBox="0 0 300 170" width="100%" height="auto" role="img"
           aria-label={`Risk ${risk} of 100, band ${band}`}>
        <path
          d="M 30 140 A 120 120 0 0 1 270 140"
          className="gauge__track"
          fill="none"
          strokeWidth="18"
          strokeLinecap="round"
        />
        <path
          d="M 30 140 A 120 120 0 0 1 270 140"
          className={`gauge__fill band-${band}`}
          fill="none"
          strokeWidth="18"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${ARC_LENGTH}`}
        />
        <text x="150" y="120" textAnchor="middle" className={`gauge__value band-${band}`}>
          {risk}
        </text>
        <text x="150" y="152" textAnchor="middle" className="gauge__scale">
          risk score
        </text>
      </svg>

      <div className={`gauge__band band-${band}`}>{band}</div>
      <div className="gauge__action">{action}</div>
      {stale && (
        <div className="gauge__stale">
          no speech detected — holding last score
        </div>
      )}
    </div>
  );
}
