// Mirrors backend/models.py. Keep the two in step by hand -- there is no
// codegen in the MVP, so a field renamed on one side must be renamed here.

export type Band = "LOW" | "MEDIUM" | "HIGH";

export interface LayerScore {
  layer: string;
  score: number;       // 1.0 = maximally synthetic
  confidence: number;
  abstain: boolean;
  detail: Record<string, number>;
}

export interface RunInfo {
  type: "run";
  session_id: string;
  source: string;
  /** Ground truth from the clip's folder. For checking by hand -- the
   *  detector never receives this. */
  source_label: string | null;
}

export interface Verdict {
  type: "verdict";
  session_id: string;
  source: string;
  source_label: string | null;
  t: number;           // seconds since session start
  risk: number;        // 0-100
  band: Band;
  band_changed: boolean;
  layers: LayerScore[];
  latency_ms: number;
  detector_name: string;
}

export interface SessionInfo {
  type: "session";
  session_id: string;
  sample_rate: number;
  window_seconds: number;
  hop_seconds: number;
  detector_name: string;
  detector_is_fallback: boolean;
}

export interface ErrorMessage {
  type: "error";
  message: string;
  fatal: boolean;
}

export type ServerMessage = Verdict | SessionInfo | RunInfo | ErrorMessage;

export interface ClipInfo {
  filename: string;
  label: string | null;
  duration_seconds: number;
}

export const BAND_ACTION: Record<Band, string> = {
  LOW: "Pass",
  // Advisory only. It must never block the call -- a false positive on a bad
  // line should cost a verification step, not the customer's transaction.
  MEDIUM: "Advise MFA",
  HIGH: "Hold & escalate",
};
