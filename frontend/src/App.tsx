/**
 * Dashboard shell. Owns the socket, the mic handle and the verdict history;
 * every child is presentational.
 *
 * Layout, top to bottom: status bar, risk gauge, timeline, layer cards,
 * evidence panel, event log. The gauge is the only thing that has to read at
 * a glance on a compressed video upload -- everything else is supporting
 * detail for someone who pauses.
 */
export default function App() {
  // State to hold:
  //   verdicts: Verdict[]        (trim to the last ~120 for the chart)
  //   sessionInfo: SessionInfo | null
  //   status, micActive, clips, error
  //
  // Effects:
  //   - fetch /api/clips on mount
  //   - open DetectorSocket on mount, close on unmount
  //
  // Warn visibly if sessionInfo.detector_is_fallback -- recording the fallback
  // by mistake is the failure mode that wastes a whole take.
  throw new Error("not implemented");
}
