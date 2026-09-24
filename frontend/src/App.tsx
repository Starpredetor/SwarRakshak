import { useCallback, useEffect, useRef, useState } from "react";

import { DetectorSocket, type SocketStatus } from "./api/ws";
import { startMicCapture, type MicHandle } from "./audio/micCapture";
import { Controls } from "./components/Controls";
import { EventLog } from "./components/EventLog";
import { EvidencePanel } from "./components/EvidencePanel";
import { LayerCards } from "./components/LayerCards";
import { RiskGauge } from "./components/RiskGauge";
import { RiskTimeline } from "./components/RiskTimeline";
import { BAND_ACTION, type ClipInfo, type SessionInfo, type Verdict } from "./types";

// Two minutes of history at a 1 s hop. Enough to show a mid-call swap, short
// enough that the chart stays readable when it is scaled down for video.
const MAX_VERDICTS = 120;

const WS_URL =
  import.meta.env.VITE_WS_URL ?? `ws://${window.location.host}/ws/stream`;

/**
 * Dashboard shell. Owns the socket, the mic handle and the verdict history;
 * every child is presentational.
 *
 * Layout, top to bottom: controls, risk gauge, timeline, layer cards, evidence
 * panel, event log. The gauge is the only thing that has to read at a glance
 * on a compressed video upload -- everything else is supporting detail for
 * someone who pauses.
 */
export default function App() {
  const [verdicts, setVerdicts] = useState<Verdict[]>([]);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [status, setStatus] = useState<SocketStatus>("connecting");
  const [clips, setClips] = useState<ClipInfo[]>([]);
  const [micActive, setMicActive] = useState(false);
  const [replaying, setReplaying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<DetectorSocket | null>(null);
  const micRef = useRef<MicHandle | null>(null);

  useEffect(() => {
    const socket = new DetectorSocket(
      WS_URL,
      (message) => {
        if (message.type === "session") {
          setSession(message);
          // A reconnect starts a new server-side session, so the old history
          // belongs to a call that no longer exists.
          setVerdicts([]);
          setError(null);
        } else if (message.type === "verdict") {
          setVerdicts((prev) => [...prev, message].slice(-MAX_VERDICTS));
        } else if (message.type === "error") {
          setError(message.message);
          setReplaying(false);
        }
      },
      setStatus,
    );

    socketRef.current = socket;
    socket.connect();

    return () => {
      micRef.current?.stop();
      micRef.current = null;
      socket.close();
      socketRef.current = null;
    };
  }, []);

  useEffect(() => {
    fetch("/api/clips")
      .then((response) => (response.ok ? response.json() : []))
      .then(setClips)
      .catch(() => setClips([]));
  }, []);

  const toggleMic = useCallback(async () => {
    if (micRef.current) {
      micRef.current.stop();
      micRef.current = null;
      setMicActive(false);
      return;
    }

    try {
      micRef.current = await startMicCapture(
        (buf) => socketRef.current?.sendPCM(buf),
        session?.sample_rate ?? 16000,
      );
      setMicActive(true);
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error
          ? `Microphone unavailable: ${err.message}`
          : "Microphone unavailable",
      );
    }
  }, [session]);

  const startReplay = useCallback((filename: string) => {
    socketRef.current?.requestReplay(filename);
    setReplaying(true);
    setError(null);
  }, []);

  const stopReplay = useCallback(() => {
    socketRef.current?.stopReplay();
    setReplaying(false);
  }, []);

  const latest = verdicts.at(-1);
  const risk = latest?.risk ?? 0;
  const band = latest?.band ?? "LOW";
  // Every layer abstaining means the window was silent or unusable. The gauge
  // says so rather than presenting a held score as a fresh reading.
  const stale = Boolean(latest && latest.layers.every((layer) => layer.abstain));
  const dspDetail =
    latest?.layers.find((layer) => layer.layer === "l1b_dsp")?.detail ?? {};

  return (
    <div className="app">
      <header className="header">
        <h1>
          SwarRakshak <span className="header__sub">live voice integrity</span>
        </h1>
        <span className="header__note">
          Feature-only processing — no audio is recorded or stored.
        </span>
      </header>

      {session?.detector_is_fallback && (
        <div className="banner banner--warn">
          Running the offline fallback detector. Accuracy is materially worse
          than the trained model — do not record a demo against this.
        </div>
      )}

      {error && <div className="banner banner--error">{error}</div>}

      <Controls
        micActive={micActive}
        onToggleMic={toggleMic}
        clips={clips}
        onReplay={startReplay}
        onStopReplay={stopReplay}
        replaying={replaying}
        status={status}
        detectorName={session?.detector_name ?? "—"}
        detectorIsFallback={session?.detector_is_fallback ?? false}
      />

      <div className="grid">
        <RiskGauge risk={risk} band={band} action={BAND_ACTION[band]} stale={stale} />
        <RiskTimeline verdicts={verdicts} />
      </div>

      <LayerCards
        layers={latest?.layers ?? []}
        detectorName={session?.detector_name ?? "l1_spoof"}
      />

      <div className="grid">
        <EvidencePanel detail={dspDetail} />
        <EventLog verdicts={verdicts} />
      </div>

      <footer className="footer">
        {latest
          ? `window ${latest.t.toFixed(1)}s · ${latest.latency_ms.toFixed(0)} ms · ${latest.detector_name}`
          : "waiting for audio…"}
      </footer>
    </div>
  );
}
