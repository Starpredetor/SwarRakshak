import type { ServerMessage } from "../types";

export type SocketStatus = "connecting" | "open" | "closed";

const BASE_RETRY_MS = 500;
const MAX_RETRY_MS = 8000;

/**
 * WebSocket client for /ws/stream.
 *
 * Binary frames out (int16 PCM), JSON frames in. Reconnects with exponential
 * backoff, because a socket that dies silently mid-demo is indistinguishable
 * on screen from a model that has stopped detecting -- the gauge just stops
 * moving either way.
 */
export class DetectorSocket {
  private socket: WebSocket | null = null;
  private retryMs = BASE_RETRY_MS;
  private retryTimer: number | null = null;
  private closedByUs = false;

  constructor(
    private url: string,
    private onMessage: (msg: ServerMessage) => void,
    private onStatus: (status: SocketStatus) => void,
  ) {}

  connect(): void {
    this.closedByUs = false;
    this.onStatus("connecting");

    const socket = new WebSocket(this.url);
    socket.binaryType = "arraybuffer";
    this.socket = socket;

    socket.onopen = () => {
      this.retryMs = BASE_RETRY_MS;
      this.onStatus("open");
    };

    socket.onmessage = (event) => {
      try {
        this.onMessage(JSON.parse(event.data as string) as ServerMessage);
      } catch {
        // A frame we cannot parse is dropped rather than thrown: one bad
        // message must not take down the stream.
      }
    };

    socket.onclose = () => {
      this.onStatus("closed");
      this.socket = null;
      if (!this.closedByUs) this.scheduleRetry();
    };

    // onerror is always followed by onclose, so retry is handled there.
    socket.onerror = () => socket.close();
  }

  private scheduleRetry(): void {
    if (this.retryTimer !== null) return;
    this.retryTimer = window.setTimeout(() => {
      this.retryTimer = null;
      this.retryMs = Math.min(this.retryMs * 2, MAX_RETRY_MS);
      this.connect();
    }, this.retryMs);
  }

  get isOpen(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  /** Send a chunk of int16 PCM. Dropped if the socket is not open. */
  sendPCM(buf: ArrayBuffer): void {
    if (this.isOpen) this.socket!.send(buf);
  }

  /** Ask the server to replay a bundled clip through the same pipeline. */
  requestReplay(filename: string): void {
    this.send({ type: "replay", filename });
  }

  stopReplay(): void {
    this.send({ type: "stop" });
  }

  private send(command: Record<string, unknown>): void {
    if (this.isOpen) this.socket!.send(JSON.stringify(command));
  }

  close(): void {
    this.closedByUs = true;
    if (this.retryTimer !== null) {
      window.clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    this.socket?.close();
    this.socket = null;
  }
}
