import type { ServerMessage } from "../types";

/**
 * WebSocket client for /ws/stream.
 *
 * Binary frames out (int16 PCM), JSON frames in. Reconnects with backoff,
 * because a socket that dies silently mid-demo is indistinguishable from a
 * model that has stopped detecting.
 */
export class DetectorSocket {
  constructor(
    private url: string,
    private onMessage: (msg: ServerMessage) => void,
    private onStatus: (status: "connecting" | "open" | "closed") => void,
  ) {}

  connect(): void {
    throw new Error("not implemented");
  }

  /** Send a chunk of int16 PCM. No-op if the socket is not open. */
  sendPCM(_buf: ArrayBuffer): void {
    throw new Error("not implemented");
  }

  /** Ask the server to replay a bundled clip through the same pipeline. */
  requestReplay(_filename: string): void {
    throw new Error("not implemented");
  }

  close(): void {
    throw new Error("not implemented");
  }
}
