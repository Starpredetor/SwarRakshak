import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Verdict } from "../types";

/**
 * Rolling risk-over-time chart with the band thresholds drawn in.
 *
 * This is what makes a mid-call voice swap legible: a single number cannot
 * show that the line sat flat for thirty seconds and then climbed. The
 * threshold lines are what let a viewer read the climb as crossing a
 * decision boundary rather than as a wobble.
 */
export function RiskTimeline({ verdicts }: { verdicts: Verdict[] }) {
  const data = verdicts.map((v) => ({ t: v.t, risk: v.risk }));

  return (
    <div className="panel">
      <h2 className="panel__title">Risk over time</h2>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -16 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis
            dataKey="t"
            stroke="var(--muted)"
            tick={{ fontSize: 11 }}
            tickFormatter={(t: number) => `${t.toFixed(0)}s`}
          />
          <YAxis domain={[0, 100]} stroke="var(--muted)" tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{
              background: "var(--panel)",
              border: "1px solid var(--border)",
              borderRadius: 8,
            }}
            labelFormatter={(t) => `t = ${Number(t).toFixed(1)}s`}
          />
          <ReferenceLine y={40} stroke="var(--medium)" strokeDasharray="4 4" />
          <ReferenceLine y={75} stroke="var(--high)" strokeDasharray="4 4" />
          <Line
            type="monotone"
            dataKey="risk"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
