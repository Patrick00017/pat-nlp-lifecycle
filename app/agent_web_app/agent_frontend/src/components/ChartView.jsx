import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import IssuePanel from './IssuePanel';

export default function ChartView({ event, onBack }) {
  if (!event) return null;
  const sv = event.set_values;
  if (!sv || !sv.data) {
    return <div style={{ padding: 20, color: '#6b7280' }}>该事件无计算数据</div>;
  }

  const cols = sv.columns;
  const speedIdx = cols.indexOf('speed');
  const valIdx = cols.indexOf('value');
  const minIdx = cols.indexOf('min_glue');
  const maxIdx = cols.indexOf('max_glue');

  const chartData = sv.data.map(row => ({
    speed: row[speedIdx],
    value: parseFloat(row[valIdx]) || 0,
    min: row[minIdx] ? parseFloat(row[minIdx]) : undefined,
    max: row[maxIdx] ? parseFloat(row[maxIdx]) : undefined,
  }));

  return (
    <div style={{ background: '#fff', borderRadius: 8, border: '1px solid #e5e7eb' }}>
      {/* header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '12px 16px', borderBottom: '1px solid #e5e7eb',
      }}>
        <button
          onClick={onBack}
          style={{
            background: 'none', border: '1px solid #d1d5db', borderRadius: 4,
            padding: '4px 10px', cursor: 'pointer', fontSize: 13,
          }}
        >
          ← 返回时间线
        </button>
        <span style={{ fontWeight: 600, fontSize: 14 }}>
          {event.event_id} · {event.part || event.func}
        </span>
        {event.material && (
          <span style={{ fontSize: 12, color: '#6b7280' }}>
            {event.material} / {event.flute_type}
          </span>
        )}
      </div>

      {/* chart */}
      <div style={{ padding: '16px 16px 0' }}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
            <XAxis
              dataKey="speed"
              label={{ value: '车速', position: 'insideBottomRight', offset: -5, style: { fontSize: 12, fill: '#6b7280' } }}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              label={{ value: '糊间隙值', angle: -90, position: 'insideLeft', style: { fontSize: 12, fill: '#6b7280' } }}
              tick={{ fontSize: 11 }}
            />
            <Tooltip />
            {chartData[0]?.min !== undefined && (
              <Line type="monotone" dataKey="min" stroke="#fbbf24" strokeWidth={1} strokeDasharray="4 4" dot={false} name="下限" />
            )}
            {chartData[0]?.max !== undefined && (
              <Line type="monotone" dataKey="max" stroke="#f87171" strokeWidth={1} strokeDasharray="4 4" dot={false} name="上限" />
            )}
            <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4, fill: '#3b82f6' }} name="实际值" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* issues */}
      <div style={{ padding: '0 16px 12px' }}>
        <IssuePanel issues={[...(event.errors || []), ...(event.warnings || [])]} />
      </div>
    </div>
  );
}
