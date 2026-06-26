import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import IssuePanel from './IssuePanel';

export default function ChartView({ event, onBack, materialEvents }) {
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

  // 用 material_dismatch 的 args.id 精确匹配换材事件
  const matErrorIds = (event.errors || [])
    .filter(e => e.type === 'material_dismatch')
    .map(e => e.args?.id)
    .filter(Boolean);
  const relatedMaterials = [];
  if (matErrorIds.length > 0 && materialEvents) {
    const sorted = [...materialEvents].sort((a, b) => new Date(a.time) - new Date(b.time));
    const matchedIndices = matErrorIds
      .map(id => sorted.findIndex(m => m.id === id))
      .filter(i => i >= 0);
    if (matchedIndices.length > 0) {
      const minIdx = Math.max(0, Math.min(...matchedIndices) - 5);
      const maxIdx = Math.min(sorted.length - 1, Math.max(...matchedIndices) + 5);
      const idSet = new Set(matErrorIds);
      for (const m of sorted.slice(minIdx, maxIdx + 1)) {
        relatedMaterials.push({ ...m, _isMatch: idSet.has(m.id), _t: new Date(m.time).getTime() });
      }
    }
  }

  const reasonMap = { normal: '正常换材', reset: '复位' };

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
        <IssuePanel errors={event.errors} warnings={event.warnings} passes={event.passes} />
      </div>

      {/* 附近换材记录 */}
      {relatedMaterials.length > 0 && (
        <div style={{ padding: '0 16px 12px', borderTop: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 6, paddingTop: 8 }}>
            📋 附近换材记录 ({relatedMaterials.length})
          </div>
          {(() => {
            const reasonMap = { normal: '正常换材', reset: '复位' };
            const matchedItems = relatedMaterials.filter(m => m._isMatch);
            if (matchedItems.length === 0) return null;
            const matchIndices = relatedMaterials
              .map((m, i) => m._isMatch ? i : -1)
              .filter(i => i >= 0);
            const minMatchIdx = Math.min(...matchIndices);
            const maxMatchIdx = Math.max(...matchIndices);
            const before = relatedMaterials.filter((m, i) => !m._isMatch && i < minMatchIdx);
            const after = relatedMaterials.filter((m, i) => !m._isMatch && i > maxMatchIdx);
            const rows = [];

            const renderItem = (me, idx, isMatch = false) => (
              <div key={idx} style={{
                display: 'flex', gap: 8, fontSize: 13, padding: '3px 8px',
                color: '#6b7280', fontFamily: 'monospace',
                background: isMatch ? '#fef3c7' : (idx % 2 === 0 ? '#f9fafb' : 'transparent'),
                borderRadius: 4, fontWeight: isMatch ? 600 : 400,
              }}>
                <span>{me.time ? me.time.slice(11, 19) : ''}</span>
                <span style={{ fontWeight: 600, minWidth: 32 }}>{(me.part || '').toUpperCase()}</span>
                <span style={{ flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{me.msg}</span>
                <span style={{ color: '#9ca3af' }}>{reasonMap[me.reason] || me.reason || ''}</span>
              </div>
            );

            if (before.length > 0) {
              rows.push(<div key="before-title" style={{ fontSize: 12, color: '#9ca3af', padding: '2px 8px', fontWeight: 500 }}>— 之前 —</div>);
              before.forEach((m, i) => rows.push(renderItem(m, i)));
            }
            rows.push(<div key="recent-title" style={{ fontSize: 12, color: '#d97706', padding: '2px 8px', fontWeight: 500, marginTop: before.length > 0 ? 4 : 0 }}>⭐ 最近</div>);
            matchedItems.forEach((m, i) => rows.push(renderItem(m, i, true)));
            if (after.length > 0) {
              rows.push(<div key="after-title" style={{ fontSize: 12, color: '#9ca3af', padding: '2px 8px', fontWeight: 500, marginTop: 4 }}>— 之后 —</div>);
              after.forEach((m, i) => rows.push(renderItem(m, i)));
            }
            return rows;
          })()}
        </div>
      )}
    </div>
  );
}
