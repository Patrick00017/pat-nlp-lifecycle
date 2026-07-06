import IssuePanel from './IssuePanel';

const COL_NAMES = {
  speed: '车速',
  min_glue: '最小糊间隙',
  max_glue: '最大糊间隙',
  min_weight: '最小克重',
  max_weight: '最大克重',
  current_glue_weight: '当前克重',
  speed_factor: '车速系数',
  min_speed: '最低车速',
  qdm_factor: 'QDM 系数',
  ui_factor: '界面系数',
  offset: '偏移量',
  value: '结果值',
};

export default function ChartView({ event, onBack, materialEvents }) {
  if (!event) return null;
  const sv = event.set_values;
  if (!sv || !sv.data) {
    return <div style={{ padding: 20, color: '#6b7280' }}>该事件无计算数据</div>;
  }

  const cols = sv.columns;

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

  const reasonMap = { normal: '正常换材', reset: '复位', hq: '横切校验', real: '实际材质' };

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
            材质：{event.material} 楞型：{event.flute_type}
          </span>
        )}
      </div>

      {/* data table */}
      <div style={{ padding: '16px', overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'monospace' }}>
          <thead>
            <tr style={{ background: '#f9fafb' }}>
              {cols.map((col, i) => (
                <th key={i} style={{
                  padding: '6px 10px', textAlign: 'right', borderBottom: '2px solid #e5e7eb',
                  whiteSpace: 'nowrap', color: '#374151', fontWeight: 600,
                }}>
                  {COL_NAMES[col] || col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sv.data.map((row, ri) => (
              <tr key={ri} style={{ background: ri % 2 === 0 ? '#fff' : '#f9fafb' }}>
                {row.map((cell, ci) => (
                  <td key={ci} style={{
                    padding: '4px 10px', textAlign: 'right', borderBottom: '1px solid #f3f4f6',
                    whiteSpace: 'nowrap',
                    fontWeight: cols[ci] === 'value' ? 600 : 400,
                    color: cols[ci] === 'value' ? '#1e40af' : '#374151',
                  }}>
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
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
                <span>{me.time ? me.time.slice(11) : ''}</span>
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
