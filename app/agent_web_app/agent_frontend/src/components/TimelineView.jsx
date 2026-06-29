import IssueBadge from './IssueBadge';

function fmtMaterial(msg) {
  const m = msg?.match(/\(([^,]+),([^,]+),([^)]+)\)\s*->\s*\(([^,]+),([^,]+),([^)]+)\)/);
  if (!m) return msg;
  return `(材质：${m[1]},门幅：${m[2]},楞型：${m[3]}) -> (材质：${m[4]},门幅：${m[5]},楞型：${m[6]})`;
}

const EVENT_LABELS = {
  'G1': 'HandleGuGlueMsg', 'G2': 'SetGlueSF*', 'G3': '任务终止',
  'G4': 'SF糊间隙计算', 'G5': 'SF写值完成', 'G6': '换材通知',
  'G7': 'SetGlueGu', 'G8': '延迟取消', 'G9': '部位匹配失败',
  'G10': '降级匹配', 'G11': '立即换材', 'G12': 'GU写值完成',
  'G13': '写入终止', 'G14': 'GU糊间隙计算', 'G15': '写值取消',
};

export default function TimelineView({ events = [], onSelectEvent }) {
  if (events.length === 0) {
    return <div style={{ color: '#9ca3af', padding: 20 }}>该位置无事件</div>;
  }
  return (
    <div style={{ position: 'relative', paddingLeft: 28 }}>
      {/* vertical line */}
      <div style={{
        position: 'absolute', left: 11, top: 8, bottom: 8, width: 2,
        background: '#e5e7eb',
      }} />
      {events.map((evt, i) => {
        const label = EVENT_LABELS[evt.event_id] || evt.event_id;
        const isClickable = evt.set_values && evt.set_values.data;
        return (
          <div
            key={i}
            onClick={() => isClickable && onSelectEvent?.(evt)}
            style={{
              position: 'relative', marginBottom: 12, padding: '8px 12px',
              borderRadius: 6, cursor: isClickable ? 'pointer' : 'default',
              background: isClickable ? '#f9fafb' : 'transparent',
              border: '1px solid',
              borderColor: isClickable ? '#e5e7eb' : 'transparent',
              transition: 'background 0.15s',
            }}
            onMouseEnter={e => { if (isClickable) e.currentTarget.style.background = '#f3f4f6'; }}
            onMouseLeave={e => { if (isClickable) e.currentTarget.style.background = '#f9fafb'; }}
          >
            {/* dot */}
            <div style={{
              position: 'absolute', left: -17, top: 12, width: 10, height: 10,
              borderRadius: '50%', background: isClickable ? '#3b82f6' : '#d1d5db',
              border: '2px solid white',
            }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 12, color: '#6b7280', fontFamily: 'monospace' }}>
                {evt.time ? evt.time.slice(11) : ''}
              </span>
              <span style={{ fontWeight: 600, fontSize: 13 }}>{[evt.event_id, evt.part].filter(Boolean).join(' · ')}</span>
              {label !== evt.event_id && <span style={{ fontSize: 12, color: '#374151' }}>{label}</span>}
              {(evt.errors || []).map((err, j) => (
                <IssueBadge key={j} type={err.type} detail={err.detail} args={err.args} />
              ))}
              {(evt.warnings || []).map((w, j) => (
                <IssueBadge key={`w${j}`} type={w.type} detail={w.detail} args={w.args} />
              ))}
            </div>
            <div style={{ fontSize: 12, color: '#6b7280', marginTop: 2 }}>
              {evt.material && evt.material.includes('->')
                ? <span>{fmtMaterial(evt.material)}{evt.flute_type ? `  原因：${evt.flute_type}` : ''}</span>
                : evt.material && <span>材质：{evt.material}  楞型：{evt.flute_type || ''}</span>
              }
            </div>
          </div>
        );
      })}
    </div>
  );
}
