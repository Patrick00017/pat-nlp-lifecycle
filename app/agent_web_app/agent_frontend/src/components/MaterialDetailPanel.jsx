export default function MaterialDetailPanel({ event, onBack }) {
  if (!event) return null;

  const reasonMap = { normal: '正常换材', hq: '横切校验', real: '实际材质', reset: '初始化' };
  const fmtMat = (msg) => {
    const m = msg?.match(/\(([^,]+),([^,]+),([^)]+)\)\s*->\s*\(([^,]+),([^,]+),([^)]+)\)/);
    if (!m) return msg;
    return `(材质：${m[1]},门幅：${m[2]},楞型：${m[3]}) -> (材质：${m[4]},门幅：${m[5]},楞型：${m[6]})`;
  };

  return (
    <div style={{ background: '#fff', borderRadius: 8, border: '1px solid #e5e7eb', maxWidth: 560, width: '100%' }}>
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
        >← 返回</button>
        <span style={{ fontWeight: 600, fontSize: 14 }}>{(event.part || '').toUpperCase()} 换材详情</span>
      </div>

      {/* body */}
      <div style={{ padding: 16, fontSize: 13, lineHeight: 1.8 }}>
        <div style={{ fontFamily: 'monospace', color: '#374151', marginBottom: 8 }}>
          {fmtMat(event.msg || '')}
        </div>
        <div style={{ color: '#6b7280', fontSize: 12 }}>
          时间：{event.time}
        </div>
        <div style={{ color: '#6b7280', fontSize: 12 }}>
          原因：{reasonMap[event.reason] || event.reason}
        </div>
        {event.remaining_mm_list?.length > 0 && (
          <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: 12, marginTop: 8 }}>
            <div style={{ fontWeight: 600, fontSize: 13, color: '#374151', marginBottom: 8 }}>
              同材剩余米数（换材前10秒内）
            </div>
            {(() => {
              const vals = event.remaining_mm_list;
              const max = Math.max(...vals);
              const min = Math.min(...vals);
              const range = (max - min) || 1;
              const barH = 60;
              const w = Math.min(480, vals.length * 50);
              const barW = Math.max(w / vals.length - 2, 8);
              return (
                <div style={{ position: 'relative', width: w, height: barH + 28 }}>
                  <svg width={w} height={barH + 28} style={{ overflow: 'visible' }}>
                    {/* connecting line */}
                    <polyline
                      points={vals.map((v, i) => {
                        const x = i * (barW + 2) + barW / 2;
                        const y = barH - ((v - min) / range) * (barH - 12) - 4;
                        return `${x},${y}`;
                      }).join(' ')}
                      fill="none" stroke="#3b82f6" strokeWidth="1.5" opacity="0.5"
                    />
                    {/* bars */}
                    {vals.map((v, i) => {
                      const x = i * (barW + 2);
                      const h = Math.max(((v - min) / range) * (barH - 12), 3);
                      const y = barH - h;
                      const ratio = 1 - ((v - min) / range);
                      return (
                        <g key={i}>
                          <rect x={x} y={y} width={barW} height={h} rx={2}
                            fill={`hsl(${220 + ratio * 40}, 70%, ${50 + ratio * 20}%)`}
                            opacity={0.85}
                          />
                          <text x={x + barW / 2} y={barH + 16} textAnchor="middle"
                            fontSize="9" fill="#6b7280" fontFamily="monospace">
                            {v.toFixed(3)}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                </div>
              );
            })()}
          </div>
        )}
      </div>
    </div>
  );
}
