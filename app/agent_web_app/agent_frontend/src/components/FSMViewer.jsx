import { useState, useEffect } from 'react';
import { fetchFSMResults } from '../api';
import TimelineView from './TimelineView';
import ChartView from './ChartView';

const POSITIONS = ['GU1', 'GU2', 'GU3', 'SF1', 'SF2', 'SF3', 'MAT'];

export default function FSMViewer() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [position, setPosition] = useState('GU2');
  const [selectedEvent, setSelectedEvent] = useState(null);

  useEffect(() => {
    fetchFSMResults()
      .then(setData)
      .catch(e => console.error('FSM data load failed:', e))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 20, color: '#6b7280' }}>加载 FSM 结果...</div>;
  if (!data) return <div style={{ padding: 20, color: '#dc2626' }}>无法加载 FSM 结果文件</div>;

  const isMat = position === 'MAT';
  const events = isMat
    ? (data.material_events || []).map(e => {
        const reasonMap = { 'normal': '正常换材', 'reset': '复位' };
        const reason = reasonMap[e.reason] || e.reason || '';
        return {
          event_id: (e.part || '').toUpperCase(),
          time: e.time,
          material: e.msg,
          flute_type: reason,
          errors: [],
          warnings: [],
        };
      })
    : (data.glue_events?.[position] || []);
  const selectedData = !isMat && selectedEvent?.set_values ? selectedEvent : null;

  return (
    <div style={{ marginTop: 16, border: '1px solid #e5e7eb', borderRadius: 8, overflow: 'hidden' }}>
      {/* header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '10px 16px',
        background: '#f9fafb', borderBottom: '1px solid #e5e7eb',
      }}>
        <span style={{ fontWeight: 600, fontSize: 14 }}>FSM 诊断结果</span>
        <span style={{ fontSize: 12, color: '#6b7280' }}>
          {Object.values(data.glue_events || {}).reduce((s, arr) => s + (arr?.length || 0), 0)} 个事件
        </span>
      </div>

      {/* position tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid #e5e7eb' }}>
        {POSITIONS.map(pos => {
          const cnt = data.glue_events?.[pos]?.length || 0;
          return (
            <button
              key={pos}
              onClick={() => { setPosition(pos); setSelectedEvent(null); }}
              style={{
                flex: 1, padding: '8px 4px', border: 'none',
                background: position === pos ? '#fff' : '#f9fafb',
                borderBottom: position === pos ? '2px solid #3b82f6' : '2px solid transparent',
                fontWeight: position === pos ? 600 : 400,
                fontSize: 13, cursor: 'pointer', color: position === pos ? '#1e40af' : '#6b7280',
                transition: 'all 0.1s',
              }}
            >
              {pos}
              <span style={{ fontSize: 11, marginLeft: 4, color: '#9ca3af' }}>({cnt})</span>
            </button>
          );
        })}
      </div>

      {/* content */}
      <div style={{ padding: 16, minHeight: 200 }}>
        {selectedData ? (
          <ChartView event={selectedEvent} onBack={() => setSelectedEvent(null)} materialEvents={data.material_events} />
        ) : (
          <TimelineView events={events} onSelectEvent={(evt) => setSelectedEvent(evt)} />
        )}
      </div>
    </div>
  );
}
