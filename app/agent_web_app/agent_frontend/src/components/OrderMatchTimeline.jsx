import { useState, useEffect } from 'react';
import { fetchFSMResults } from '../api';

const SLOTS = ['时间', '订单', 'ls0', 'ms1', 'ls1', 'ms2', 'ls2', 'df', 'GU1', 'GU2', 'GU3', 'SF1.ms1', 'SF1.ls1', 'SF2.ms2', 'SF2.ls2'];
const SLOT_LABELS = { 时间: '时间', 订单: '订单', ls0: 'LS0', ms1: 'MS1', ls1: 'LS1', ms2: 'MS2', ls2: 'LS2', df: 'DF', GU1: 'GU1', GU2: 'GU2', GU3: 'GU3',
  'SF1.ms1': 'SF1 ms1', 'SF1.ls1': 'SF1 ls1', 'SF2.ms2': 'SF2 ms2', 'SF2.ls2': 'SF2 ls2' };

const GLUE_SUB_PARENT = { 'SF1.ms1': 'SF1', 'SF1.ls1': 'SF1', 'SF2.ms2': 'SF2', 'SF2.ls2': 'SF2' };

function getGlueSubSlot(slot, seg) {
  const parent = GLUE_SUB_PARENT[slot];
  if (!parent) return seg.glue?.[slot] || null;
  const g = seg.glue?.[parent];
  if (!g) return null;
  const parts = (g.material || '').split('/');
  const subSlot = slot.split('.')[1];
  const mat = slot === parent + '.ms1' || slot === parent + '.ms2' ? (parts[0] || '') : (parts[1] || '');
  const subAnalysis = (g.analysis || []).filter(a => a.slot === subSlot);
  return { material: mat, analysis: subAnalysis, time: g.time };
}

function glueVerdictColor(analysis) {
  if (!analysis || analysis.length === 0) return '#10b981';
  const v = analysis[0]?.verdict || '';
  if (v.startsWith('实际材质')) return '#8b5cf6';
  if (v.startsWith('未知')) return '#ef4444';
  if (v.startsWith('换材提前')) return '#f59e0b';
  if (v.startsWith('换材滞后')) return '#f97316';
  return '#10b981';
}

function glueVerdictTextColor(analysis) {
  if (!analysis || analysis.length === 0) return '#065f46';
  const v = analysis[0]?.verdict || '';
  if (v.startsWith('实际材质')) return '#fff';
  if (v.startsWith('未知')) return '#fff';
  return '#fff';
}

function segmentColor(match, actual) {
  if (actual === '-' || actual === '-.-.-.-.-') return '#e5e7eb';
  return match ? '#10b981' : '#ef4444';
}

function segmentTextColor(match, actual) {
  if (actual === '-' || actual === '-.-.-.-.-') return '#9ca3af';
  return match ? '#065f46' : '#fff';
}

export default function OrderMatchTimeline() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFSMResults()
      .then(setData)
      .catch(e => console.error('Order data load failed:', e))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 20, color: '#6b7280' }}>加载中...</div>;
  if (!data?.order_check) return <div style={{ padding: 20, color: '#6b7280' }}>无订单匹配数据</div>;

  const oc = data.order_check;
  const matReasonMap = {};
  for (const e of (data.material_events || [])) {
    matReasonMap[e.id] = e.reason || '';
  }

  // Collect all unique time points
  const timeSet = new Set();
  for (const m of oc.material_list) if (m.time) timeSet.add(m.time);
  for (const pos of ['GU1','GU2','GU3','SF1','SF2']) {
    for (const e of (data.glue_events?.[pos] || [])) if (e.time) timeSet.add(e.time);
  }
  const sortedTimes = [...timeSet].map(t => new Date(t).getTime()).sort((a, b) => a - b);
  const minT = sortedTimes.length > 0 ? sortedTimes[0] : 0;
  const maxT = sortedTimes.length > 0 ? sortedTimes[sortedTimes.length - 1] : 1;
  const totalDuration = maxT - minT || 1;

  // Build unified segments
  const segments = sortedTimes.map((t, i) => {
    const nextT = (i + 1 < sortedTimes.length) ? sortedTimes[i + 1] : maxT + 1;
    return { startMs: t, wd: Math.max(((nextT - t) / totalDuration) * 100, 0.3) };
  });

  // Fill snapshot data
  let mi = 0, lastOrderData = null, lastSlotData = {};
  const glueCursors = {};
  for (const pos of ['GU1','GU2','GU3','SF1','SF2']) glueCursors[pos] = 0;

  for (const seg of segments) {
    while (mi < oc.material_list.length) {
      const mt = new Date(oc.material_list[mi].time || 0).getTime();
      if (mt > seg.startMs) break;
      lastOrderData = {
        order_id: oc.order_list[mi] || '?',
        paper_code: oc.summary?.[oc.order_list[mi]]?.paper_code || '',
        width: oc.summary?.[oc.order_list[mi]]?.width || '',
        time: oc.material_list[mi].time,
      };
      for (const slot of ['ls0','ms1','ls1','ms2','ls2','df']) {
        const info = oc.match_list[mi]?.slots?.[slot] || {};
        lastSlotData[slot] = {
          actual: info.actual_material || info.actual || '-',
          expected: info.expected_material || info.expected || '-',
          actualW: info.actual_width ?? '',
          expectedW: info.expected_width ?? '',
          match: info.match ?? false,
          reason: matReasonMap[info.id] || '',
        };
      }
      mi++;
    }
    seg.order = lastOrderData ? { ...lastOrderData } : null;
    seg.slots = JSON.parse(JSON.stringify(lastSlotData));
    seg.glue = {};
    for (const pos of ['GU1','GU2','GU3','SF1','SF2']) {
      const events = (data.glue_events?.[pos] || []);
      while (glueCursors[pos] < events.length) {
        const et = new Date(events[glueCursors[pos]].time || 0).getTime();
        if (et > seg.startMs) break;
        seg.glue[pos] = {
          material: events[glueCursors[pos]].material || '',
          analysis: events[glueCursors[pos]].analysis || [],
          time: events[glueCursors[pos]].time,
        };
        glueCursors[pos]++;
      }
    }
  }

  // Time labels
  let lastHHMM = '';
  for (const seg of segments) {
    const d = new Date(seg.startMs);
    const iso = `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
    seg.timeLabel = iso.slice(0, 5) !== lastHHMM ? iso : '';
    lastHHMM = iso.slice(0, 5);
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: '#fff', borderRadius: 8, border: '1px solid #e5e7eb', overflow: 'hidden' }}>
      <div style={{ padding: '10px 16px', background: '#f9fafb', borderBottom: '1px solid #e5e7eb', fontWeight: 600, fontSize: 14 }}>
        订单材质匹配
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', fontFamily: 'monospace', fontSize: 11, width: '100%' }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 1 }}>
            <tr style={{ background: '#f9fafb' }}>
              {SLOTS.map(s => (
                <th key={s} style={{
                  padding: '4px 6px', borderBottom: '2px solid #e5e7eb',
                  fontSize: 10, fontWeight: 600, color: '#374151',
                  textAlign: 'center', whiteSpace: 'nowrap',
                }}>{SLOT_LABELS[s]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {segments.map((seg, j) => {
              const o = seg.order || {};
              const isOrderNew = j > 0 && seg.order?.order_id !== segments[j - 1]?.order?.order_id;
              return (
                <tr key={j} style={{
                  borderTop: isOrderNew ? '2px solid #fff' : '1px solid #f3f4f6',
                }}>
                  {SLOTS.map(slot => {
                    let content, color = 'transparent', textColor, title = '';
                    if (slot === '时间') {
                      color = 'transparent'; textColor = '#9ca3af';
                      content = <span style={{ fontSize: 10 }}>{seg.timeLabel}</span>;
                    } else if (slot === '订单') {
                      color = '#e5e7eb'; textColor = '#374151';
                      content = (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <span style={{ fontWeight: 600 }}>{o.order_id}</span>
                          {o.paper_code && <span style={{ fontSize: 7, opacity: 0.5 }}>{o.paper_code}</span>}
                        </div>
                      );
                    } else if (slot.startsWith('GU') || slot.startsWith('SF')) {
                      const g = getGlueSubSlot(slot, seg);
                      color = g ? glueVerdictColor(g.analysis) : '#e5e7eb';
                      textColor = g ? glueVerdictTextColor(g.analysis) : '#9ca3af';
                      content = g ? <span style={{ fontSize: 10 }}>{SLOT_LABELS[slot]} {g.material}</span> : null;
                    } else {
                      const s = seg.slots?.[slot] || { actual: '-', expected: '-', match: false };
                      color = segmentColor(s.match, s.actual);
                      textColor = segmentTextColor(s.match, s.actual);
                      content = (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <span>{s.actual}</span>
                          {(s.actualW || 0) > 0 && <span style={{ fontSize: 8, opacity: 0.6 }}>{s.actualW}</span>}
                          {s.reason && <span style={{ fontSize: 7, opacity: 0.5 }}>
                            {{normal:'正常换材',hq:'横切校验',real:'实际材质',reset:'初始化'}[s.reason] || s.reason}
                          </span>}
                        </div>
                      );
                    }
                    return (
                      <td key={slot} style={{
                        padding: '2px 4px', background: color, color: textColor,
                        textAlign: 'center', verticalAlign: 'middle', fontSize: 10,
                        lineHeight: 1.2, borderRight: '1px solid rgba(255,255,255,0.4)',
                      }}>{content}</td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
