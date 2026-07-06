import { useState, useEffect } from 'react';
import { fetchFSMResults } from '../api';

const SLOTS = ['时间', '订单', 'ls0', 'ms1', 'ls1', 'ms2', 'ls2', 'df', 'GU1', 'GU2', 'GU3', 'SF1', 'SF2'];
const SLOT_LABELS = { 时间: '时间', 订单: '订单', ls0: 'LS0', ms1: 'MS1', ls1: 'LS1', ms2: 'MS2', ls2: 'LS2', df: 'DF', GU1: 'GU1', GU2: 'GU2', GU3: 'GU3', SF1: 'SF1', SF2: 'SF2' };

function glueVerdictColor(analysis) {
  if (!analysis || analysis.length === 0) return '#10b981';
  const v = analysis[0]?.verdict || '';
  if (v.startsWith('未知')) return '#ef4444';
  if (v.startsWith('换材提前')) return '#f59e0b';
  if (v.startsWith('换材滞后')) return '#f97316';
  return '#10b981';
}

function glueVerdictTextColor(analysis) {
  if (!analysis || analysis.length === 0) return '#065f46';
  const v = analysis[0]?.verdict || '';
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
  const times = oc.material_list.map(m => new Date(m.time || 0).getTime());
  // 扩展到胶水事件时间范围
  for (const pos of ['GU1','GU2','GU3','SF1','SF2']) {
    for (const e of (data.glue_events?.[pos] || [])) {
      times.push(new Date(e.time || 0).getTime());
    }
  }
  const minT = times.length > 0 ? Math.min(...times) : 0;
  const maxT = times.length > 0 ? Math.max(...times) : 1;
  const totalDuration = maxT - minT || 1;

  const matReasonMap = {};
  for (const e of (data.material_events || [])) {
    matReasonMap[e.id] = e.reason || '';
  }

  // Build segments: for each slot, a list of {wd%, color, label, tooltip, textColor}
  const slotSegments = {};
  for (const s of SLOTS) slotSegments[s] = [];

  for (let i = 0; i < oc.material_list.length; i++) {
    const t = times[i] || 0;
    const nextT = (i + 1 < times.length) ? times[i + 1] : maxT + 1;
    const wd = Math.max(((nextT - t) / totalDuration) * 100, 0.3);
    const timeFull = oc.material_list[i].time || '';
    const timeHHMM = timeFull.slice(11, 16);
    const prevHHMM = i > 0 ? (oc.material_list[i - 1]?.time || '').slice(11, 16) : '';
    const showTime = i === 0 || timeHHMM !== prevHHMM;

    // 时间轴段：仅每分钟变显
    slotSegments['时间'].push({ wd, label: showTime ? timeFull.slice(11, 19) : '', time: oc.material_list[i].time });

    for (const slot of ['ls0', 'ms1', 'ls1', 'ms2', 'ls2', 'df']) {
      const info = oc.match_list[i]?.slots?.[slot] || {};
      const actual = info.actual_material || info.actual || '-';
      const expected = info.expected_material || info.expected || '-';
      const actualW = info.actual_width ?? '';
      const expectedW = info.expected_width ?? '';
      const match = info.match ?? false;
      const reason = matReasonMap[info.id] || '';
      slotSegments[slot].push({ wd, actual, expected, actualW, expectedW, match, reason, time: oc.material_list[i].time });
    }
    // 订单段
    slotSegments['订单'].push({
      wd, label: oc.order_list[i] || '?',
      paper_code: oc.summary?.[oc.order_list[i]]?.paper_code || '',
      width: oc.summary?.[oc.order_list[i]]?.width || '',
      time: oc.material_list[i].time,
    });
  }

  // ── 胶水事件段 ──
  for (const pos of ['GU1','GU2','GU3','SF1','SF2']) {
    const events = (data.glue_events?.[pos] || []);
    for (let i = 0; i < events.length; i++) {
      const t = new Date(events[i].time || 0).getTime();
      const nextT = (i + 1 < events.length) ? new Date(events[i + 1].time || 0).getTime() : maxT + 1;
      const wd = Math.max(((nextT - t) / totalDuration) * 100, 0.3);
      const analysis = events[i].analysis || [];
      const verdict = analysis[0]?.verdict || '';
      const material = events[i].material || '';
      slotSegments[pos].push({
        wd, material, verdict, analysis,
        time: events[i].time,
      });
    }
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: '#fff', borderRadius: 8, border: '1px solid #e5e7eb', overflow: 'hidden' }}>
      <div style={{ padding: '10px 16px', background: '#f9fafb', borderBottom: '1px solid #e5e7eb', fontWeight: 600, fontSize: 14 }}>
        订单材质匹配
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        <div style={{ minWidth: Math.max(slotSegments['订单'].length * 120, 800) }}>
          {SLOTS.map((slot, si) => {
            const isTime = slot === '时间';
            const isOrder = slot === '订单';
            const isGlue = slot === 'GU1' || slot === 'GU2' || slot === 'GU3' || slot === 'SF1' || slot === 'SF2';
            const prevSlot = si > 0 ? SLOTS[si - 1] : '';
            const needSep = isGlue && (prevSlot === '' || !['GU1','GU2','GU3','SF1','SF2'].includes(prevSlot));
            return (
            <>
              {/* Glue section separator */}
              {needSep && slot === 'GU1' && (
                <div key="glue-sep" style={{
                  height: 24, display: 'flex', alignItems: 'center',
                  background: '#f0f9ff', borderBottom: '2px solid #bae6fd',
                  paddingLeft: 12, fontSize: 11, fontWeight: 600, color: '#0369a1',
                }}>
                  胶水赋值
                </div>
              )}
            <div key={slot} style={{
              display: 'flex', height: isTime ? 28 : 44,
              borderBottom: isTime ? '2px solid #e5e7eb' : '1px solid #f3f4f6',
            }}>
              <div style={{
                width: 48, minWidth: 48, display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 600, color: '#6b7280', background: '#f9fafb',
                borderRight: '1px solid #e5e7eb', flexShrink: 0,
              }}>
                {isTime ? '' : SLOT_LABELS[slot]}
              </div>
              <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
                {slotSegments[slot].map((seg, j) => {
                  let color, textColor;
                  if (isTime) {
                    color = 'transparent'; textColor = '#9ca3af';
                  } else if (isOrder) {
                    color = '#e5e7eb'; textColor = '#374151';
                  } else if (isGlue) {
                    color = glueVerdictColor(seg.analysis);
                    textColor = glueVerdictTextColor(seg.analysis);
                  } else {
                    color = segmentColor(seg.match, seg.actual);
                    textColor = segmentTextColor(seg.match, seg.actual);
                  }
                  // title
                  let title = seg.time || '';
                  if (isOrder) title = `${seg.label} | ${seg.paper_code || ''} | ${seg.width || ''} | ${seg.time?.slice(11, 26) || ''}`;
                  else if (isGlue) {
                    const v = seg.analysis?.[0]?.verdict || '正常';
                    title = `${slot} | ${seg.material || ''} | ${v} | ${seg.time?.slice(11, 26) || ''}`;
                  } else {
                    title = `${seg.actual} vs ${seg.expected} | ${seg.actualW || 0} vs ${seg.expectedW || 0} | ${seg.match ? '✅ 匹配' : '❌ 不匹配'} | ${seg.time?.slice(11, 26) || ''}`;
                  }
                  return (
                    <div key={j} title={title} style={{
                      width: `${seg.wd}%`, minWidth: 120,
                      background: color, color: textColor,
                      display: 'flex', flexDirection: isGlue ? 'row' : 'column',
                      alignItems: 'center', justifyContent: 'center',
                      fontSize: isTime ? 9 : 11, fontWeight: isTime ? 400 : isOrder ? 600 : 500,
                      fontFamily: 'monospace',
                      whiteSpace: (slot === 'df' || isOrder || isGlue) ? 'normal' : 'nowrap',
                      overflow: 'hidden', textOverflow: 'ellipsis',
                      padding: '1px 3px', lineHeight: 1.2,
                      wordBreak: (slot === 'df' || isOrder || isGlue) ? 'break-all' : undefined,
                      borderRight: j % 5 === 0 && !isTime ? '2px solid #fff' : '1px solid rgba(255,255,255,0.4)',
                      cursor: 'default',
                    }}>
                      {isTime ? (<span>{seg.label}</span>) : isOrder ? (
                        <>
                          <span>{seg.label}</span>
                          {seg.paper_code && (<span style={{ fontSize: 7, opacity: 0.5 }}>{seg.paper_code}</span>)}
                          {seg.width > 0 && (<span style={{ fontSize: 7, opacity: 0.4 }}>{seg.width}</span>)}
                        </>
                      ) : isGlue ? (
                        <>
                          <span style={{ fontSize: 10 }}>{slot}</span>
                          {seg.material && (<span style={{ fontSize: 9, opacity: 0.8, marginLeft: 3 }}>{seg.material}</span>)}
                        </>
                      ) : (
                        <>
                          <span>{seg.actual}</span>
                          {seg.actualW > 0 && (<span style={{ fontSize: 8, opacity: 0.6 }}>{seg.actualW}</span>)}
                          {seg.reason && (
                            <span style={{ fontSize: 7, opacity: 0.5 }}>
                              {{normal:'正常换材',hq:'横切校验',real:'实际材质',reset:'初始化'}[seg.reason] || seg.reason}
                            </span>
                          )}
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
            </>
          )})}
        </div>
      </div>
    </div>
  );
}
