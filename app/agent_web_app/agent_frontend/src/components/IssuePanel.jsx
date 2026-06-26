import { useState } from 'react';

const ISSUE_META = {
  material_dismatch:   { title: '材质不匹配', emoji: '🔴' },
  material_not_exist:  { title: '缺少换材记录', emoji: '🔴' },
  qdm_dismatch:        { title: 'QDM 系数不匹配', emoji: '🔴' },
  qdm_not_exist:       { title: 'QDM 配置不存在', emoji: '🔴' },
  weight_dismatch:     { title: '克重与档案不一致', emoji: '🟠' },
  weight_not_exist:    { title: '无法查询克重档案', emoji: '🟠' },
  basedoc_dismatch:    { title: '基础参数设定不匹配', emoji: '🟡' },
  basedoc_not_exist:   { title: '基础资料缺失', emoji: '🟡' },
  speed_coef_dismatch: { title: '车速系数不匹配', emoji: '🔵' },
  speed_coef_not_exist:{ title: '车速系数配置缺失', emoji: '🔵' },
  no_set_values:       { title: '无计算结果', emoji: '⚪' },
  material_pass:       { title: '材质匹配成功', emoji: '✅' },
  qdm_pass:            { title: 'QDM 系数匹配', emoji: '✅' },
  weight_pass:         { title: '克重匹配', emoji: '✅' },
  basedoc_pass:        { title: '基础设置匹配', emoji: '✅' },
  speed_pass:          { title: '车速系数匹配', emoji: '✅' },
  cancel:              { title: '取消', emoji: '↩️' },
};

const SECTION_META = {
  error:   { label: '错误', color: '#dc2626', icon: '🔴' },
  warning: { label: '警告', color: '#f97316', icon: '🟠' },
  pass:    { label: '通过', color: '#16a34a', icon: '✅' },
};

function Section({ items, sectionKey, meta, open, onToggle }) {
  if (items.length === 0) return null;
  return (
    <div style={{ marginBottom: 8 }}>
      <div
        onClick={onToggle}
        style={{
          fontSize: 13, fontWeight: 600, color: meta.color, marginBottom: 4,
          cursor: 'pointer', userSelect: 'none', display: 'flex', alignItems: 'center', gap: 4,
        }}
      >
        <span style={{ fontSize: 11 }}>{open ? '▼' : '▶'}</span>
        <span>{meta.icon} {meta.label}（{items.length}）</span>
      </div>
      {open && items.map((item, i) => {
        const m = ISSUE_META[item.type] || { title: item.type, emoji: '❓' };
        return (
          <div key={i} style={{
            display: 'flex', alignItems: 'flex-start', gap: 6,
            padding: '4px 8px', fontSize: 13, lineHeight: 1.5,
            borderRadius: 4, marginBottom: 2,
            background: `${meta.color}08`,
          }}>
            <span>{m.emoji}</span>
            <div>
              <strong>{m.title}</strong>
              <br />
              <span style={{ color: '#6b7280' }}>{item.args?.msg || item.detail}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function IssuePanel({ errors = [], warnings = [], passes = [] }) {
  const [sections, setSections] = useState({ error: true, warning: false, pass: false });
  const toggle = (key) => setSections(s => ({ ...s, [key]: !s[key] }));

  if (errors.length === 0 && warnings.length === 0 && passes.length === 0) {
    return <div style={{ color: '#6b7280', fontSize: 13, padding: 8 }}>未发现异常</div>;
  }

  return (
    <div style={{ borderTop: '1px solid #e5e7eb', padding: '12px 0' }}>
      {['error', 'warning', 'pass'].map(key => (
        <Section
          key={key}
          items={key === 'error' ? errors : key === 'warning' ? warnings : passes}
          sectionKey={key}
          meta={SECTION_META[key]}
          open={sections[key]}
          onToggle={() => toggle(key)}
        />
      ))}
    </div>
  );
}
