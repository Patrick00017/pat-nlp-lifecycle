const ISSUE_META = {
  material_dismatch:   { severity: 'error',   title: '材质不匹配', emoji: '🔴' },
  material_not_exist:  { severity: 'error',   title: '缺少换材记录', emoji: '🔴' },
  qdm_dismatch:        { severity: 'error',   title: 'QDM 系数不匹配', emoji: '🔴' },
  qdm_not_exist:       { severity: 'error',   title: 'QDM 配置不存在', emoji: '🔴' },
  weight_dismatch:     { severity: 'warning', title: '克重与档案不一致', emoji: '🟠' },
  weight_not_exist:    { severity: 'warning', title: '无法查询克重档案', emoji: '🟠' },
  basedoc_dismatch:    { severity: 'warning', title: '基础参数设定不匹配', emoji: '🟡' },
  basedoc_not_exist:   { severity: 'warning', title: '基础资料缺失', emoji: '🟡' },
  speed_coef_dismatch: { severity: 'info',    title: '车速系数不匹配', emoji: '🔵' },
  speed_coef_not_exist:{ severity: 'info',    title: '车速系数配置缺失', emoji: '🔵' },
  no_set_values:       { severity: 'info',    title: '无计算结果', emoji: '⚪' },
};

const GROUP_LABEL = {
  error:   { label: '错误', color: '#dc2626' },
  warning: { label: '警告', color: '#f97316' },
  info:    { label: '提示', color: '#3b82f6' },
};

export default function IssuePanel({ issues = [] }) {
  if (issues.length === 0) {
    return <div style={{ color: '#6b7280', fontSize: 13, padding: 8 }}>未发现异常</div>;
  }

  const groups = { error: [], warning: [], info: [] };
  for (const is of issues) {
    const meta = ISSUE_META[is.type] || { severity: 'info', title: is.type, emoji: '❓' };
    groups[meta.severity].push({ ...is, meta });
  }

  return (
    <div style={{ borderTop: '1px solid #e5e7eb', padding: '12px 0' }}>
      {['error', 'warning', 'info'].map(sev => {
        if (groups[sev].length === 0) return null;
        const gl = GROUP_LABEL[sev];
        return (
          <div key={sev} style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: gl.color, marginBottom: 4 }}>
              {gl.label} ({groups[sev].length})
            </div>
            {groups[sev].map((item, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'flex-start', gap: 6,
                padding: '4px 8px', fontSize: 13, lineHeight: 1.5,
                borderRadius: 4, marginBottom: 2,
                background: `${gl.color}08`,
              }}>
                <span>{item.meta.emoji}</span>
                <div>
                  <strong>{item.meta.title}</strong>
                  <br />
                  <span style={{ color: '#6b7280' }}>{item.detail}</span>
                </div>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}
