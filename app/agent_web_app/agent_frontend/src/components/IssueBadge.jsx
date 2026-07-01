const ISSUE_STYLES = {
  material_dismatch:   { color: '#dc2626', bg: '#fef2f2', icon: '❌', label: '换材' },
  material_not_exist:  { color: '#dc2626', bg: '#fef2f2', icon: '❌', label: '换材' },
  qdm_dismatch:        { color: '#dc2626', bg: '#fef2f2', icon: '❌', label: 'QDM' },
  qdm_not_exist:       { color: '#dc2626', bg: '#fef2f2', icon: '❌', label: 'QDM' },
  weight_dismatch:     { color: '#f97316', bg: '#fff7ed', icon: '⚠', label: '克重' },
  weight_not_exist:    { color: '#f97316', bg: '#fff7ed', icon: '⚠', label: '克重' },
  basedoc_dismatch:    { color: '#eab308', bg: '#fefce8', icon: '⚠', label: '基础设置' },
  basedoc_not_exist:   { color: '#eab308', bg: '#fefce8', icon: '⚠', label: '基础资料' },
  speed_coef_dismatch: { color: '#8b5cf6', bg: '#f5f3ff', icon: 'ℹ', label: '车速系数' },
  speed_coef_not_exist:{ color: '#8b5cf6', bg: '#f5f3ff', icon: 'ℹ', label: '车速系数' },
  no_set_values:       { color: '#9ca3af', bg: '#f9fafb', icon: '💤', label: '无数据' },
  cancel:              { color: '#eab308', bg: '#fefce8', icon: '⚠️', label: '部位未启用' },
};

export default function IssueBadge({ type, detail, args }) {
  const style = ISSUE_STYLES[type] || { color: '#6b7280', bg: '#f3f4f6', icon: '?', label: type };
  const displayText = args?.msg || style.label;
  return (
    <span
      title={detail}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 4,
        fontSize: 12, fontWeight: 500,
        color: style.color, background: style.bg,
        padding: '2px 8px', borderRadius: 4,
        border: `1px solid ${style.color}22`,
        cursor: 'default',
      }}
    >
      <span style={{ fontSize: 13 }}>{style.icon}</span>
      <span style={{ fontSize: 11, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{displayText}</span>
    </span>
  );
}
