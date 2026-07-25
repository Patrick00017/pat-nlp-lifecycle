import './Tabs.css'

export default function Tabs({ tabs, activeTab, onChange, children, headerRight }) {
  return (
    <div className="app-layout">
      <div className="tabs-header">
        <div className="tabs-header-left">
          {tabs.map(tab => (
            <button
              key={tab.key}
              className={`tab-btn ${activeTab === tab.key ? 'active' : ''}`}
              onClick={() => onChange(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        {headerRight && <div className="tabs-header-right">{headerRight}</div>}
      </div>
      <div className="tabs-content">
        {children}
      </div>
    </div>
  )
}
