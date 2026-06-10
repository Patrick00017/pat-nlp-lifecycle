import './Tabs.css'

export default function Tabs({ tabs, activeTab, onChange, children }) {
  return (
    <div className="app-layout">
      <div className="tabs-header">
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
      <div className="tabs-content">
        {children}
      </div>
    </div>
  )
}
