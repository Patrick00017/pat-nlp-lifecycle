import FSMViewer from './FSMViewer';

export default function FSMTab() {
  return (
    <div className="container">
      <h1>事件诊断</h1>
      <div style={{ background: '#fff', borderRadius: 12, border: '1px solid #e2e8f0', padding: 16 }}>
        <FSMViewer />
      </div>
    </div>
  );
}
