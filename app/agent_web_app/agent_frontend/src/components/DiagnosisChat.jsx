import FSMViewer from './FSMViewer';
import OpenCodeChat from './OpenCodeChat';

export default function DiagnosisChat({ sharedThreadId, setSharedThreadId }) {
  return (
    <div style={{
      display: 'flex', height: '100%', gap: 12, padding: '12px 16px',
      overflow: 'hidden',
    }}>
      <div style={{ flex: 1, minWidth: 0, overflow: 'auto' }}>
        <FSMViewer />
      </div>
      <div style={{ width: 420, flexShrink: 0, display: 'flex', flexDirection: 'column' }}>
        <OpenCodeChat sharedThreadId={sharedThreadId} setSharedThreadId={setSharedThreadId} />
      </div>
    </div>
  );
}
