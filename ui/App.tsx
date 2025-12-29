import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import PreviewPanel from './components/PreviewPanel';
import ResultsPanel from './components/ResultsPanel';
import LogsPanel from './components/LogsPanel';
import AugmentationPanel from './components/AugmentationPanel';
import AugmentationViewer from './components/AugmentationViewer';
import ReportGenerator from './components/ReportGenerator';
import ReportViewer from './components/ReportViewer';
import { api } from './services/api';
import { AppState, V4Config, V3Config, LogEntry, AugmentationConfig, ReportConfig } from './types';

const App: React.FC = () => {
  const [state, setState] = useState<AppState>({
    version: 'v4',
    activeView: 'dashboard',
    file: null,
    filePreviewUrl: null,
    datasetPath: '',
    v4Config: {
      task: 'classification',
      deployment: 'cloud',
      priority: 'balanced',
    },
    v3Config: {
      prefs: '',
      model: '',
    },
    augmentationConfig: {
      num_to_generate: 4,
      brightness: 0.5,
      contrast: 0.5,
      saturation: 0.5,
      rotation: 15,
      color_jitter: false,
      rotate: true,
      blur: false,
      crop: false,
    },
    reportConfig: {
      format: 'markdown',
      include_metadata: true,
      include_charts: false,
    },
    status: 'idle',
    results: null,
    augmentationResults: null,
    reportResults: null,
  });

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [serverStatus, setServerStatus] = useState(false);
  const [serverVersion, setServerVersion] = useState<string | null>(null);

  const addLog = (level: LogEntry['level'], message: string) => {
    const entry: LogEntry = {
      id: Math.random().toString(36).substr(2, 9),
      timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
      level,
      message,
    };
    setLogs((prev) => [...prev, entry]);
  };

  useEffect(() => {
    const init = async () => {
      try {
        addLog('info', 'Connecting to Atlas server...');
        const health = await api.checkHealth();
        setServerStatus(true);
        setServerVersion(health.version);
        addLog('success', `Connected to imgshape v${health.version}`);
        
        if (health.v4_available) {
            const v4info = await api.getV4Info();
            addLog('info', `Loaded v4 capabilities: [${v4info.features.join(', ')}]`);
        }
      } catch (err) {
        setServerStatus(false);
        addLog('error', 'Failed to connect to backend server. Is it running on port 8000?');
      }
    };
    init();
  }, []);

  const handleStateChange = (updates: Partial<AppState>) => {
    setState((prev) => ({ ...prev, ...updates }));
  };

  const handleV4ConfigChange = (updates: Partial<V4Config>) => {
    setState((prev) => ({ ...prev, v4Config: { ...prev.v4Config, ...updates } }));
  };

  const handleV3ConfigChange = (updates: Partial<V3Config>) => {
    setState((prev) => ({ ...prev, v3Config: { ...prev.v3Config, ...updates } }));
  };

  const handleAugmentationConfigChange = (updates: Partial<AugmentationConfig>) => {
    setState((prev) => ({ ...prev, augmentationConfig: { ...prev.augmentationConfig, ...updates } }));
  };

  const handleReportConfigChange = (updates: Partial<ReportConfig>) => {
    setState((prev) => ({ ...prev, reportConfig: { ...prev.reportConfig, ...updates } }));
  };

  const executeAction = async (actionName: string, actionFn: () => Promise<any>, successCallback?: (data: any) => void) => {
    setState(prev => ({ ...prev, status: 'loading' }));
    addLog('info', `Starting ${actionName}...`);

    try {
      const data = await actionFn();
      
      if (successCallback) {
        successCallback(data);
      } else {
        setState(prev => ({ ...prev, status: 'success', results: data }));
      }
      
      addLog('success', `${actionName} completed successfully.`);
      
      if (data.profiles) {
          addLog('info', `Fingerprint extracted: ${Object.keys(data.profiles).length} profiles`);
      }
    } catch (err: any) {
      console.error(err);
      setState(prev => ({ ...prev, status: 'error' }));
      addLog('error', `${actionName} failed: ${err.message}`);
    }
  };

  const handleAnalyze = () => {
    if (!state.file && !state.datasetPath) {
      addLog('warning', 'Please provide a file or dataset path.');
      return;
    }
    if (state.version === 'v4') {
        executeAction('v4 Analysis', () => api.analyzeV4(state.file, state.datasetPath, state.v4Config));
    } else {
        executeAction('v3 Analysis', () => api.analyzeV3(state.file, state.datasetPath));
    }
  };

  const handleFingerprint = () => {
    if (!state.file && !state.datasetPath) {
      addLog('warning', 'Please provide a file or dataset path.');
      return;
    }
    executeAction('v4 Fingerprint', () => api.fingerprintV4(state.file, state.datasetPath));
  };

  const handleRecommend = () => {
    if (!state.file && !state.datasetPath) {
      addLog('warning', 'Please provide a file or dataset path.');
      return;
    }
    executeAction('v3 Recommendation', () => api.recommendV3(state.file, state.datasetPath, state.v3Config.prefs));
  };

  const handleAugment = () => {
    if (!state.file && !state.datasetPath) {
      addLog('warning', 'Please provide a file or dataset path for augmentation.');
      return;
    }
    executeAction(
      'Augmentation Batch', 
      () => api.augment(state.file, state.datasetPath, state.augmentationConfig),
      (data) => setState(prev => ({ ...prev, status: 'success', augmentationResults: data }))
    );
  };

  const handleGenerateReport = () => {
    if (!state.results) {
      addLog('warning', 'No analysis results available for report.');
      return;
    }
    executeAction(
      'Report Generation',
      () => api.generateReport(state.results, state.reportConfig),
      (data) => setState(prev => ({ ...prev, status: 'success', reportResults: data }))
    );
  };

  const handleDownloadJson = () => {
      if (!state.results) return;
      const jsonString = JSON.stringify(state.results, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `imgshape_results_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      addLog('success', 'Results downloaded.');
  };

  const handleCopyJson = () => {
      if (!state.results) return;
      navigator.clipboard.writeText(JSON.stringify(state.results, null, 2));
      addLog('success', 'JSON copied to clipboard.');
  };

  const renderContent = () => {
    switch (state.activeView) {
      case 'augmentation':
        return (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full min-h-[400px]">
            <div className="lg:col-span-1">
              <AugmentationPanel 
                config={state.augmentationConfig}
                onChange={handleAugmentationConfigChange}
                onExecute={handleAugment}
                isLoading={state.status === 'loading'}
                disabled={!state.file && !state.datasetPath}
              />
            </div>
            <div className="lg:col-span-2 h-full flex flex-col">
              <AugmentationViewer 
                results={state.augmentationResults}
                isLoading={state.status === 'loading'}
              />
            </div>
          </div>
        );
      case 'report':
        return (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full min-h-[400px]">
             <div className="lg:col-span-1">
               <ReportGenerator
                 config={state.reportConfig}
                 onChange={handleReportConfigChange}
                 onGenerate={handleGenerateReport}
                 isLoading={state.status === 'loading'}
                 hasResults={!!state.results}
               />
             </div>
             <div className="lg:col-span-2 h-full flex flex-col">
               <ReportViewer
                 report={state.reportResults}
                 config={state.reportConfig}
                 isLoading={state.status === 'loading'}
               />
             </div>
          </div>
        );
      case 'dashboard':
      default:
        return (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 min-h-[400px]">
            <PreviewPanel 
                file={state.file}
                imageUrl={state.filePreviewUrl}
                datasetPath={state.datasetPath}
                isLoading={state.status === 'loading'}
            />
            <ResultsPanel 
                data={state.results}
                status={state.status}
            />
          </div>
        );
    }
  };

  return (
    <div className="flex flex-col h-screen bg-background text-gray-100 overflow-hidden">
      <Header serverVersion={serverVersion} serverStatus={serverStatus} />
      
      <main className="flex flex-1 overflow-hidden">
        <Sidebar 
          state={state}
          onStateChange={handleStateChange}
          onConfigV4Change={handleV4ConfigChange}
          onConfigV3Change={handleV3ConfigChange}
          onAnalyze={handleAnalyze}
          onFingerprint={handleFingerprint}
          onRecommend={handleRecommend}
          onDownloadJson={handleDownloadJson}
          onCopyJson={handleCopyJson}
        />

        <div className="flex-1 flex flex-col p-6 space-y-6 overflow-y-auto custom-scrollbar">
            {/* Main Content Area */}
            <div className="flex-1 min-h-[400px] flex flex-col">
              {renderContent()}
            </div>
            
            {/* Logs Area - Always Visible */}
            <div className="flex-none h-48 md:h-64">
                <LogsPanel logs={logs} />
            </div>
        </div>
      </main>
    </div>
  );
};

export default App;