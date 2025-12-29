import React from 'react';
import { Download, ExternalLink, FileText } from 'lucide-react';
import { ReportConfig } from '../types';

interface ReportViewerProps {
  report: { id: string; content: string; url?: string } | null;
  config: ReportConfig;
  isLoading: boolean;
}

const ReportViewer: React.FC<ReportViewerProps> = ({ report, config, isLoading }) => {
  const handleDownload = () => {
    if (!report) return;
    
    // If URL is provided (e.g. for PDF)
    if (report.url) {
        window.open(report.url, '_blank');
        return;
    }

    // For text based formats
    const blob = new Blob([report.content], { 
      type: config.format === 'html' ? 'text/html' : 'text/markdown' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `report_${report.id}.${config.format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <div className="flex-1 bg-surface border border-gray-800 rounded-xl flex items-center justify-center p-8">
        <div className="flex flex-col items-center space-y-4">
          <div className="w-12 h-12 border-4 border-pink-500/30 border-t-pink-500 rounded-full animate-spin" />
          <p className="text-pink-400 font-medium animate-pulse">Synthesizing report...</p>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="flex-1 bg-surface border border-gray-800 rounded-xl flex flex-col items-center justify-center p-8 text-gray-500 min-h-[400px]">
        <div className="w-16 h-16 bg-surfaceHighlight/50 rounded-full flex items-center justify-center mb-4">
          <FileText className="w-8 h-8 opacity-40" />
        </div>
        <p>No report generated.</p>
        <p className="text-xs mt-2 opacity-60">Complete analysis then generate a report.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-surface border border-gray-800 rounded-xl flex flex-col shadow-sm overflow-hidden h-full">
      <div className="px-4 py-3 border-b border-gray-800 bg-surfaceHighlight/30 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">Report Preview</h3>
        <button
          onClick={handleDownload}
          className="flex items-center space-x-2 text-xs bg-pink-600 hover:bg-pink-700 text-white px-3 py-1.5 rounded-md transition-colors"
        >
          {report.url ? <ExternalLink className="w-3 h-3" /> : <Download className="w-3 h-3" />}
          <span>Download {config.format.toUpperCase()}</span>
        </button>
      </div>
      
      <div className="flex-1 overflow-auto bg-[#0d1117] p-6 custom-scrollbar">
        {config.format === 'html' ? (
          <iframe 
            srcDoc={report.content} 
            className="w-full h-full border-0 bg-white rounded"
            title="Report Preview" 
          />
        ) : config.format === 'pdf' && report.url ? (
           <div className="flex flex-col items-center justify-center h-full text-gray-400">
               <p className="mb-4">PDF Preview not available inline.</p>
               <a href={report.url} target="_blank" rel="noreferrer" className="text-pink-400 underline">Open PDF</a>
           </div>
        ) : (
          <pre className="text-sm font-mono text-gray-300 whitespace-pre-wrap font-sans">
            {report.content}
          </pre>
        )}
      </div>
    </div>
  );
};

export default ReportViewer;