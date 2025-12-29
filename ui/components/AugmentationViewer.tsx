import React from 'react';
import { Download, Image as ImageIcon } from 'lucide-react';

interface AugmentationViewerProps {
  results: { images: Array<{ base64: string; label: string }> } | null;
  isLoading: boolean;
}

const AugmentationViewer: React.FC<AugmentationViewerProps> = ({ results, isLoading }) => {
  const handleDownload = (base64: string, index: number) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${base64}`;
    link.download = `augmented_${index + 1}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (isLoading) {
    return (
      <div className="flex-1 bg-surface border border-gray-800 rounded-xl flex items-center justify-center p-8">
        <div className="flex flex-col items-center space-y-4">
          <div className="w-12 h-12 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin" />
          <p className="text-purple-400 font-medium animate-pulse">Running augmentation pipeline...</p>
        </div>
      </div>
    );
  }

  if (!results || !results.images || results.images.length === 0) {
    return (
      <div className="flex-1 bg-surface border border-gray-800 rounded-xl flex flex-col items-center justify-center p-8 text-gray-500 min-h-[400px]">
        <div className="w-16 h-16 bg-surfaceHighlight/50 rounded-full flex items-center justify-center mb-4">
          <ImageIcon className="w-8 h-8 opacity-40" />
        </div>
        <p>No augmented images generated yet.</p>
        <p className="text-xs mt-2 opacity-60">Configure parameters and click Generate Batch.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-surface border border-gray-800 rounded-xl p-6 overflow-y-auto custom-scrollbar min-h-[400px]">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-gray-300 font-medium">Generated Batch ({results.images.length})</h3>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {results.images.map((img, idx) => (
          <div key={idx} className="group relative bg-black/40 rounded-lg overflow-hidden border border-gray-800 hover:border-purple-500/50 transition-all aspect-square">
            <img 
              src={`data:image/png;base64,${img.base64}`} 
              alt={`Augmented ${idx}`}
              className="w-full h-full object-contain p-2"
            />
            <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
              <button
                onClick={() => handleDownload(img.base64, idx)}
                className="bg-white text-black p-2 rounded-full hover:bg-gray-200 transition-transform hover:scale-110"
                title="Download"
              >
                <Download className="w-5 h-5" />
              </button>
            </div>
            <div className="absolute bottom-0 left-0 right-0 bg-black/80 px-2 py-1 text-[10px] text-gray-400 truncate">
              {img.label || `Augmented ${idx + 1}`}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AugmentationViewer;