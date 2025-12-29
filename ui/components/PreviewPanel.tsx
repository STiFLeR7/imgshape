import React from 'react';
import { Image as ImageIcon, FileDigit } from 'lucide-react';

interface PreviewPanelProps {
  file: File | null;
  imageUrl: string | null;
  datasetPath: string;
  isLoading: boolean;
}

const PreviewPanel: React.FC<PreviewPanelProps> = ({ file, imageUrl, datasetPath, isLoading }) => {
  return (
    <div className="bg-surface border border-gray-800 rounded-xl overflow-hidden shadow-2xl relative min-h-[300px] flex flex-col">
      <div className="px-4 py-3 border-b border-gray-800 bg-surfaceHighlight/50 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center">
          <ImageIcon className="w-4 h-4 mr-2 text-accent" />
          Preview
        </h3>
        {file && (
          <span className="text-xs text-gray-500 font-mono bg-gray-900 px-2 py-1 rounded border border-gray-800">
            {file.name}
          </span>
        )}
      </div>

      <div className="flex-1 bg-gray-900/50 flex items-center justify-center p-8 relative">
        {isLoading && (
          <div className="absolute inset-0 bg-black/60 z-10 flex flex-col items-center justify-center backdrop-blur-sm">
            <div className="w-64 h-2 bg-gray-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-accent to-purple-500 animate-shimmer w-1/2 rounded-full" />
            </div>
            <p className="mt-4 text-sm text-accent font-medium animate-pulse">Processing dataset...</p>
          </div>
        )}

        {imageUrl ? (
          <div className="relative group max-w-full max-h-[400px]">
            <img
              src={imageUrl}
              alt="Preview"
              className="max-w-full max-h-[400px] object-contain rounded-lg shadow-lg border border-gray-800"
            />
            <div className="absolute bottom-2 left-2 right-2 bg-black/80 backdrop-blur-md px-3 py-2 rounded-lg text-xs text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity flex justify-between">
              <span>{file?.type || 'Image'}</span>
              <span>{(file?.size ? (file.size / 1024).toFixed(1) : 0)} KB</span>
            </div>
          </div>
        ) : datasetPath ? (
           <div className="text-center">
             <div className="w-16 h-16 bg-surfaceHighlight rounded-xl flex items-center justify-center mx-auto mb-4 border border-gray-700">
                <FileDigit className="w-8 h-8 text-gray-400" />
             </div>
             <h4 className="text-gray-300 font-medium">Server Dataset</h4>
             <p className="text-sm text-gray-500 mt-1 font-mono">{datasetPath}</p>
           </div>
        ) : (
          <div className="text-center text-gray-600">
             <div className="w-16 h-16 bg-surfaceHighlight/30 rounded-full flex items-center justify-center mx-auto mb-4">
                <ImageIcon className="w-6 h-6 opacity-50" />
             </div>
             <p className="text-sm">No image selected</p>
          </div>
        )}
      </div>
      
      {/* Decorative gradient line at bottom */}
      <div className="h-0.5 w-full bg-gradient-to-r from-transparent via-accent/50 to-transparent opacity-50" />
    </div>
  );
};

export default PreviewPanel;