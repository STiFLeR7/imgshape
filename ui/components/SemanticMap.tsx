import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip as ReTooltip, ResponsiveContainer, Cell } from 'recharts';
import { Brain } from 'lucide-react';

interface SemanticMapProps {
  embedding?: number[];
  loading?: boolean;
}

const SemanticMap: React.FC<SemanticMapProps> = ({ embedding, loading }) => {
  // Generate pseudo-points around the centroid for visualization if we only have centroid
  const data = embedding ? embedding.slice(0, 100).map((val, i) => ({
    x: Math.cos(i) * val + (Math.random() - 0.5),
    y: Math.sin(i) * val + (Math.random() - 0.5),
    value: val
  })) : [];

  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 h-full flex flex-col">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Brain className="w-4 h-4 text-purple-400" />
            Semantic Signature Map
        </h3>
        <div className="flex gap-2 text-[9px] font-bold uppercase tracking-tighter">
            <span className="px-2 py-0.5 rounded bg-slate-900 text-slate-500 border border-slate-800">DINOv2 L/14</span>
            <span className="px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">Active</span>
        </div>
      </div>
      
      <div className="flex-1 min-h-[250px] relative">
        {loading ? (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-900/20 backdrop-blur-[2px] rounded-lg">
                <div className="flex flex-col items-center gap-3">
                    <div className="w-8 h-8 border-2 border-purple-500/30 border-t-purple-500 rounded-full animate-spin" />
                    <span className="text-[10px] font-bold text-purple-400 uppercase tracking-widest">Projecting Embeddings...</span>
                </div>
            </div>
        ) : embedding ? (
            <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                    <XAxis type="number" dataKey="x" hide />
                    <YAxis type="number" dataKey="y" hide />
                    <ZAxis type="number" dataKey="value" range={[20, 100]} />
                    <ReTooltip 
                        cursor={{ strokeDasharray: '3 3' }} 
                        content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                                return (
                                    <div className="bg-slate-900 border border-slate-700 p-2 rounded text-[10px] font-mono text-purple-300">
                                        Value: {payload[0].value.toFixed(4)}
                                    </div>
                                );
                            }
                            return null;
                        }}
                    />
                    <Scatter name="Embeddings" data={data}>
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={`rgba(168, 85, 247, ${0.3 + (entry.value / 10)})`} />
                        ))}
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>
        ) : (
            <div className="absolute inset-0 flex items-center justify-center border-2 border-dashed border-slate-800 rounded-lg">
                <span className="text-slate-600 text-xs font-medium italic">No semantic data available</span>
            </div>
        )}
      </div>
    </div>
  );
};

export default SemanticMap;
