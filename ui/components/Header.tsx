import React from 'react';
import { Layers, Github, FileText, Activity } from 'lucide-react';

interface HeaderProps {
  serverVersion: string | null;
  serverStatus: boolean;
}

const Header: React.FC<HeaderProps> = ({ serverVersion, serverStatus }) => {
  return (
    <header className="h-16 flex items-center justify-between px-6 border-b border-gray-800 bg-surface/80 backdrop-blur-md sticky top-0 z-50">
      <div className="flex items-center space-x-4">
        <div className="flex items-center gap-3">
          {/* Logo replacement */}
          <img src="imgshape.png" alt="imgshape logo" className="h-8 w-auto object-contain" onError={(e) => {
             // Fallback if image fails to load
             e.currentTarget.style.display = 'none';
             e.currentTarget.nextElementSibling?.classList.remove('hidden');
          }}/>
          <div className="hidden p-2 bg-accent/20 rounded-lg">
             <Layers className="w-6 h-6 text-accent" />
          </div>
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            imgshape 
            <span className="text-xs font-normal px-2 py-0.5 rounded-full bg-gray-800 text-gray-400 border border-gray-700">Atlas</span>
          </h1>
        </div>
        {serverVersion && (
           <span className="hidden md:inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-green-900/30 text-green-400 border border-green-900/50">
             v{serverVersion}
           </span>
        )}
      </div>

      <div className="flex items-center space-x-6 text-sm font-medium text-gray-400">
        <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${serverStatus ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 'bg-red-500'}`}></div>
            <span className={serverStatus ? 'text-green-500' : 'text-red-500'}>
                {serverStatus ? 'System Online' : 'Disconnected'}
            </span>
        </div>
        <a href="https://github.com/STiFLeR7/imgshape/blob/master/README.md" target="_blank" rel="noopener noreferrer" className="flex items-center hover:text-white transition-colors">
          <FileText className="w-4 h-4 mr-2" />
          Docs
        </a>
        <a href="https://github.com/STiFLeR7/imgshape" target="_blank" rel="noopener noreferrer" className="flex items-center hover:text-white transition-colors">
          <Github className="w-4 h-4 mr-2" />
          GitHub
        </a>
      </div>
    </header>
  );
};

export default Header;