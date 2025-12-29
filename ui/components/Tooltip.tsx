import React, { ReactNode } from 'react';

interface TooltipProps {
  text: string;
  children: ReactNode;
  position?: 'right' | 'top' | 'bottom' | 'left';
  className?: string;
}

export const Tooltip: React.FC<TooltipProps> = ({ text, children, position = 'right', className = '' }) => {
  const positionClasses = {
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
  };

  return (
    <div className={`group relative flex w-full ${className}`}>
      {children}
      <div className={`pointer-events-none absolute ${positionClasses[position]} w-max max-w-[200px] opacity-0 transition-opacity duration-200 group-hover:opacity-100 z-[100]`}>
        <div className="px-2 py-1.5 text-xs font-medium text-gray-200 bg-gray-900 rounded-md shadow-xl border border-gray-700 whitespace-normal text-center relative z-50">
          {text}
        </div>
      </div>
    </div>
  );
};