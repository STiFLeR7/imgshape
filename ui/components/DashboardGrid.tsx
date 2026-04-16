import React from 'react';

const DashboardGrid: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-12 gap-6 animate-in fade-in duration-700 pb-10">
      {children}
    </div>
  );
};

export default DashboardGrid;
