import { render, screen } from '@testing-library/react';
import { Activity } from 'lucide-react';
import { describe, it, expect } from 'vitest';
import KPICard from './KPICard';

describe('KPICard', () => {
  it('renders label and value', () => {
    render(
      <KPICard 
        label="Total Images" 
        value="1,234" 
        icon={Activity} 
      />
    );
    
    expect(screen.getByText('Total Images')).toBeInTheDocument();
    expect(screen.getByText('1,234')).toBeInTheDocument();
  });

  it('renders subtext when provided', () => {
    render(
      <KPICard 
        label="Total Images" 
        value="1,234" 
        subtext="+10% from last scan"
        icon={Activity} 
      />
    );
    
    expect(screen.getByText('+10% from last scan')).toBeInTheDocument();
  });

  it('renders loading state', () => {
    render(
      <KPICard 
        label="Total Images" 
        value="1,234" 
        icon={Activity} 
        loading={true}
      />
    );
    
    // Value should not be visible when loading
    expect(screen.queryByText('1,234')).not.toBeInTheDocument();
  });
});
