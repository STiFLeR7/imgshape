import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import ControlDrawer from './ControlDrawer';
import { AppState } from '../types';
import React from 'react';

const mockState: AppState = {
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
  driftResults: null,
  augmentationResults: null,
  reportResults: null,
  gpuActive: false,
};

describe('ControlDrawer', () => {
  const mockOnClose = vi.fn();
  const mockOnStateChange = vi.fn();
  const mockOnConfigV4Change = vi.fn();
  const mockOnConfigV3Change = vi.fn();

  it('renders when open', () => {
    render(
      <ControlDrawer
        isOpen={true}
        onClose={mockOnClose}
        state={mockState}
        onStateChange={mockOnStateChange}
        onConfigV4Change={mockOnConfigV4Change}
        onConfigV3Change={mockOnConfigV3Change}
      />
    );

    expect(screen.getByText('Configuration')).toBeDefined();
    expect(screen.getByText('API Version')).toBeDefined();
    expect(screen.getByText('v4.0.0 (Atlas)')).toBeDefined();
  });

  it('calls onClose when close button is clicked', () => {
    render(
      <ControlDrawer
        isOpen={true}
        onClose={mockOnClose}
        state={mockState}
        onStateChange={mockOnStateChange}
        onConfigV4Change={mockOnConfigV4Change}
        onConfigV3Change={mockOnConfigV3Change}
      />
    );

    const closeButton = screen.getByLabelText('Close settings');
    fireEvent.click(closeButton);
    expect(mockOnClose).toHaveBeenCalled();
  });

  it('displays v4 config when version is v4', () => {
    render(
      <ControlDrawer
        isOpen={true}
        onClose={mockOnClose}
        state={mockState}
        onStateChange={mockOnStateChange}
        onConfigV4Change={mockOnConfigV4Change}
        onConfigV3Change={mockOnConfigV3Change}
      />
    );

    expect(screen.getByText('Task Type')).toBeDefined();
    expect(screen.getByText('Deployment')).toBeDefined();
    expect(screen.getByText('Priority')).toBeDefined();
  });

  it('displays v3 config when version is v3', () => {
    const v3State = { ...mockState, version: 'v3' as const };
    render(
      <ControlDrawer
        isOpen={true}
        onClose={mockOnClose}
        state={v3State}
        onStateChange={mockOnStateChange}
        onConfigV4Change={mockOnConfigV4Change}
        onConfigV3Change={mockOnConfigV3Change}
      />
    );

    expect(screen.getByText('Preferences')).toBeDefined();
    expect(screen.getByText('Target Model')).toBeDefined();
  });
});
