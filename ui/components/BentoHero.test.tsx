import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import BentoHero from './BentoHero';
import React from 'react';

describe('BentoHero', () => {
  it('renders real branding and version', () => {
    render(<BentoHero />);
    expect(screen.getByText(/imgshape/i)).toBeDefined();
    expect(screen.getByText(/v4.2.0 NOW STABLE/i)).toBeDefined();
  });
});
