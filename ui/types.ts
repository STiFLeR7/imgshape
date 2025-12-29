export type LogLevel = 'info' | 'success' | 'warning' | 'error';

export interface LogEntry {
  id: string;
  timestamp: string;
  level: LogLevel;
  message: string;
}

export type TaskType = 'classification' | 'detection' | 'segmentation' | 'generation' | 'other';
export type DeploymentTarget = 'cloud' | 'edge' | 'mobile' | 'embedded' | 'other';
export type Priority = 'accuracy' | 'speed' | 'size' | 'balanced';

export interface V4Config {
  task: TaskType;
  deployment: DeploymentTarget;
  priority: Priority;
  maxModelSize?: number;
}

export interface V3Config {
  prefs: string;
  model: string;
}

export interface AugmentationConfig {
  num_to_generate: number;
  brightness: number;
  contrast: number;
  saturation: number;
  rotation: number;
  color_jitter: boolean;
  rotate: boolean;
  blur: boolean;
  crop: boolean;
}

export interface ReportConfig {
  format: 'markdown' | 'html' | 'pdf';
  include_metadata: boolean;
  include_charts: boolean;
}

export interface AppState {
  version: 'v3' | 'v4';
  activeView: 'dashboard' | 'augmentation' | 'report';
  file: File | null;
  filePreviewUrl: string | null;
  datasetPath: string;
  v4Config: V4Config;
  v3Config: V3Config;
  augmentationConfig: AugmentationConfig;
  reportConfig: ReportConfig;
  status: 'idle' | 'loading' | 'success' | 'error';
  results: any | null;
  augmentationResults: { images: Array<{ base64: string; label: string }> } | null;
  reportResults: { id: string; content: string; url?: string } | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  v4_available: boolean;
}

export interface FingerprintResponse {
  schema_version: string;
  extracted_at: string;
  sample_count: number;
  profiles: {
    spatial: Record<string, any>;
    signal: Record<string, any>;
    distribution: Record<string, any>;
    quality: Record<string, any>;
    semantic: Record<string, any>;
  };
}