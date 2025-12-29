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

export interface AppState {
  version: 'v3' | 'v4';
  file: File | null;
  filePreviewUrl: string | null;
  datasetPath: string;
  v4Config: V4Config;
  v3Config: V3Config;
  status: 'idle' | 'loading' | 'success' | 'error';
  results: any | null;
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
