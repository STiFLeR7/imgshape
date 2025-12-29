import { V4Config, V3Config, AugmentationConfig, ReportConfig } from '../types';

const BASE_URL = 'http://localhost:8000';

class ApiService {
  async checkHealth() {
    try {
      const response = await fetch(`${BASE_URL}/health`);
      if (!response.ok) throw new Error('Health check failed');
      return await response.json();
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

  async getV4Info() {
    const response = await fetch(`${BASE_URL}/v4/info`);
    if (!response.ok) throw new Error(`API Error: ${response.status}`);
    return await response.json();
  }

  async fingerprintV4(file: File | null, datasetPath: string, sampleLimit?: number) {
    const formData = new FormData();
    if (file) formData.append('file', file);
    if (datasetPath) formData.append('dataset_path', datasetPath);
    if (sampleLimit) formData.append('sample_limit', sampleLimit.toString());

    const response = await fetch(`${BASE_URL}/v4/fingerprint`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Fingerprint failed: ${response.status} - ${errorText}`);
    }
    return await response.json();
  }

  async analyzeV4(file: File | null, datasetPath: string, config: V4Config) {
    const formData = new FormData();
    if (file) formData.append('file', file);
    if (datasetPath) formData.append('dataset_path', datasetPath);
    
    formData.append('task', config.task);
    formData.append('deployment', config.deployment);
    formData.append('priority', config.priority);
    
    if (config.maxModelSize) {
      formData.append('max_model_size_mb', config.maxModelSize.toString());
    }

    const response = await fetch(`${BASE_URL}/v4/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Analysis failed: ${response.status} - ${errorText}`);
    }
    return await response.json();
  }

  // Augmentation
  async augment(file: File | null, datasetPath: string, config: AugmentationConfig) {
    const formData = new FormData();
    if (file) formData.append('file', file);
    if (datasetPath) formData.append('dataset_path', datasetPath);

    // Append config parameters
    Object.entries(config).forEach(([key, value]) => {
      formData.append(key, value.toString());
    });

    const response = await fetch(`${BASE_URL}/augment`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Augmentation failed: ${response.status} - ${errorText}`);
    }
    return await response.json();
  }

  // Report Generation
  async generateReport(analysisResults: any, config: ReportConfig) {
    const response = await fetch(`${BASE_URL}/generate_report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        results: analysisResults,
        ...config
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Report generation failed: ${response.status} - ${errorText}`);
    }
    return await response.json();
  }

  // Legacy v3 endpoints
  async analyzeV3(file: File | null, datasetPath: string) {
    const formData = new FormData();
    if (file) formData.append('file', file);
    if (datasetPath) formData.append('dataset_path', datasetPath);

    const response = await fetch(`${BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error(`v3 Analysis failed: ${response.status}`);
    return await response.json();
  }

  async recommendV3(file: File | null, datasetPath: string, prefs: string) {
    const formData = new FormData();
    if (file) formData.append('file', file);
    if (datasetPath) formData.append('dataset_path', datasetPath);
    formData.append('prefs', prefs);

    const response = await fetch(`${BASE_URL}/recommend`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error(`v3 Recommend failed: ${response.status}`);
    return await response.json();
  }

  async compatibilityV3(file: File | null, datasetPath: string, model: string) {
    const formData = new FormData();
    if (file) formData.append('file', file);
    if (datasetPath) formData.append('dataset_path', datasetPath);
    formData.append('model', model);

    const response = await fetch(`${BASE_URL}/compatibility`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error(`v3 Compatibility failed: ${response.status}`);
    return await response.json();
  }
}

export const api = new ApiService();