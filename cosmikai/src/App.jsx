import React, { useState } from 'react';
import { Upload, FileText, BarChart3, Settings, Zap, TrendingUp, AlertCircle, CheckCircle, XCircle, Eye, Download } from 'lucide-react';

const ExoplanetDetectionApp = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [detectionResults, setDetectionResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelStats, setModelStats] = useState({
    accuracy: 0.87,
    totalPredictions: 1247,
    planetsFound: 342,
    falsePositives: 23
  });

  // Mock detection function - replace with actual API call
  const runDetection = async () => {
    setIsProcessing(true);
    
    // Simulate API processing time
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Mock results - replace with actual API response
    const mockResults = {
      planet_detected: Math.random() > 0.4,
      confidence: Math.random() * 0.4 + 0.6, // 0.6-1.0
      period_days: Math.random() * 10 + 1,
      transit_depth_ppm: Math.random() * 5000 + 1000,
      planet_radius_earth: Math.random() * 3 + 0.5,
      processing_time_seconds: 1.8,
      lightcurve_data: Array.from({length: 100}, (_, i) => ({
        time: i * 0.1,
        flux: 1.0 + (Math.random() - 0.5) * 0.01 - (i > 30 && i < 40 ? 0.02 : 0)
      }))
    };
    
    setDetectionResults(mockResults);
    setIsProcessing(false);
    setModelStats(prev => ({
      ...prev,
      totalPredictions: prev.totalPredictions + 1,
      planetsFound: prev.planetsFound + (mockResults.planet_detected ? 1 : 0)
    }));
  };

  const FileUploadArea = () => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="border-2 border-dashed border-blue-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
        <Upload className="mx-auto h-12 w-12 text-blue-400 mb-4" />
        <div className="space-y-2">
          <h3 className="text-lg font-medium text-gray-900">Upload Light Curve Data</h3>
          <p className="text-gray-500">Drag and drop your CSV or FITS files, or click to browse</p>
          <p className="text-sm text-gray-400">Supported formats: CSV, FITS, TSV</p>
        </div>
        <input
          type="file"
          className="mt-4 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          accept=".csv,.fits,.tsv"
          onChange={(e) => {
            setUploadedFile(e.target.files[0]);
            setDetectionResults(null);
          }}
        />
      </div>
      
      {uploadedFile && (
        <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center">
            <FileText className="h-5 w-5 text-green-600 mr-2" />
            <span className="text-green-800 font-medium">{uploadedFile.name}</span>
            <span className="ml-auto text-green-600 text-sm">
              {(uploadedFile.size / 1024).toFixed(1)} KB
            </span>
          </div>
          <button
            onClick={runDetection}
            disabled={isProcessing}
            className="mt-4 w-full bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {isProcessing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Analyzing Light Curve...
              </>
            ) : (
              <>
                <Zap className="h-4 w-4 mr-2" />
                Detect Exoplanets
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );

  const DetectionResults = () => {
    if (!detectionResults) return null;

    return (
      <div className="space-y-6">
        {/* Main Result Card */}
        <div className={`rounded-lg shadow-md p-6 ${detectionResults.planet_detected ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'} border`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              {detectionResults.planet_detected ? (
                <CheckCircle className="h-8 w-8 text-green-600 mr-3" />
              ) : (
                <XCircle className="h-8 w-8 text-gray-600 mr-3" />
              )}
              <div>
                <h2 className="text-2xl font-bold text-gray-900">
                  {detectionResults.planet_detected ? '🪐 Exoplanet Detected!' : 'No Exoplanet Found'}
                </h2>
                <p className="text-gray-600">
                  Confidence: {(detectionResults.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">Processing Time</div>
              <div className="text-lg font-semibold">{detectionResults.processing_time_seconds}s</div>
            </div>
          </div>
        </div>

        {/* Planet Parameters */}
        {detectionResults.planet_detected && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <BarChart3 className="h-5 w-5 mr-2 text-blue-600" />
              Detected Planet Parameters
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {detectionResults.period_days.toFixed(2)}
                </div>
                <div className="text-sm text-gray-600">Orbital Period (days)</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {detectionResults.transit_depth_ppm.toFixed(0)}
                </div>
                <div className="text-sm text-gray-600">Transit Depth (ppm)</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">
                  {detectionResults.planet_radius_earth.toFixed(2)}
                </div>
                <div className="text-sm text-gray-600">Planet Radius (Earth = 1)</div>
              </div>
            </div>
          </div>
        )}

        {/* Light Curve Visualization */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <TrendingUp className="h-5 w-5 mr-2 text-green-600" />
            Light Curve Analysis
          </h3>
          <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center border-2 border-dashed border-gray-300">
            <div className="text-center text-gray-500">
              <Eye className="h-8 w-8 mx-auto mb-2" />
              <p>Interactive light curve visualization</p>
              <p className="text-sm">(Chart.js/D3 integration goes here)</p>
            </div>
          </div>
          <div className="mt-4 flex justify-between items-center text-sm text-gray-600">
            <span>Data points: {detectionResults.lightcurve_data?.length || 0}</span>
            <button className="flex items-center text-blue-600 hover:text-blue-800">
              <Download className="h-4 w-4 mr-1" />
              Export Data
            </button>
          </div>
        </div>
      </div>
    );
  };

  const ModelStatistics = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <BarChart3 className="h-5 w-5 mr-2 text-blue-600" />
          Model Performance
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {(modelStats.accuracy * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Accuracy</div>
          </div>
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {modelStats.totalPredictions}
            </div>
            <div className="text-sm text-gray-600">Total Predictions</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">
              {modelStats.planetsFound}
            </div>
            <div className="text-sm text-gray-600">Planets Found</div>
          </div>
          <div className="text-center p-4 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-600">
              {modelStats.falsePositives}
            </div>
            <div className="text-sm text-gray-600">False Positives</div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Detection History</h3>
        <div className="space-y-3">
          {[
            { file: 'kepler_001.csv', result: 'Planet Detected', confidence: 0.92, time: '2 min ago' },
            { file: 'tess_047.fits', result: 'No Planet', confidence: 0.34, time: '15 min ago' },
            { file: 'k2_data.csv', result: 'Planet Detected', confidence: 0.78, time: '1 hour ago' },
            { file: 'lightcurve_x.csv', result: 'No Planet', confidence: 0.12, time: '2 hours ago' },
          ].map((item, index) => (
            <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center">
                <FileText className="h-4 w-4 text-gray-400 mr-2" />
                <span className="font-medium">{item.file}</span>
              </div>
              <div className="flex items-center space-x-4 text-sm">
                <span className={item.result === 'Planet Detected' ? 'text-green-600' : 'text-gray-600'}>
                  {item.result}
                </span>
                <span className="text-gray-500">{(item.confidence * 100).toFixed(0)}%</span>
                <span className="text-gray-400">{item.time}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const ModelSettings = () => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <Settings className="h-5 w-5 mr-2 text-gray-600" />
        Model Configuration
      </h3>
      <div className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Detection Threshold
          </label>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.1"
            defaultValue="0.5"
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-sm text-gray-500 mt-1">
            <span>Conservative</span>
            <span>Aggressive</span>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Model Type
          </label>
          <select className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
            <option>Random Forest (Current)</option>
            <option>Convolutional Neural Network</option>
            <option>Ensemble Method</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Feature Selection
          </label>
          <div className="space-y-2">
            {['Transit Depth', 'Orbital Period', 'Signal-to-Noise Ratio', 'Transit Duration'].map((feature) => (
              <label key={feature} className="flex items-center">
                <input
                  type="checkbox"
                  defaultChecked
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="ml-2 text-sm text-gray-700">{feature}</span>
              </label>
            ))}
          </div>
        </div>

        <button className="w-full bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
          Update Model Configuration
        </button>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <div className="text-2xl mr-3">🪐</div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Cosmik AI</h1>
                <p className="text-sm text-gray-600">Automated Exoplanet Detection System</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                Model Status: <span className="text-green-600 font-medium">Active</span>
              </div>
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'upload', label: 'Upload & Detect', icon: Upload },
              { id: 'statistics', label: 'Statistics', icon: BarChart3 },
              { id: 'settings', label: 'Model Settings', icon: Settings },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center px-3 py-4 text-sm font-medium border-b-2 ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <tab.icon className="h-4 w-4 mr-2" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'upload' && (
          <div className="space-y-8">
            <FileUploadArea />
            <DetectionResults />
          </div>
        )}
        {activeTab === 'statistics' && <ModelStatistics />}
        {activeTab === 'settings' && <ModelSettings />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex justify-between items-center text-sm text-gray-600">
            <div>
              Built for NASA's Exoplanet Detection Challenge
            </div>
            <div>
              Data sources: Kepler, K2, TESS missions
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default ExoplanetDetectionApp;