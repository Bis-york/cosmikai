import React, { useEffect, useMemo, useState } from 'react';
import { Upload, FileText, BarChart3, Settings, Zap, TrendingUp, AlertCircle, CheckCircle, XCircle, Eye, Download } from 'lucide-react';
import OrbitVisualizer from './OrbitVisualizer';
import logoBlack from './assets/logo-transp.png';

const FileUploadArea = ({ 
  inputMode, 
  handleModeSwitch, 
  uploadedFile, 
  setUploadedFile, 
  setDetectionResults,
  starName,
  setStarName,
  mission,
  setMission,
  customMission,
  setCustomMission,
  showAdvanced,
  setShowAdvanced,
  nBins,
  setNBins,
  confidenceThreshold,
  setConfidenceThreshold,
  runDetection,
  isProcessing,
  errorMessage
}) => (
  <div className="bg-white rounded-lg shadow-md p-6">
      {/* Toggle between Upload and Query modes */}
      <div className="flex gap-4 mb-6">
        <button
          onClick={() => handleModeSwitch('upload')}
          className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors ${
            inputMode === 'upload'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          <Upload className="inline-block h-4 w-4 mr-2" />
          Upload Data
        </button>
        <button
          onClick={() => handleModeSwitch('query')}
          className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors ${
            inputMode === 'query'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          <BarChart3 className="inline-block h-4 w-4 mr-2" />
          Query by Star
        </button>
      </div>

      {/* Upload Mode */}
      {inputMode === 'upload' && (
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
      )}

      {/* Query Mode */}
      {inputMode === 'query' && (
        <div className="border-2 border-blue-300 rounded-lg p-8">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Star Name or ID
              </label>
              <input
                type="text"
                placeholder="e.g., Kepler-10, KIC 11446443, TIC 307210830"
                value={starName}
                onChange={(e) => setStarName(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              <p className="mt-1 text-sm text-gray-500">
                Enter the star's name, KIC ID, TIC ID, or EPIC number
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Mission/Survey
              </label>
              <select
                value={mission}
                onChange={(e) => {
                  setMission(e.target.value);
                  if (e.target.value !== 'Other') {
                    setCustomMission('');
                  }
                }}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="Kepler">Kepler</option>
                <option value="K2">K2</option>
                <option value="TESS">TESS</option>
                <option value="Other">Other</option>
              </select>
            </div>

            {mission === 'Other' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Custom Mission Name
                </label>
                <input
                  type="text"
                  placeholder="e.g., CoRoT, JWST, etc."
                  value={customMission}
                  onChange={(e) => setCustomMission(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            )}

            {/* Advanced Options Toggle */}
            <div className="pt-4 border-t border-gray-200">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center text-sm text-blue-600 hover:text-blue-800"
              >
                <Settings className="h-4 w-4 mr-1" />
                {showAdvanced ? 'Hide' : 'Show'} Advanced Options
              </button>
            </div>

            {/* Advanced Options Panel */}
            {showAdvanced && (
              <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Number of Bins (nBins)
                  </label>
                  <input
                    type="number"
                    value={nBins}
                    onChange={(e) => setNBins(parseInt(e.target.value) || 2001)}
                    min="100"
                    max="10000"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    Number of bins for the BLS periodogram (default: 2001)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Confidence Threshold: {(confidenceThreshold * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>10% (More Results)</span>
                    <span>90% (Higher Precision)</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Action Button - shown for both modes when ready */}
      {((inputMode === 'upload' && uploadedFile) || (inputMode === 'query' && starName.trim())) && (
        <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          {inputMode === 'upload' && uploadedFile && (
            <div className="flex items-center mb-4">
              <FileText className="h-5 w-5 text-green-600 mr-2" />
              <span className="text-green-800 font-medium">{uploadedFile.name}</span>
              <span className="ml-auto text-green-600 text-sm">
                {(uploadedFile.size / 1024).toFixed(1)} KB
              </span>
            </div>
          )}
          
          {inputMode === 'query' && starName.trim() && (
            <div className="flex items-center mb-4">
              <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
              <span className="text-green-800 font-medium">
                {starName} ({mission})
              </span>
            </div>
          )}

          <button
            onClick={runDetection}
            disabled={isProcessing}
            className="w-full bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {isProcessing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                {inputMode === 'query' ? 'Fetching data from CosmiKai backend...' : 'Analyzing Light Curve...'}
              </>
            ) : (
              <>
                <Zap className="h-4 w-4 mr-2" />
                Detect Exoplanets
              </>
            )}
          </button>

          {errorMessage && (
            <div className="mt-4 flex items-start gap-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
              <AlertCircle className="h-4 w-4 mt-0.5" />
              <p>{errorMessage}</p>
            </div>
          )}
        </div>
      )}
    </div>

);

const ExoplanetDetectionApp = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [detectionResults, setDetectionResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressStep, setProgressStep] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [modelStats, setModelStats] = useState({
    accuracy_pct: null,
    total_predictions: 0,
    planets_found: 0,
    false_positives: 0,
  });
  const [historyEntries, setHistoryEntries] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState(null);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [statsRefreshKey, setStatsRefreshKey] = useState(0);

  // Mock detection function - replace with actual API call
  const API_BASE_URL = useMemo(
    () => import.meta.env.VITE_API_BASE_URL ?? 'https://api.flyingwaffle.ca',
    [],
  );
  const VISUAL_BASE_URL = useMemo(
    () => import.meta.env.VITE_VISUAL_BASE_URL ?? 'https://visuals.flyingwaffle.ca',
    [],
  );

const runDetection = async () => {
  setErrorMessage(null);
  setIsProcessing(true);
  setProgressStep('queued');
  setDetectionResults(null);

     if (inputMode === 'upload') {   try {
     if (!uploadedFile) throw new Error('Please choose a CSV file.');
     const text = await uploadedFile.text(); // read file in browser
     // You can add star/mission fields in the UI if needed; keeping it simple here.
     const payload = {
       pipeline: 'data_analyzer',
       config: {
         parameter_csv: text,    // <<< key that triggers data_analyzer
         mission: mission === 'Other' ? customMission.trim() : mission,
       },
       threshold: Number(confidenceThreshold),
     };

     setProgressStep('loading-lightcurve');
     const response = await fetch(`${API_BASE_URL}/predict`, {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify(payload),
     });
     if (!response.ok) {
       const message = await response.text();
       throw new Error(message || `Request failed with status ${response.status}`);
     }

     setProgressStep('waiting-response');
     const resultJson = await response.json();
     const entries = Object.entries(resultJson);
     if (!entries.length) throw new Error('Unexpected empty response from backend.');

     setProgressStep('parsing-result');
     const [target, details] = entries[0];   // same normalization as query path
     const detection = {
       target,
       mission: details.mission ?? (mission === 'Other' ? customMission.trim() : mission),
       confidence: Number(details.confidence ?? 0),
       threshold: Number(details.threshold ?? confidenceThreshold),
       hasCandidate: Boolean(details.has_candidate),
       periodDays: details.period_days ?? null,
       durationDays: details.duration_days ?? null,
       transitTime: details.transit_time ?? null,
       device: details.device ?? 'unknown',
       nbins: details.nbins ?? nBins,
       dataPoints: details.data_points ?? null,
              lightCurve: details.light_curve_points ?? [],
       raw: details,
     };
     setDetectionResults(detection);
     setProgressStep(null);
     setHistoryRefreshKey((k) => k + 1);
     setStatsRefreshKey((k) => k + 1);
   } catch (err) {
     console.error(err);
     setDetectionResults({ error: err.message || 'Upload failed' });
     setErrorMessage(err.message || 'Upload failed');
     setProgressStep('failed');
   } finally {
     setIsProcessing(false);
   }
   return;
 }

    const trimmedStar = starName.trim();
    if (!trimmedStar) {
      setErrorMessage('Please enter a star name or identifier.');
      setIsProcessing(false);
      return;
    }

    const missionSelection = mission === 'Other' ? customMission.trim() : mission;
    if (!missionSelection) {
      setErrorMessage('Please specify the mission/survey to query.');
      setIsProcessing(false);
      return;
    }

    const payload = {
      config: {
        target_name: trimmedStar,
        mission: missionSelection,
      },
    };

    if (nBins) {
      payload.config.nbins = nBins;
    }

    if (confidenceThreshold) {
      payload.threshold = Number(confidenceThreshold);
    }

    try {
      setProgressStep('loading-lightcurve');
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `Request failed with status ${response.status}`);
      }

      setProgressStep('waiting-response');
      const resultJson = await response.json();
      const entries = Object.entries(resultJson);
      if (!entries.length) {
        throw new Error('Unexpected empty response from backend.');
      }

      setProgressStep('parsing-result');
      const [target, details] = entries[0];
      const detection = {
        target,
        mission: details.mission ?? missionSelection,
        confidence: typeof details.confidence === 'number' ? details.confidence : 0,
        threshold: typeof details.threshold === 'number' ? details.threshold : confidenceThreshold,
        hasCandidate: Boolean(details.has_candidate),
        periodDays: details.period_days ?? null,
        durationDays: details.duration_days ?? null,
        transitTime: details.transit_time ?? null,
        device: details.device ?? 'unknown',
        nbins: details.nbins ?? nBins,
        dataPoints: details.data_points ?? null,
        lightCurve: details.light_curve_points ?? [],
        raw: details,
      };

      setDetectionResults(detection);
      setProgressStep(null);
      setHistoryRefreshKey((key) => key + 1);
      setStatsRefreshKey((key) => key + 1);
    } catch (error) {
      console.error('Prediction request failed:', error);
      setDetectionResults({ error: error.message || 'Unknown error occurred' });
      setErrorMessage(error.message || 'Failed to contact CosmiKai backend.');
      setProgressStep('failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const [inputMode, setInputMode] = useState('upload'); // 'upload' or 'query'
  const [starName, setStarName] = useState('');
  const [mission, setMission] = useState('Kepler');
  const [customMission, setCustomMission] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [nBins, setNBins] = useState(512);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [show3DView, setShow3DView] = useState(false);

  const handleModeSwitch = (mode) => {
    setInputMode(mode);
    setDetectionResults(null);
    setProgressStep(null);
    if (mode === 'upload') {
      setStarName('');
    } else {
      setUploadedFile(null);
    }
  };

  useEffect(() => {
    if (activeTab !== 'statistics') return;
    let cancelled = false;

    const fetchHistory = async () => {
      setHistoryLoading(true);
      setHistoryError(null);
      try {
        const res = await fetch(`${API_BASE_URL}/db/stars`);
        if (!res.ok) {
          throw new Error(await res.text());
        }
        const data = await res.json();
        const targets = Array.isArray(data.targets) ? data.targets : [];
        if (!targets.length) {
          if (!cancelled) setHistoryEntries([]);
          return;
        }

        const detailResults = await Promise.all(
          targets.map(async (target) => {
            try {
              const detailRes = await fetch(`${API_BASE_URL}/db/stars/${encodeURIComponent(target)}`);
              if (!detailRes.ok) {
                throw new Error(await detailRes.text());
              }
              const detailJson = await detailRes.json();
              const [displayTarget, detail] = Object.entries(detailJson)[0] ?? [target, {}];
              return {
                key: target,
                displayTarget,
                detail,
              };
            } catch (err) {
              console.error(`Failed to fetch history detail for ${target}:`, err);
              return null;
            }
          })
        );

        const cleaned = detailResults
          .filter(Boolean)
          .sort((a, b) => {
            const ta = a.detail.cached_timestamp ? new Date(a.detail.cached_timestamp).getTime() : 0;
            const tb = b.detail.cached_timestamp ? new Date(b.detail.cached_timestamp).getTime() : 0;
            return tb - ta;
          });

        if (!cancelled) {
          setHistoryEntries(cleaned);
        }
      } catch (err) {
        if (!cancelled) {
          setHistoryError(err.message || 'Failed to load detection history');
        }
      } finally {
        if (!cancelled) setHistoryLoading(false);
      }
    };

    fetchHistory();

    return () => {
      cancelled = true;
    };
  }, [activeTab, API_BASE_URL, historyRefreshKey]);

  useEffect(() => {
    if (activeTab !== 'statistics') return;
    let cancelled = false;

    const fetchStats = async () => {
      setStatsLoading(true);
      setStatsError(null);
      try {
        const res = await fetch(`${API_BASE_URL}/db/stats`);
        if (!res.ok) {
          throw new Error(await res.text());
        }
        const data = await res.json();
        if (!cancelled && data) {
          setModelStats({
            accuracy_pct: typeof data.accuracy_pct === 'number' ? data.accuracy_pct : 0,
            total_predictions: data.total_predictions ?? 0,
            planets_found: data.planets_found ?? 0,
            false_positives: data.false_positives ?? 0,
          });
        }
      } catch (error) {
        console.error('Failed to load stats:', error);
        if (!cancelled) setStatsError(error.message || 'Failed to load statistics');
      } finally {
        if (!cancelled) setStatsLoading(false);
      }
    };

    fetchStats();

    return () => {
      cancelled = true;
    };
  }, [activeTab, API_BASE_URL, statsRefreshKey]);

  // const FileUploadArea = () => (
    
  // );

const DetectionResults = () => {
    const showProgress = isProcessing || progressStep && progressStep !== 'completed' && !detectionResults;

    if (showProgress) {
      const steps = [
        { id: 'queued', label: 'Queued request' },
        { id: 'loading-lightcurve', label: 'Fetching light curve' },
        { id: 'waiting-response', label: 'Running inference' },
        { id: 'parsing-result', label: 'Preparing results' },
      ];

      const activeIndex = steps.findIndex((step) => step.id === progressStep);
      const progressPercent = activeIndex >= 0 ? ((activeIndex + 1) / steps.length) * 100 : 5;

      return (
        <div className="bg-green-50 border border-green-200 rounded-lg p-6">
          <div className="flex items-center mb-4">
            <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
            <span className="text-green-800 font-medium">
              {starName} ({mission})
            </span>
          </div>

          <div className="relative overflow-hidden rounded-lg h-12 bg-blue-200/40 backdrop-blur">
            <div
              className="absolute inset-y-0 left-0 bg-blue-500/60 transition-all duration-500"
              style={{ width: `${progressPercent}%` }}
            />
            <div className="relative z-10 flex items-center justify-center h-full text-white text-sm font-medium">
              {steps[activeIndex]?.label ?? 'Starting...'}
            </div>
          </div>

          <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs text-gray-600">
            {steps.map((step, index) => (
              <div
                key={step.id}
                className={`flex items-center gap-2 p-2 rounded border text-gray-600 ${
                  index <= activeIndex ? 'border-blue-300 bg-blue-50' : 'border-transparent'
                }`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    index < activeIndex
                      ? 'bg-blue-500'
                      : index === activeIndex
                        ? 'bg-blue-400 animate-pulse'
                        : 'bg-gray-300'
                  }`}
                />
                {step.label}
              </div>
            ))}
          </div>
        </div>
      );
    }

    if (!detectionResults) return null;

    if (detectionResults.error) {
      return (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-red-700 flex items-start gap-3">
          <AlertCircle className="h-6 w-6 mt-1" />
          <div>
            <h2 className="text-xl font-semibold mb-1">Prediction failed</h2>
            <p>{detectionResults.error}</p>
          </div>
        </div>
      );
    }

    return (
      <div className="space-y-6">
        {/* Main Result Card */}
        <div className={`rounded-lg shadow-md p-6 ${detectionResults.hasCandidate ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'} border`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              {detectionResults.hasCandidate ? (
                <CheckCircle className="h-8 w-8 text-green-600 mr-3" />
              ) : (
                <XCircle className="h-8 w-8 text-gray-600 mr-3" />
              )}
              <div>
                <h2 className="text-2xl font-bold text-gray-900">
                  {detectionResults.hasCandidate ? '🪐 Exoplanet Candidate Detected!' : 'No Confident Exoplanet Signal'}
                </h2>
                <p className="text-gray-600">
                  Confidence: {(detectionResults.confidence * 100).toFixed(1)}% (threshold {(detectionResults.threshold * 100).toFixed(0)}%)
                </p>
                <p className="text-sm text-gray-500">
                  Target: {detectionResults.target} · Mission: {detectionResults.mission}
                </p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">Processing Time</div>
              <div className="text-lg font-semibold">{(detectionResults.raw?.elapsed_seconds ?? 0).toFixed(2)}s</div>
            </div>
          </div>
        </div>

        {/* Detection Parameters */}
        {detectionResults.hasCandidate && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <BarChart3 className="h-5 w-5 mr-2 text-blue-600" />
              Detected Planet Parameters
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {detectionResults.periodDays ? detectionResults.periodDays.toFixed(2) : '—'}
                </div>
                <div className="text-sm text-gray-600">Orbital Period (days)</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {detectionResults.durationDays ? detectionResults.durationDays.toFixed(3) : '—'}
                </div>
                <div className="text-sm text-gray-600">Transit Duration (days)</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">
                  {detectionResults.transitTime ? detectionResults.transitTime.toFixed(3) : '—'}
                </div>
                <div className="text-sm text-gray-600">Transit Time (days)</div>
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
            <span>Data points: {detectionResults.dataPoints ?? detectionResults.lightCurve.length ?? 0}</span>
            <div className="flex gap-2">
              <button
                onClick={() => setShow3DView(true)}
                className="flex items-center text-white bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition-colors"
              >
                <Eye className="h-4 w-4 mr-1" />
                View in 3D
              </button>
              <button className="flex items-center text-blue-600 hover:text-blue-800">
                <Download className="h-4 w-4 mr-1" />
                Export Data
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const openVisualExplorer = (entry) => {
    const targetName = entry?.detail?.original_target || entry?.displayTarget || entry?.key;
    if (!targetName) return;
    const url = `${VISUAL_BASE_URL}?target=${encodeURIComponent(targetName)}`;
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  const formatTimestamp = (value) => {
    if (!value) return '—';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '—';
    return date.toLocaleString();
  };

  const ModelStatistics = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <BarChart3 className="h-5 w-5 mr-2 text-blue-600" />
          Model Performance
        </h3>
        {statsError && (
          <div className="mb-4 text-sm text-red-600">{statsError}</div>
        )}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {statsLoading
                ? '—'
                : modelStats.accuracy_pct != null
                ? `${modelStats.accuracy_pct.toFixed(1)}%`
                : '—'}
            </div>
            <div className="text-sm text-gray-600">Accuracy</div>
          </div>
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {statsLoading ? '—' : modelStats.total_predictions}
            </div>
            <div className="text-sm text-gray-600">Total Predictions</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">
              {statsLoading ? '—' : modelStats.planets_found}
            </div>
            <div className="text-sm text-gray-600">Planets Found</div>
          </div>
          <div className="text-center p-4 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-600">
              {statsLoading ? '—' : modelStats.false_positives}
            </div>
            <div className="text-sm text-gray-600">False Positives</div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Detection History</h3>
        {historyLoading && (
          <div className="py-6 text-sm text-gray-500">Loading cached detections…</div>
        )}
        {historyError && !historyLoading && (
          <div className="py-4 text-sm text-red-600">{historyError}</div>
        )}
        {!historyLoading && !historyError && (
          <div className="space-y-3">
            {historyEntries.length === 0 && (
              <div className="text-sm text-gray-500">No cached detections yet. Run a query to populate history.</div>
            )}
            {historyEntries.map((entry) => {
              const detail = entry.detail || {};
              const targetName = detail.original_target || entry.displayTarget || entry.key;
              const missionName = detail.mission || '—';
              const confidence = typeof detail.confidence === 'number' ? `${(detail.confidence * 100).toFixed(1)}%` : '—';
              const status = detail.has_candidate ? 'Candidate' : 'No Candidate';
              const statusColor = detail.has_candidate ? 'text-green-600' : 'text-gray-600';
              const updated = formatTimestamp(detail.cached_timestamp);

              return (
                <button
                  key={entry.key}
                  onClick={() => openVisualExplorer(entry)}
                  className="w-full flex items-center justify-between p-3 bg-gray-50 hover:bg-blue-50 transition rounded-lg text-left"
                >
                  <div className="flex items-center">
                    <FileText className="h-4 w-4 text-gray-400 mr-2" />
                    <div>
                      <div className="font-medium text-gray-900">{targetName}</div>
                      <div className="text-xs text-gray-500">Mission: {missionName}</div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4 text-sm">
                    <span className={statusColor}>{status}</span>
                    <span className="text-gray-500">{confidence}</span>
                    <span className="text-gray-400">{updated}</span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
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
              <img
                src={logoBlack}
                alt="Cosmik AI logo"
                className="h-10 w-10 mr-3 object-contain"
              />
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
            <FileUploadArea 
              inputMode={inputMode}
              handleModeSwitch={handleModeSwitch}
              uploadedFile={uploadedFile}
              setUploadedFile={setUploadedFile}
              setDetectionResults={setDetectionResults}
              starName={starName}
              setStarName={setStarName}
              mission={mission}
              setMission={setMission}
              customMission={customMission}
              setCustomMission={setCustomMission}
              showAdvanced={showAdvanced}
              setShowAdvanced={setShowAdvanced}
              nBins={nBins}
              setNBins={setNBins}
              confidenceThreshold={confidenceThreshold}
              setConfidenceThreshold={setConfidenceThreshold}
              runDetection={runDetection}
              isProcessing={isProcessing}
              errorMessage={errorMessage}
            />
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

      {/* 3D Visualization Overlay */}
      {show3DView && detectionResults?.hasCandidate && (
        <OrbitVisualizer
          starName={detectionResults.target || starName || 'Unknown Star'}
          detections={[
            {
              name: `${detectionResults.target || starName || 'Star'}-b`,
              confidence: detectionResults.confidence,
              period_days: detectionResults.periodDays ?? undefined,
              planet_radius_earth: detectionResults.raw?.size_vs_earth ?? undefined,
              transit_depth_ppm:
                detectionResults.raw?.transit_depth_fraction != null
                  ? detectionResults.raw.transit_depth_fraction * 1e6
                  : undefined,
              eq_temp_k: detectionResults.raw?.equilibrium_temperature_k ?? undefined,
              folded_lightcurve: Array.isArray(detectionResults.lightCurve)
                ? {
                    phase: detectionResults.lightCurve.map((point) => point[0]),
                    flux: detectionResults.lightCurve.map((point) => point[1]),
                  }
                : undefined,
            },
          ]}
          onClose={() => setShow3DView(false)}
        />
      )}
    </div>
  );
};

export default ExoplanetDetectionApp;
