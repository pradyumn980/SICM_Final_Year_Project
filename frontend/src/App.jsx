import { useEffect, useMemo, useState } from 'react'
import {
  Activity, AlertCircle, ArrowRight, CheckCircle2, ChevronDown, CircleHelp,
  ClipboardList, HeartPulse, Info, LoaderCircle, RotateCcw, ShieldCheck,
  Stethoscope, Waves,
} from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const fields = [
  {
    name: 'spo2_mean', label: 'Mean oxygen saturation', short: 'SpO₂ mean', unit: '%',
    helper: 'Average peripheral oxygen saturation in the observation window.', min: 0, max: 100, step: 0.1, placeholder: 'e.g. 96.5', icon: Waves,
  },
  {
    name: 'systolic_bp_max', label: 'Maximum systolic blood pressure', short: 'SBP maximum', unit: 'mmHg',
    helper: 'Highest recorded systolic blood pressure in the observation window.', min: 30, max: 300, step: 0.1, placeholder: 'e.g. 145', icon: Activity,
  },
  {
    name: 'systolic_bp_mean', label: 'Mean systolic blood pressure', short: 'SBP mean', unit: 'mmHg',
    helper: 'Average systolic blood pressure in the observation window.', min: 30, max: 300, step: 0.1, placeholder: 'e.g. 112', icon: Activity,
  },
  {
    name: 'rbc_min', label: 'Minimum red blood cell count', short: 'RBC minimum', unit: '10⁶/µL',
    helper: 'Lowest RBC count reported by the laboratory in the observation window.', min: 0.1, max: 10, step: 0.01, placeholder: 'e.g. 3.25', icon: Waves,
  },
  {
    name: 'anion_gap_max', label: 'Maximum anion gap', short: 'Anion gap maximum', unit: 'mEq/L',
    helper: 'Highest anion-gap measurement in the observation window.', min: 0, max: 60, step: 0.1, placeholder: 'e.g. 19', icon: Activity,
  },
  {
    name: 'diastolic_bp_mean', label: 'Mean diastolic blood pressure', short: 'DBP mean', unit: 'mmHg',
    helper: 'Average diastolic blood pressure in the observation window.', min: 10, max: 200, step: 0.1, placeholder: 'e.g. 66', icon: Activity,
  },
  {
    name: 'creatinine_min', label: 'Minimum creatinine', short: 'Creatinine minimum', unit: 'mg/dL',
    helper: 'Lowest serum creatinine measurement in the observation window.', min: 0, max: 30, step: 0.01, placeholder: 'e.g. 0.9', icon: Waves,
  },
]

const initialValues = Object.fromEntries(fields.map((f) => [f.name, '']))

function Metric({ label, value, suffix = '' }) {
  return <div className="metric"><span>{label}</span><strong>{value}{suffix}</strong></div>
}

function App() {
  const [values, setValues] = useState(initialValues)
  const [status, setStatus] = useState('checking')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  useEffect(() => {
    let alive = true
    fetch(`${API_URL}/health`)
      .then((r) => r.ok ? r.json() : Promise.reject(new Error('API unavailable')))
      .then((data) => { if (alive) setStatus(data.model_ready ? 'ready' : 'missing') })
      .catch(() => { if (alive) setStatus('offline') })
    return () => { alive = false }
  }, [])

  const completed = useMemo(() => fields.filter((f) => values[f.name] !== '').length, [values])

  function updateField(name, value) {
    setValues((prev) => ({ ...prev, [name]: value }))
    setError('')
    setResult(null)
  }

  function validate() {
    for (const field of fields) {
      const raw = values[field.name]
      if (raw === '' || !Number.isFinite(Number(raw))) return `${field.label} is required.`
      const n = Number(raw)
      if (n < field.min || n > field.max) return `${field.label} must be between ${field.min} and ${field.max} ${field.unit}.`
    }
    if (Number(values.systolic_bp_max) < Number(values.systolic_bp_mean)) {
      return 'Maximum systolic blood pressure should not be lower than its mean.'
    }
    return ''
  }

  async function submit(event) {
    event.preventDefault()
    const validation = validate()
    if (validation) { setError(validation); setResult(null); return }
    setBusy(true); setError(''); setResult(null)
    try {
      const payload = Object.fromEntries(fields.map((f) => [f.name, Number(values[f.name])]))
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(data.detail || `Prediction request failed (${response.status}).`)
      setResult(data)
    } catch (e) {
      setError(e.message || 'Could not contact the prediction service.')
    } finally { setBusy(false) }
  }

  function reset() { setValues(initialValues); setResult(null); setError('') }

  const isElevated = result?.prediction === 1
  const probability = result ? Math.round(result.risk_probability * 1000) / 10 : 0

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="SICM research home">
          <span className="brand-mark"><HeartPulse size={21} /></span>
          <span><b>SICM</b><small>RESEARCH TOOL</small></span>
        </a>
        <div className="topbar-right">
          <span className={`connection ${status}`}><i />{status === 'ready' ? 'Model connected' : status === 'missing' ? 'Model artifact missing' : status === 'offline' ? 'API offline' : 'Checking model'}</span>
          <span className="private-tag"><ShieldCheck size={14} /> Research prototype</span>
        </div>
      </header>

      <main id="top" className="layout">
        <section className="hero">
          <div className="eyebrow"><span className="eyebrow-line" /> ICU DECISION-SUPPORT RESEARCH</div>
          <h1>Sepsis-associated<br /><em>cardiac risk</em> assessment</h1>
          <p className="hero-copy">Enter the patient’s aggregated clinical measurements to generate a model-based estimate for the reduced-LVEF proxy outcome.</p>
          <div className="hero-meta">
            <span><Stethoscope size={15} /> 7 selected predictors</span>
            <span><Activity size={15} /> Nested-CV candidate model</span>
            <span><ShieldCheck size={15} /> Research use only</span>
          </div>
        </section>

        <section className="notice" role="note">
          <Info size={18} />
          <div><b>Use window-level summaries, not a single bedside reading.</b><span>The selected predictors are aggregates from the project’s 24-hour feature window: [outcome time − 36 hours, outcome time − 12 hours). This is an echo-relative research window, not automatically the first 12 hours after sepsis onset.</span></div>
        </section>

        <div className="content-grid">
          <section className="form-panel panel">
            <div className="panel-heading">
              <div><div className="section-kicker">PATIENT INPUT</div><h2>Clinical measurements</h2><p>Enter all seven values in the units shown.</p></div>
              <div className="completion"><b>{completed}/7</b><span>completed</span></div>
            </div>
            <div className="progress-track"><span style={{ width: `${completed / 7 * 100}%` }} /></div>

            <form onSubmit={submit} noValidate>
              <div className="fields-grid">
                {fields.map((field, index) => {
                  const Icon = field.icon
                  return <div className="field" key={field.name}>
                    <label htmlFor={field.name}><span className="field-number">0{index + 1}</span>{field.label}</label>
                    <div className="input-wrap">
                      <Icon size={17} className="input-icon" />
                      <input id={field.name} name={field.name} type="number" inputMode="decimal" min={field.min} max={field.max} step={field.step} placeholder={field.placeholder} value={values[field.name]} onChange={(e) => updateField(field.name, e.target.value)} required />
                      <span className="unit">{field.unit}</span>
                    </div>
                    <small>{field.helper}</small>
                  </div>
                })}
              </div>

              {error && <div className="error-box"><AlertCircle size={17} /><span>{error}</span></div>}

              <div className="form-actions">
                <button type="button" className="reset-btn" onClick={reset}><RotateCcw size={16} /> Reset values</button>
                <button className="submit-btn" type="submit" disabled={busy || status !== 'ready'}>
                  {busy ? <><LoaderCircle className="spin" size={17} /> Analysing</> : <>Generate research estimate <ArrowRight size={17} /></>}
                </button>
              </div>
              {status !== 'ready' && <p className="model-hint">{status === 'missing' ? 'The API is reachable, but the fitted model artifact has not been installed yet. Follow the Colab export instructions in the project README.' : status === 'offline' ? 'Start the backend API to enable model inference.' : 'Checking backend connection…'}</p>}
            </form>
          </section>

          <aside className="side-column">
            <section className={`result-panel panel ${result ? 'has-result' : ''}`} aria-live="polite">
              <div className="section-kicker">MODEL OUTPUT</div>
              <div className="result-title-row"><h2>Prediction result</h2><span className="result-symbol"><HeartPulse size={20} /></span></div>
              {!result ? <div className="empty-result">
                <div className="empty-icon"><ClipboardList size={28} /></div>
                <h3>Awaiting patient inputs</h3>
                <p>Complete the seven measurements and submit the form to request an estimate from the fitted model.</p>
                <div className="empty-line"><span /><span /><span /></div>
              </div> : <div className="prediction-output">
                <div className={`risk-badge ${isElevated ? 'elevated' : 'lower'}`}>{isElevated ? 'Above model threshold' : 'Below model threshold'}</div>
                <div className="probability"><strong>{probability.toFixed(1)}<small>%</small></strong><span>Model-estimated score</span></div>
                <div className="meter"><span style={{ width: `${Math.min(100, Math.max(0, probability))}%` }} /></div>
                <div className="result-description">{result.interpretation || 'This is a model classification for the project’s reduced-LVEF proxy endpoint, not a clinical diagnosis.'}</div>
                <div className="result-facts">
                  <Metric label="Decision threshold" value={`${(result.threshold * 100).toFixed(1)}%`} />
                  <Metric label="Model" value={result.model_name || '7-feature LR'} />
                </div>
                <div className="result-foot"><CheckCircle2 size={15} /> Response returned by prediction API</div>
              </div>}
            </section>

            <section className="model-card panel">
              <div className="section-kicker">CURRENT CANDIDATE</div>
              <h3>Seven-feature Logistic Regression</h3>
              <p>Compact candidate selected for the first interface because it uses seven fields and offers competitive nested-CV results with a comparatively interpretable model.</p>
              <div className="model-stats">
                <Metric label="Pooled OOF PR-AUC" value="0.4141" />
                <Metric label="Pooled OOF ROC-AUC" value="0.6744" />
                <Metric label="F1 at 0.50" value="0.4546" />
                <Metric label="Balanced accuracy" value="0.6226" />
              </div>
              <div className="candidate-note"><Info size={15} /> These are nested cross-validation metrics, not a guarantee of an individual patient’s outcome.</div>
            </section>
          </aside>
        </div>

        <section className="disclaimer">
          <div className="disclaimer-icon"><CircleHelp size={18} /></div>
          <div><b>Research prototype — not for clinical decision-making</b><p>The project target is LVEF below 50% on an eligible subsequent echocardiogram among septic ICU patients. It is a reduced-LVEF proxy and does not establish new dysfunction, reversibility, or a confirmed diagnosis of sepsis-induced cardiomyopathy. Do not use this output to diagnose, treat, triage, or withhold care. Any clinical use would require clinical review, rigorous external and prospective validation, calibration assessment, and appropriate regulatory oversight.</p></div>
        </section>
      </main>

      <footer><span>SEPSIS-ASSOCIATED CARDIAC RISK · ACADEMIC PROJECT</span><span>Model service: {API_URL}</span></footer>
    </div>
  )
}

export default App
