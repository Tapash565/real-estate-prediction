import { useState, useEffect } from 'react'
import { Home, TrendingUp, Search, Info, Loader2, AlertCircle } from 'lucide-react'
import SearchableDropdown from './components/SearchableDropdown'
import { FormProvider, FormInput, FormValidationSummary, ValidationError } from './components/FormValidation'
import { Skeleton, CardSkeleton, StatsCardSkeleton, FullPageLoader } from './components/Skeleton'

interface Prediction {
    predicted_price: number;
    currency: string;
}

interface ModelInfo {
    model_type: string;
    metrics: {
        'R^2 Score'?: number;
        'RMSE'?: number;
        'MAE'?: number;
    };
}

interface Categories {
    [key: string]: string[];
}

function App() {
    const [loading, setLoading] = useState(false)
    const [prediction, setPrediction] = useState<Prediction | null>(null)
    const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
    const [categories, setCategories] = useState<Categories | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [dataLoading, setDataLoading] = useState(true)

    const [formData, setFormData] = useState({
        bed: 3,
        bath: 2,
        house_size: 2000,
        acre_lot: 0.25,
        city: 'Manchester',
        state: 'New Hampshire',
        zip_code: '03101',
        brokered_by: 'Keller Williams Realty',
        status: 'for_sale'
    })

    const [validationErrors, setValidationErrors] = useState<ValidationError[]>([])

    useEffect(() => {
        const fetchData = async () => {
            setDataLoading(true)
            setError(null)

            try {
                // Fetch model info
                const modelRes = await fetch('/api/v1/model/info')
                if (!modelRes.ok) throw new Error('Failed to fetch model info')
                const modelData = await modelRes.json()
                setModelInfo(modelData)

                // Fetch categories
                const catRes = await fetch('/api/v1/model/categories')
                if (!catRes.ok) throw new Error('Failed to fetch categories')
                const catData = await catRes.json()
                setCategories(catData)
            } catch (err) {
                console.error("Error fetching initial data:", err)
                setError(err instanceof Error ? err.message : 'An error occurred')
            } finally {
                setDataLoading(false)
            }
        }

        fetchData()
    }, [])

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target
        setFormData((prev: typeof formData) => ({
            ...prev,
            [name]: ['bed', 'bath', 'house_size', 'acre_lot'].includes(name) ? parseFloat(value) : value
        }))
    }

    const handleCategoryChange = (name: string, value: string) => {
        setFormData((prev: typeof formData) => ({
            ...prev,
            [name]: value
        }))
    }

    const validateForm = (): boolean => {
        const errors: ValidationError[] = []

        if (!formData.city || formData.city.length < 1) {
            errors.push({ field: 'city', message: 'City is required', type: 'required' })
        }
        if (!formData.state || formData.state.length < 2) {
            errors.push({ field: 'state', message: 'State is required', type: 'required' })
        }
        if (!formData.zip_code || formData.zip_code.length < 1) {
            errors.push({ field: 'zip_code', message: 'ZIP code is required', type: 'required' })
        }
        if (formData.house_size <= 0) {
            errors.push({ field: 'house_size', message: 'House size must be positive', type: 'custom' })
        }
        if (formData.acre_lot < 0) {
            errors.push({ field: 'acre_lot', message: 'Acre lot cannot be negative', type: 'custom' })
        }
        if (formData.bed < 0) {
            errors.push({ field: 'bed', message: 'Bedrooms cannot be negative', type: 'custom' })
        }
        if (formData.bath < 0) {
            errors.push({ field: 'bath', message: 'Bathrooms cannot be negative', type: 'custom' })
        }

        setValidationErrors(errors)
        return errors.length === 0
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()

        if (!validateForm()) {
            return
        }

        setLoading(true)
        setError(null)

        try {
            const response = await fetch('/api/v1/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer test-token'
                },
                body: JSON.stringify(formData)
            })

            if (!response.ok) {
                const errorData = await response.json()
                throw new Error(errorData.detail || errorData.error || 'Prediction failed')
            }

            const data = await response.json()
            setPrediction(data)
        } catch (err) {
            console.error("Prediction failed:", err)
            setError(err instanceof Error ? err.message : 'An error occurred')
        } finally {
            setLoading(false)
        }
    }

    if (dataLoading) {
        return <FullPageLoader message="Loading application..." />
    }

    return (
        <div className="container">
            {error && (
                <div className="error-banner">
                    <AlertCircle size={20} />
                    <span>{error}</span>
                    <button onClick={() => setError(null)} className="error-banner__close">×</button>
                </div>
            )}

            <section className="hero">
                <div className="gradient-bg"></div>
                <h1>Predict Property Values</h1>
                <p>AI-powered real estate valuation using advanced machine learning models trained on millions of data points.</p>

                <div className="form-container glass">
                    {validationErrors.length > 0 && (
                        <FormValidationSummary errors={validationErrors} />
                    )}
                    <form onSubmit={handleSubmit}>
                        <div className="grid">
                            <div className="input-group slider-group">
                                <div className="slider-header">
                                    <label>Bedrooms</label>
                                    <span className="slider-value">{formData.bed}</span>
                                </div>
                                <input
                                    type="range"
                                    name="bed"
                                    min="1"
                                    max="10"
                                    value={formData.bed}
                                    onChange={handleInputChange}
                                />
                                <input type="number" name="bed" value={formData.bed} onChange={handleInputChange} step="1" required />
                            </div>

                            <div className="input-group slider-group">
                                <div className="slider-header">
                                    <label>Bathrooms</label>
                                    <span className="slider-value">{formData.bath}</span>
                                </div>
                                <input
                                    type="range"
                                    name="bath"
                                    min="1"
                                    max="10"
                                    value={formData.bath}
                                    onChange={handleInputChange}
                                    step="0.5"
                                />
                                <input type="number" name="bath" value={formData.bath} onChange={handleInputChange} step="0.5" required />
                            </div>

                            <div className="input-group">
                                <label>House Size (sqft)</label>
                                <input type="number" name="house_size" value={formData.house_size} onChange={handleInputChange} required />
                            </div>
                            <div className="input-group">
                                <label>Acre Lot</label>
                                <input type="number" name="acre_lot" value={formData.acre_lot} onChange={handleInputChange} step="0.01" required />
                            </div>
                        </div>

                        <div className="grid">
                            <SearchableDropdown
                                label="City"
                                options={categories?.city || []}
                                value={formData.city}
                                onChange={(val) => handleCategoryChange('city', val)}
                                placeholder="Select City"
                            />

                            <SearchableDropdown
                                label="State"
                                options={categories?.state || []}
                                value={formData.state}
                                onChange={(val) => handleCategoryChange('state', val)}
                                placeholder="Select State"
                            />

                            <div className="input-group">
                                <label>ZIP Code</label>
                                <input type="text" name="zip_code" value={formData.zip_code} onChange={handleInputChange} required />
                            </div>

                            <SearchableDropdown
                                label="Status"
                                options={categories?.status || []}
                                value={formData.status}
                                onChange={(val) => handleCategoryChange('status', val)}
                                placeholder="Select Status"
                            />
                        </div>

                        <button type="submit" className="submit-btn" disabled={loading}>
                            {loading ? <Loader2 className="animate-spin" /> : <Search size={20} />}
                            {loading ? 'Analyzing Market...' : 'Calculate Prediction'}
                        </button>
                    </form>

                    {prediction && (
                        <div className="result-card glass">
                            <h3>Estimated Market Value</h3>
                            <div className="price">
                                ${prediction.predicted_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                            </div>
                            <div className="stat-label">Currency: {prediction.currency}</div>
                        </div>
                    )}
                </div>
            </section>

            <section className="stats">
                <div className="stats-grid">
                    {dataLoading ? (
                        <>
                            <StatsCardSkeleton />
                            <StatsCardSkeleton />
                            <StatsCardSkeleton />
                        </>
                    ) : modelInfo ? (
                        <>
                            <div className="stat-item glass">
                                <div className="stat-value">{modelInfo.model_type}</div>
                                <div className="stat-label">Model Architecture</div>
                            </div>
                            <div className="stat-item glass">
                                <div className="stat-value">{(modelInfo.metrics?.['R^2 Score'] || 0).toFixed(4)}</div>
                                <div className="stat-label">Accuracy (R² Score)</div>
                            </div>
                            <div className="stat-item glass">
                                <div className="stat-value">
                                    ${(modelInfo.metrics?.['MAE'] || 0).toLocaleString()}
                                </div>
                                <div className="stat-label">Mean Absolute Error</div>
                            </div>
                        </>
                    ) : (
                        <div className="stats-placeholder">Model information unavailable</div>
                    )}
                </div>
            </section>

            <footer style={{ textAlign: 'center', padding: '4rem 0', color: '#64748b', fontSize: '0.875rem' }}>
                <p>© 2026 Real Estate Price Intelligence. Built with FastAPI & React.</p>
            </footer>
        </div>
    )
}

export default App
