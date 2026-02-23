import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
    children: ReactNode;
    fallback?: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
    errorInfo: ErrorInfo | null;
}

/**
 * Error Boundary component to catch JavaScript errors anywhere in the child component tree.
 * Displays a fallback UI instead of crashing the whole app.
 */
export class ErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error, errorInfo: null };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('ErrorBoundary caught an error:', error, errorInfo);
        this.setState({ error, errorInfo });

        // In production, you might want to log to an error tracking service
        if (import.meta.env.PROD && (window as any).Sentry) {
            (window as any).Sentry.captureException(error);
        }
    }

    handleReload = () => {
        window.location.reload();
    };

    handleReset = () => {
        this.setState({ hasError: false, error: null, errorInfo: null });
    };

    render() {
        if (this.state.hasError) {
            if (this.props.fallback) {
                return this.props.fallback;
            }

            return (
                <div className="error-boundary">
                    <div className="error-boundary__content">
                        <div className="error-boundary__icon">
                            <AlertTriangle size={64} />
                        </div>
                        <h1 className="error-boundary__title">Something went wrong</h1>
                        <p className="error-boundary__description">
                            We apologize for the inconvenience. An unexpected error has occurred.
                        </p>

                        {import.meta.env.DEV && this.state.error && (
                            <div className="error-boundary__details">
                                <h3>Error Details:</h3>
                                <pre className="error-boundary__error">
                                    {this.state.error.toString()}
                                </pre>
                                {this.state.errorInfo && (
                                    <pre className="error-boundary__stack">
                                        {this.state.errorInfo.componentStack}
                                    </pre>
                                )}
                            </div>
                        )}

                        <div className="error-boundary__actions">
                            <button
                                className="error-boundary__button error-boundary__button--primary"
                                onClick={this.handleReload}
                            >
                                <RefreshCw size={18} />
                                Reload Page
                            </button>
                            <button
                                className="error-boundary__button"
                                onClick={this.handleReset}
                            >
                                Try Again
                            </button>
                        </div>
                    </div>

                    <style>{`
                        .error-boundary {
                            min-height: 100vh;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            padding: 2rem;
                            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                        }

                        .error-boundary__content {
                            background: rgba(30, 41, 59, 0.8);
                            border: 1px solid rgba(148, 163, 184, 0.2);
                            border-radius: 16px;
                            padding: 3rem;
                            max-width: 600px;
                            width: 100%;
                            text-align: center;
                            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                        }

                        .error-boundary__icon {
                            color: #f59e0b;
                            margin-bottom: 1.5rem;
                            display: flex;
                            justify-content: center;
                        }

                        .error-boundary__title {
                            color: #f8fafc;
                            font-size: 1.875rem;
                            font-weight: 700;
                            margin: 0 0 1rem 0;
                        }

                        .error-boundary__description {
                            color: #94a3b8;
                            margin-bottom: 1.5rem;
                            line-height: 1.6;
                        }

                        .error-boundary__details {
                            background: rgba(15, 23, 42, 0.8);
                            border-radius: 8px;
                            padding: 1rem;
                            margin: 1.5rem 0;
                            text-align: left;
                            overflow-x: auto;
                        }

                        .error-boundary__details h3 {
                            color: #f8fafc;
                            font-size: 0.875rem;
                            margin: 0 0 0.5rem 0;
                        }

                        .error-boundary__error,
                        .error-boundary__stack {
                            color: #f87171;
                            font-family: monospace;
                            font-size: 0.75rem;
                            white-space: pre-wrap;
                            word-break: break-word;
                            margin: 0;
                        }

                        .error-boundary__stack {
                            color: #94a3b8;
                            margin-top: 0.5rem;
                        }

                        .error-boundary__actions {
                            display: flex;
                            gap: 1rem;
                            justify-content: center;
                            flex-wrap: wrap;
                            margin-top: 1.5rem;
                        }

                        .error-boundary__button {
                            display: inline-flex;
                            align-items: center;
                            gap: 0.5rem;
                            padding: 0.75rem 1.5rem;
                            border-radius: 8px;
                            font-weight: 500;
                            cursor: pointer;
                            transition: all 0.2s;
                            background: transparent;
                            border: 1px solid rgba(148, 163, 184, 0.3);
                            color: #e2e8f0;
                        }

                        .error-boundary__button:hover {
                            background: rgba(148, 163, 184, 0.1);
                        }

                        .error-boundary__button--primary {
                            background: linear-gradient(135deg, #3b82f6, #2563eb);
                            border: none;
                            color: white;
                        }

                        .error-boundary__button--primary:hover {
                            background: linear-gradient(135deg, #2563eb, #1d4ed8);
                        }
                    `}</style>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
