import { Loader2 } from 'lucide-react';

/**
 * Skeleton component for loading states
 */
export const Skeleton = ({ className = '' }: { className?: string }) => (
    <div className={`skeleton ${className}`}>
        <style>{`
            .skeleton {
                background: linear-gradient(
                    90deg,
                    rgba(148, 163, 184, 0.1) 25%,
                    rgba(148, 163, 184, 0.2) 50%,
                    rgba(148, 163, 184, 0.1) 75%
                );
                background-size: 200% 100%;
                animation: skeleton-loading 1.5s ease-in-out infinite;
                border-radius: 4px;
            }

            @keyframes skeleton-loading {
                0% {
                    background-position: 200% 0;
                }
                100% {
                    background-position: -200% 0;
                }
            }
        `}</style>
    </div>
);

/**
 * Card skeleton for content loading
 */
export const CardSkeleton = () => (
    <div className="card-skeleton">
        <Skeleton className="card-skeleton__header" />
        <Skeleton className="card-skeleton__content" />
        <Skeleton className="card-skeleton__content card-skeleton__content--short" />
        <style>{`
            .card-skeleton {
                background: rgba(30, 41, 59, 0.6);
                border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 12px;
                padding: 1.5rem;
            }

            .card-skeleton__header {
                height: 24px;
                width: 60%;
                margin-bottom: 1rem;
            }

            .card-skeleton__content {
                height: 16px;
                width: 100%;
                margin-bottom: 0.5rem;
            }

            .card-skeleton__content--short {
                width: 40%;
            }
        `}</style>
    </div>
);

/**
 * Stats card skeleton
 */
export const StatsCardSkeleton = () => (
    <div className="stats-card-skeleton">
        <Skeleton className="stats-card-skeleton__value" />
        <Skeleton className="stats-card-skeleton__label" />
        <style>{`
            .stats-card-skeleton {
                background: rgba(30, 41, 59, 0.6);
                border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 12px;
                padding: 1.5rem;
                text-align: center;
            }

            .stats-card-skeleton__value {
                height: 40px;
                width: 80%;
                margin: 0 auto 0.75rem;
            }

            .stats-card-skeleton__label {
                height: 16px;
                width: 60%;
                margin: 0 auto;
            }
        `}</style>
    </div>
);

/**
 * Form skeleton
 */
export const FormSkeleton = () => (
    <div className="form-skeleton">
        <Skeleton className="form-skeleton__title" />
        <div className="form-skeleton__grid">
            <Skeleton className="form-skeleton__input" />
            <Skeleton className="form-skeleton__input" />
            <Skeleton className="form-skeleton__input" />
            <Skeleton className="form-skeleton__input" />
        </div>
        <Skeleton className="form-skeleton__button" />
        <style>{`
            .form-skeleton {
                background: rgba(30, 41, 59, 0.6);
                border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 12px;
                padding: 2rem;
            }

            .form-skeleton__title {
                height: 32px;
                width: 50%;
                margin-bottom: 2rem;
            }

            .form-skeleton__grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 1rem;
                margin-bottom: 2rem;
            }

            .form-skeleton__input {
                height: 48px;
                width: 100%;
            }

            .form-skeleton__button {
                height: 48px;
                width: 200px;
                margin: 0 auto;
            }

            @media (max-width: 640px) {
                .form-skeleton__grid {
                    grid-template-columns: 1fr;
                }
            }
        `}</style>
    </div>
);

/**
 * Full page loading spinner
 */
export const FullPageLoader = ({ message = 'Loading...' }: { message?: string }) => (
    <div className="full-page-loader">
        <div className="full-page-loader__content">
            <Loader2 className="full-page-loader__spinner" size={48} />
            <p className="full-page-loader__message">{message}</p>
        </div>
        <style>{`
            .full-page-loader {
                position: fixed;
                inset: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(15, 23, 42, 0.8);
                backdrop-filter: blur(4px);
                z-index: 50;
            }

            .full-page-loader__content {
                text-align: center;
            }

            .full-page-loader__spinner {
                color: #3b82f6;
                animation: spin 1s linear infinite;
                margin-bottom: 1rem;
            }

            @keyframes spin {
                from {
                    transform: rotate(0deg);
                }
                to {
                    transform: rotate(360deg);
                }
            }

            .full-page-loader__message {
                color: #e2e8f0;
                font-size: 1.125rem;
                margin: 0;
            }
        `}</style>
    </div>
);

/**
 * Inline loading spinner
 */
export const InlineLoader = ({ size = 20 }: { size?: number }) => (
    <span className="inline-loader">
        <Loader2 size={size} />
        <style>{`
            .inline-loader {
                display: inline-flex;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                from {
                    transform: rotate(0deg);
                }
                to {
                    transform: rotate(360deg);
                }
            }
        `}</style>
    </span>
);

export default Skeleton;
