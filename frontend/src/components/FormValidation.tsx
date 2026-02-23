import { useState, ReactNode, createContext, useContext } from 'react';
import { AlertCircle, CheckCircle2 } from 'lucide-react';

/**
 * Validation error type
 */
export interface ValidationError {
    field: string;
    message: string;
    type: 'required' | 'min' | 'max' | 'pattern' | 'custom';
}

/**
 * Validation rules type
 */
export interface ValidationRules {
    required?: boolean | string;
    min?: { value: number; message?: string };
    max?: { value: number; message?: string };
    minLength?: { value: number; message?: string };
    maxLength?: { value: number; message?: string };
    pattern?: { value: RegExp; message: string };
    custom?: { validator: (value: any) => boolean; message: string };
}

/**
 * Form validation context
 */
interface FormContextType {
    errors: Record<string, ValidationError>;
    touched: Record<string, boolean>;
    setError: (field: string, error: ValidationError | null) => void;
    setTouched: (field: string, touched: boolean) => void;
    validateField: (field: string, value: any, rules: ValidationRules) => boolean;
}

const FormContext = createContext<FormContextType | null>(null);

/**
 * Form validation provider
 */
export const FormProvider = ({ children }: { children: ReactNode }) => {
    const [errors, setErrors] = useState<Record<string, ValidationError>>({});
    const [touched, setTouchedState] = useState<Record<string, boolean>>({});

    const setError = (field: string, error: ValidationError | null) => {
        setErrors(prev => {
            const newErrors = { ...prev };
            if (error) {
                newErrors[field] = error;
            } else {
                delete newErrors[field];
            }
            return newErrors;
        });
    };

    const setTouched = (field: string, touched: boolean) => {
        setTouchedState(prev => ({ ...prev, [field]: touched }));
    };

    const validateField = (field: string, value: any, rules: ValidationRules): boolean => {
        // Required check
        if (rules.required && (value === undefined || value === null || value === '')) {
            const message = typeof rules.required === 'string' ? rules.required : 'This field is required';
            setError(field, { field, message, type: 'required' });
            return false;
        }

        if (value === undefined || value === null || value === '') {
            setError(field, null);
            return true;
        }

        // Min value check
        if (rules.min && typeof value === 'number' && value < rules.min.value) {
            setError(field, {
                field,
                message: rules.min.message || `Minimum value is ${rules.min.value}`,
                type: 'min',
            });
            return false;
        }

        // Max value check
        if (rules.max && typeof value === 'number' && value > rules.max.value) {
            setError(field, {
                field,
                message: rules.max.message || `Maximum value is ${rules.max.value}`,
                type: 'max',
            });
            return false;
        }

        // Min length check
        if (rules.minLength && String(value).length < rules.minLength.value) {
            setError(field, {
                field,
                message: rules.minLength.message || `Minimum length is ${rules.minLength.value}`,
                type: 'min',
            });
            return false;
        }

        // Max length check
        if (rules.maxLength && String(value).length > rules.maxLength.value) {
            setError(field, {
                field,
                message: rules.maxLength.message || `Maximum length is ${rules.maxLength.value}`,
                type: 'max',
            });
            return false;
        }

        // Pattern check
        if (rules.pattern && !rules.pattern.value.test(String(value))) {
            setError(field, {
                field,
                message: rules.pattern.message,
                type: 'pattern',
            });
            return false;
        }

        // Custom validator
        if (rules.custom && !rules.custom.validator(value)) {
            setError(field, {
                field,
                message: rules.custom.message,
                type: 'custom',
            });
            return false;
        }

        setError(field, null);
        return true;
    };

    return (
        <FormContext.Provider value={{ errors, touched, setError, setTouched, validateField }}>
            {children}
        </FormContext.Provider>
    );
};

/**
 * Hook to use form context
 */
export const useFormContext = () => {
    const context = useContext(FormContext);
    if (!context) {
        throw new Error('useFormContext must be used within a FormProvider');
    }
    return context;
};

/**
 * Form input with validation
 */
interface FormInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label: string;
    name: string;
    validation?: ValidationRules;
    helperText?: string;
}

export const FormInput = ({
    label,
    name,
    validation,
    helperText,
    className = '',
    onChange,
    onBlur,
    ...props
}: FormInputProps) => {
    const { errors, touched, setTouched, validateField } = useFormContext();
    const error = errors[name];
    const isTouched = touched[name];

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (validation) {
            validateField(name, e.target.value, validation);
        }
        onChange?.(e);
    };

    const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
        setTouched(name, true);
        if (validation) {
            validateField(name, e.target.value, validation);
        }
        onBlur?.(e);
    };

    return (
        <div className={`form-field ${className}`}>
            <label className="form-field__label" htmlFor={name}>
                {label}
                {validation?.required && <span className="form-field__required">*</span>}
            </label>
            <div className={`form-field__input-wrapper ${error && isTouched ? 'form-field--error' : ''}`}>
                <input
                    id={name}
                    name={name}
                    className="form-field__input"
                    onChange={handleChange}
                    onBlur={handleBlur}
                    aria-invalid={error && isTouched ? 'true' : 'false'}
                    aria-describedby={error && isTouched ? `${name}-error` : helperText ? `${name}-helper` : undefined}
                    {...props}
                />
                {error && isTouched && (
                    <span className="form-field__icon form-field__icon--error">
                        <AlertCircle size={20} />
                    </span>
                )}
                {!error && isTouched && (
                    <span className="form-field__icon form-field__icon--success">
                        <CheckCircle2 size={20} />
                    </span>
                )}
            </div>
            {error && isTouched && (
                <span id={`${name}-error`} className="form-field__error" role="alert">
                    {error.message}
                </span>
            )}
            {helperText && !error && (
                <span id={`${name}-helper`} className="form-field__helper">
                    {helperText}
                </span>
            )}

            <style>{`
                .form-field {
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }

                .form-field__label {
                    color: #e2e8f0;
                    font-weight: 500;
                    font-size: 0.875rem;
                }

                .form-field__required {
                    color: #f87171;
                    margin-left: 0.25rem;
                }

                .form-field__input-wrapper {
                    position: relative;
                    display: flex;
                    align-items: center;
                }

                .form-field--error .form-field__input {
                    border-color: #f87171;
                    background: rgba(248, 113, 113, 0.05);
                }

                .form-field--error .form-field__input:focus {
                    border-color: #f87171;
                    box-shadow: 0 0 0 3px rgba(248, 113, 113, 0.1);
                }

                .form-field__input {
                    width: 100%;
                    padding: 0.75rem 1rem;
                    padding-right: 2.5rem;
                    background: rgba(15, 23, 42, 0.6);
                    border: 1px solid rgba(148, 163, 184, 0.2);
                    border-radius: 8px;
                    color: #f8fafc;
                    font-size: 0.9375rem;
                    transition: all 0.2s;
                }

                .form-field__input:focus {
                    outline: none;
                    border-color: #3b82f6;
                    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
                }

                .form-field__input::placeholder {
                    color: #64748b;
                }

                .form-field__icon {
                    position: absolute;
                    right: 0.75rem;
                    display: flex;
                    align-items: center;
                    pointer-events: none;
                }

                .form-field__icon--error {
                    color: #f87171;
                }

                .form-field__icon--success {
                    color: #22c55e;
                }

                .form-field__error {
                    color: #f87171;
                    font-size: 0.875rem;
                    display: flex;
                    align-items: center;
                    gap: 0.375rem;
                }

                .form-field__helper {
                    color: #94a3b8;
                    font-size: 0.875rem;
                }
            `}</style>
        </div>
    );
};

/**
 * Form validation summary
 */
export const FormValidationSummary = ({ errors }: { errors: ValidationError[] }) => {
    if (errors.length === 0) return null;

    return (
        <div className="form-summary" role="alert">
            <div className="form-summary__header">
                <AlertCircle size={20} />
                <span>Please fix the following errors:</span>
            </div>
            <ul className="form-summary__list">
                {errors.map((error, index) => (
                    <li key={index} className="form-summary__item">
                        <strong>{error.field}:</strong> {error.message}
                    </li>
                ))}
            </ul>

            <style>{`
                .form-summary {
                    background: rgba(248, 113, 113, 0.1);
                    border: 1px solid rgba(248, 113, 113, 0.3);
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                }

                .form-summary__header {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    color: #f87171;
                    font-weight: 500;
                    margin-bottom: 0.5rem;
                }

                .form-summary__list {
                    margin: 0;
                    padding-left: 1.5rem;
                    color: #e2e8f0;
                }

                .form-summary__item {
                    margin-bottom: 0.25rem;
                }

                .form-summary__item strong {
                    color: #f87171;
                }
            `}</style>
        </div>
    );
};

export default FormProvider;
