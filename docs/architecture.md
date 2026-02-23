# Architecture Diagram

## System Overview

```mermaid
graph TB
    subgraph Clients
        Web[Web Browser]
        API_Client[API Clients]
        Mobile[Mobile App]
    end

    subgraph Frontend["Frontend Layer"]
        React[React + TypeScript]
        Vite[Vite Build]
        EB[Error Boundaries]
        FV[Form Validation]
        LS[Loading States]
    end

    subgraph APILayer["API Layer (FastAPI)"]
        FastAPI[FastAPI Application]

        subgraph Middleware
            RL[Rate Limiting]
            AUTH[Authentication]
            CORS[CORS]
            LOG[Request Logging]
            SEC[Security Headers]
        end

        subgraph Endpoints
            Health[Health Check]
            Predict[/predict]
            BatchPredict[/predict/batch]
            Explain[/predict/explain]
            ModelInfo[/model/*]
        end

        subgraph Monitoring
            Prom[Prometheus Metrics]
            Trace[OpenTelemetry Tracing]
            Sentry[Sentry Error Tracking]
        end
    end

    subgraph MLService["ML Services"]
        Model[Scikit-learn Model]
        Pipeline[Preprocessing Pipeline]
        Features[Feature Engineering]
        Cache[Prediction Cache]
    end

    subgraph Infrastructure["Infrastructure"]
        Redis[(Redis Cache)]
        MLFlow[(MLflow Tracking)]
        FileStorage[(Model Storage)]
    end

    subgraph DataLayer["Data Layer"]
        CSV[(CSV Dataset)]
        Logs[(Prediction Logs)]
        Monitor[(Monitoring DB)]
    end

    subgraph DevOps["DevOps"]
        Docker[Docker]
        CI[GitHub Actions]
        Test[pytest]
        Lint[Ruff + mypy]
    end

    %% Connections
    Web -->|HTTPS| React
    Mobile -->|HTTPS| FastAPI
    API_Client -->|HTTPS| FastAPI

    React -->|Fetch API| FastAPI

    FastAPI --> RL
    FastAPI --> AUTH
    FastAPI --> CORS
    FastAPI --> LOG
    FastAPI --> SEC

    RL --> Endpoints
    AUTH --> Endpoints

    Predict --> Model
    BatchPredict --> Model
    Explain --> Model

    Model --> Pipeline
    Model --> Features
    Model --> Cache

    Cache --> Redis
    Predict -->|Log| Logs
    FastAPI --> Prom
    FastAPI --> Trace
    FastAPI --> Sentry

    Model -->|Load/Save| FileStorage
    MLService -->|Track| MLFlow

    Pipeline --> CSV
    Monitor -->|Drift Detection| MLService

    CI --> Test
    CI --> Lint
    Docker --> FastAPI
```

## Request Flow

```mermaid
sequenceDiagram
    participant Client as Client
    participant LB as Load Balancer
    participant API as FastAPI
    participant Cache as Redis Cache
    participant Auth as Auth Service
    participant Model as ML Model
    participant Metrics as Prometheus
    participant Logger as Structured Logger

    Client->>LB: POST /api/v1/predict
    LB->>API: Forward Request

    API->>Logger: Log Request (request_id)
    API->>Auth: Validate JWT/API Key
    Auth-->>API: User + Scopes

    API->>API: Rate Limit Check

    API->>Cache: Check Cache
    alt Cache Hit
        Cache-->>API: Cached Prediction
        API->>Metrics: Track (cached)
    else Cache Miss
        API->>Model: Make Prediction
        Model->>API: Prediction Result
        API->>Cache: Store Result
        API->>Metrics: Track (success)
    end

    API->>Logger: Log Response
    API-->>LB: JSON Response
    LB-->>Client: 200 OK + Prediction
```

## Data Flow

```mermaid
flowchart LR
    subgraph Input["Input Data"]
        Raw[Raw CSV]
        UserInput[User Input]
    end

    subgraph Processing["Processing"]
        Preprocess[Preprocessing Pipeline]
        Transform[Feature Transform]
        Scale[Standardization]
    end

    subgraph Model["Model"]
        Train[Training]
        Eval[Evaluation]
        CV[Cross-Validation]
        Deploy[Deployment]
    end

    subgraph Output["Output"]
        Prediction[Prediction]
        Explanation[Feature Importance]
        Cache[Cached Result]
    end

    Raw --> Preprocess
    Preprocess --> Transform
    Transform --> Scale
    Scale --> Train
    Train --> Eval
    Eval --> CV
    CV --> Deploy

    UserInput --> Preprocess
    Deploy --> Prediction
    Prediction --> Explanation
    Prediction --> Cache
```

## Component Architecture

```mermaid
graph TB
    subgraph API_Components["API Components"]
        direction TB
        Main[api/main.py]
        Schemas[api/schemas.py]
        Middleware[api/middleware.py]
        Auth[api/auth.py]
        Caching[api/caching.py]
        Monitoring[api/monitoring.py]
        Logging[api/logging_config.py]
    end

    subgraph ML_Components["ML Components"]
        direction TB
        Train[src/train.py]
        Pipeline[src/pipeline.py]
        Preprocess[src/preprocessing.py]
        Inference[src/inference.py]
        Eval[src/evaluate.py]
        Utils[src/model_utils.py]
    end

    subgraph Infrastructure_Components["Infrastructure Components"]
        direction TB
        MLFlow[src/mlflow_tracking.py]
        ABTest[src/ab_testing.py]
        Monitoring[src/model_monitoring.py]
        Retraining[src/retraining_pipeline.py]
    end

    subgraph Frontend_Components["Frontend Components"]
        direction TB
        App[App.tsx]
        ErrorBoundary[ErrorBoundary.tsx]
        Validation[FormValidation.tsx]
        Skeleton[Skeleton.tsx]
        Dropdown[SearchableDropdown.tsx]
    end

    Main --> Schemas
    Main --> Middleware
    Main --> Auth
    Main --> Caching
    Main --> Monitoring
    Main --> Logging

    Pipeline --> Train
    Pipeline --> Preprocess
    Inference --> Pipeline
    Inference --> Utils

    MLFlow --> Train
    ABTest --> Inference
    Monitoring --> Inference
    Retraining --> Train

    App --> ErrorBoundary
    App --> Validation
    App --> Skeleton
    App --> Dropdown
```

## Deployment Architecture

```mermaid
graph TB
    subgraph Production["Production Environment"]
        subgraph Docker_Network
            API_Container[API Container]
            Redis_Container[Redis Container]
            MLFlow_Container[MLflow Container]
        end

        subgraph External_Services
            Prometheus[(Prometheus)]
            Grafana[(Grafana)]
            Sentry[Sentry.io]
        end
    end

    subgraph CI_CD["CI/CD Pipeline"]
        Git[Git Push]
        GitHub[GitHub Actions]
        Test[Run Tests]
        Build[Build Image]
        Push[Push to Registry]
        Deploy[Deploy to Prod]
    end

    Git --> GitHub
    GitHub --> Test
    Test --> Build
    Build --> Push
    Push --> Deploy
    Deploy --> API_Container

    API_Container --> Redis_Container
    API_Container --> MLFlow_Container
    API_Container --> Prometheus
    API_Container --> Sentry
    Prometheus --> Grafana
```

## Security Layers

```mermaid
graph LR
    subgraph Security["Security Layers"]
        direction TB
        HTTPS[HTTPS/TLS]
        CORS[CORS Policy]
        Rate[Rate Limiting]
        Auth[Authentication]
        Input[Input Validation]
        Sanitization[Input Sanitization]
    end

    Client --> HTTPS
    HTTPS --> CORS
    CORS --> Rate
    Rate --> Auth
    Auth --> Input
    Input --> Sanitization
    Sanitization --> API

    API[Protected API]
```

## Monitoring Stack

```mermaid
graph TB
    subgraph Observability["Observability Stack"]
        API[FastAPI]

        subgraph Metrics["Metrics"]
            Prom_Client[Prometheus Client]
            Prom_Server[Prometheus Server]
            Grafana[Grafana Dashboards]
        end

        subgraph Tracing["Distributed Tracing"]
            OTel[OpenTelemetry]
            OTLP[OTLP Exporter]
            Jaeger[Jaeger/Tempo]
        end

        subgraph Logging["Logging"]
            StructLog[structlog]
            JSON_Log[JSON Logs]
            Loki[Loki/Grafana]
        end

        subgraph Errors["Error Tracking"]
            Sentry_SDK[Sentry SDK]
            Sentry[Sentry Platform]
        end
    end

    API --> Prom_Client
    Prom_Client --> Prom_Server
    Prom_Server --> Grafana

    API --> OTel
    OTel --> OTLP
    OTLP --> Jaeger

    API --> StructLog
    StructLog --> JSON_Log
    JSON_Log --> Loki

    API --> Sentry_SDK
    Sentry_SDK --> Sentry
```
