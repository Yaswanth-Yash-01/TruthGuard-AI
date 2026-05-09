import os
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

# Metric instruments — None until setup_telemetry() is called
request_counter = None
hallucination_counter = None
insufficient_counter = None
human_review_counter = None
pipeline_duration = None


def setup_telemetry(service_name: str = "truthguard") -> None:
    global request_counter, hallucination_counter, insufficient_counter
    global human_review_counter, pipeline_duration

    resource = Resource.create({SERVICE_NAME: service_name})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{OTEL_ENDPOINT}/v1/traces"))
    )
    trace.set_tracer_provider(tracer_provider)

    # Jaeger is traces-only — use no-op meter so call-sites don't need guards
    meter = metrics.get_meter("truthguard")
    request_counter = meter.create_counter(
        "truthguard.requests.total",
        description="Total questions received",
    )
    hallucination_counter = meter.create_counter(
        "truthguard.hallucinations.total",
        description="Answers flagged as hallucinated",
    )
    insufficient_counter = meter.create_counter(
        "truthguard.insufficient_info.total",
        description="Insufficient info responses",
    )
    human_review_counter = meter.create_counter(
        "truthguard.human_reviews.resolved",
        description="Human reviews resolved",
    )
    pipeline_duration = meter.create_histogram(
        "truthguard.pipeline.duration_ms",
        unit="ms",
        description="End-to-end pipeline duration",
    )


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("truthguard")
