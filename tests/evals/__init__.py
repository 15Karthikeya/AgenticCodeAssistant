import os

# Must run before any agentic_code_assistant module is imported: it makes @observe record traces.
os.environ.setdefault("ACA_TRACING", "1")
