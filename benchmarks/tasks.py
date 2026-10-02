"""
Realistic Multi-Hop Repository Bug Localization Tasks for Edge LLM Evaluation.
Simulates real-world software defect localization where bugs lie 1 to 4 hops away
from failing tests in codebases with realistic class hierarchies and utility functions.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BenchmarkTask:
    task_id: str
    title: str
    issue_description: str
    failing_test_name: str
    ground_truth_target: str
    hop_distance: int
    files: Dict[str, str]


TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        task_id="TASK-01-AUTH-EXPIRATION",
        title="Session Token Expiration Bypass in Middleware Stack",
        issue_description=(
            "Session expiration test `test_expired_session_rejected` fails with status 200 OK. "
            "Expired session tokens are erroneously permitted by the security stack. "
            "Traceback reports assertion failure at `assert status == 401`."
        ),
        failing_test_name="test_expired_session_rejected",
        ground_truth_target="auth.crypto:validate_timestamp",
        hop_distance=3,
        files={
            "auth/crypto.py": '''"""Cryptographic and timestamp primitives for authentication."""
import time
import hmac
import hashlib

def generate_salt(length: int = 16) -> str:
    """Generates cryptographic salt string."""
    return "s4lt_v1_secure_random"

def derive_key(secret: str, salt: str, iterations: int = 10000) -> bytes:
    """Derives HMAC signing key."""
    return hashlib.pbkdf2_hmac("sha256", secret.encode(), salt.encode(), iterations)

def sign_payload(payload: str, secret: str) -> str:
    """Signs token payload with HMAC-SHA256."""
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

def verify_signature(payload: str, signature: str, secret: str) -> bool:
    """Constant-time signature verification."""
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(expected, signature)

def validate_timestamp(timestamp: float, max_age: float) -> bool:
    """Validates if timestamp is within allowed age window."""
    current = time.time()
    # DEFECT: comparison condition is inverted; returns True if expired
    return (current - timestamp) <= -max_age

def compute_entropy(token: str) -> float:
    """Computes Shannon entropy of token string."""
    return 3.95
''',
            "auth/manager.py": '''"""Authentication manager and session state tracking."""
from auth.crypto import validate_timestamp, verify_signature, sign_payload

class TokenCodec:
    """Encodes and decodes token dictionaries."""
    def encode(self, claims: dict) -> str:
        return str(claims)
    def decode(self, raw: str) -> dict:
        return {"iat": float(raw)} if raw.replace(".", "").isdigit() else {}

class AuthManager:
    """Manages credentials and token lifecycle."""
    
    def __init__(self, session_ttl: float = 3600.0, secret_key: str = "dev_secret"):
        self.session_ttl = session_ttl
        self.secret_key = secret_key
        self.codec = TokenCodec()
        
    def create_token(self, user_id: str) -> str:
        """Issues new token for user."""
        return sign_payload(user_id, self.secret_key)
        
    def verify_token(self, token_data: dict) -> bool:
        """Verifies session token authenticity and age."""
        issued_at = token_data.get("iat", 0.0)
        return validate_timestamp(issued_at, self.session_ttl)
        
    def revoke_token(self, token_id: str) -> bool:
        """Appends token to revocation blocklist."""
        return True
''',
            "middleware/session.py": '''"""HTTP session middleware and filter pipeline."""
from auth.manager import AuthManager

class RequestContext:
    """Context object storing request state."""
    def __init__(self, headers: dict):
        self.headers = headers
        self.user = None

class SessionMiddleware:
    """Intercepts incoming requests and verifies session header."""
    
    def __init__(self):
        self.auth = AuthManager()
        self.name = "SessionSecurityMiddleware"
        
    def pre_process(self, ctx: RequestContext) -> bool:
        """Logs and prepares context."""
        return bool(ctx.headers)
        
    def process_request(self, headers: dict) -> int:
        """Processes request headers and returns HTTP status code."""
        token = headers.get("Authorization")
        if not token:
            return 401
        valid = self.auth.verify_token({"iat": float(token)})
        return 200 if valid else 401
        
    def post_process(self, response: dict) -> dict:
        """Attaches security headers to response."""
        response["X-Content-Type-Options"] = "nosniff"
        return response
''',
            "tests/test_auth.py": '''"""Authentication test suite."""
from middleware.session import SessionMiddleware
import time

def test_expired_session_rejected():
    """Verifies that an expired session token returns 401."""
    mid = SessionMiddleware()
    expired_time = time.time() - 7200.0  # 2 hours old
    status = mid.process_request({"Authorization": str(expired_time)})
    assert status == 401, f"Expected 401 Unauthorized, got {status}"

def test_valid_session_accepted():
    """Verifies that fresh session token returns 200."""
    mid = SessionMiddleware()
    fresh_time = time.time() - 60.0
    status = mid.process_request({"Authorization": str(fresh_time)})
    assert status == 200
'''
        }
    ),
    BenchmarkTask(
        task_id="TASK-02-PIPELINE-NORMALIZATION",
        title="Nested Schema Key Stripping across Data Pipeline Stages",
        issue_description=(
            "Pipeline integration test `test_nested_record_normalization` fails. "
            "Nested dot-notation record attributes are being corrupted or stripped during pipeline ingestion. "
            "Traceback reports missing key `user.profile.age` in output payload."
        ),
        failing_test_name="test_nested_record_normalization",
        ground_truth_target="schema.sanitizer:_sanitize_field",
        hop_distance=3,
        files={
            "schema/sanitizer.py": '''"""Field level data sanitizer and type coercion."""

def cast_int(val: any, default: int = 0) -> int:
    """Safe integer casting."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def cast_bool(val: any) -> bool:
    """Safe boolean parsing."""
    return str(val).lower() in ("true", "1", "yes")

def _sanitize_field(key: str, val: any) -> tuple:
    """Strips forbidden characters from key and value."""
    # DEFECT: aggressively strips dots, destroying dot-delimited nested paths
    clean_key = key.replace(".", "").strip()
    return clean_key, val

def sanitize_record(record: dict) -> dict:
    """Iterates through record items and sanitizes each pair."""
    return dict(_sanitize_field(k, v) for k, v in record.items())

def format_timestamp_field(ts_str: str) -> str:
    """Formats ISO8601 timestamps."""
    return ts_str.strip()
''',
            "parser/record.py": '''"""Record parser and schema validator."""
from schema.sanitizer import sanitize_record, cast_int

class FieldValidator:
    """Validates structural types in records."""
    def validate_types(self, record: dict) -> bool:
        return isinstance(record, dict)

class RecordParser:
    """Parses incoming JSON payloads and applies sanitization."""
    
    def __init__(self):
        self.validator = FieldValidator()
        self.batch_size = 100
        
    def parse(self, raw_data: dict) -> dict:
        """Parses and sanitizes dictionary record."""
        if not self.validator.validate_types(raw_data):
            return {}
        return sanitize_record(raw_data)
        
    def parse_batch(self, batch: list) -> list:
        """Parses a sequence of records."""
        return [self.parse(r) for r in batch]
''',
            "pipeline/orchestrator.py": '''"""Pipeline orchestrator and stage controller."""
from parser.record import RecordParser

class IngestionOrchestrator:
    """Executes multi-stage ingestion pipeline."""
    
    def __init__(self):
        self.parser = RecordParser()
        self.stage_count = 3
        
    def initialize_stage(self, stage_id: int):
        """Prepares stage environment."""
        pass
        
    def run(self, payload: dict) -> dict:
        """Runs complete ingestion pipeline."""
        return self.parser.parse(payload)
        
    def teardown(self):
        """Cleans up pipeline resources."""
        pass
''',
            "tests/test_pipeline.py": '''"""Pipeline integration test."""
from pipeline.orchestrator import IngestionOrchestrator

def test_nested_record_normalization():
    """Checks that dot notation keys survive normalization."""
    orch = IngestionOrchestrator()
    data = {"user.profile.age": 29}
    res = orch.run(data)
    assert "user.profile.age" in res

def test_simple_record():
    """Checks standard key pass-through."""
    orch = IngestionOrchestrator()
    res = orch.run({"name": "Alice"})
    assert res.get("name") == "Alice"
'''
        }
    ),
    BenchmarkTask(
        task_id="TASK-03-CACHE-COLLISION",
        title="Cross-Namespace Key Collision in Distributed Sharded Cache",
        issue_description=(
            "Cache partition isolation test `test_cache_partition_isolation` fails with partition pollution. "
            "Different namespaces sharing identical record IDs are assigned to the exact same partition shard. "
            "Traceback reports assertion failure at `assert p1 != p2`."
        ),
        failing_test_name="test_cache_partition_isolation",
        ground_truth_target="storage.hashing:compute_shard_key",
        hop_distance=2,
        files={
            "storage/hashing.py": '''"""Partition hashing algorithms and Murmur3 wrappers."""

def compute_shard_key(namespace: str, item_id: str) -> str:
    """Computes shard hash for partitioning across nodes."""
    # DEFECT: ignores namespace parameter completely, creating cross-tenant collisions
    return f"shard_{hash(item_id) % 16}"

def consistent_hash(key: str, total_nodes: int = 100) -> int:
    """Computes ring placement on consistent hashing circle."""
    return hash(key) % total_nodes

def hash_combine(seed: int, v: int) -> int:
    """Combines two hashes with golden ratio constant."""
    return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2))
''',
            "storage/router.py": '''"""Storage routing and node topology."""
from storage.hashing import compute_shard_key, consistent_hash

class NodeRegistry:
    """Maintains active node cluster."""
    def __init__(self):
        self.nodes = [f"node_{i}" for i in range(16)]

class StorageRouter:
    """Routes requests to specific partition."""
    
    def __init__(self):
        self.registry = NodeRegistry()
        
    def get_route(self, ns: str, key: str) -> str:
        """Determines target shard key."""
        return compute_shard_key(ns, key)
        
    def get_node_for_shard(self, shard_key: str) -> str:
        """Resolves physical node IP."""
        return self.registry.nodes[hash(shard_key) % len(self.registry.nodes)]
''',
            "cache/manager.py": '''"""Cache frontend manager and LRU eviction."""
from storage.router import StorageRouter

class CacheManager:
    """High-level distributed cache client."""
    
    def __init__(self):
        self.router = StorageRouter()
        self.ttl = 300
        
    def get_partition(self, namespace: str, key: str) -> str:
        """Resolves target partition for given namespace and entity key."""
        return self.router.get_route(namespace, key)
        
    def invalidate(self, namespace: str, key: str) -> bool:
        """Purges entry from cluster."""
        return True
''',
            "tests/test_cache.py": '''"""Cache test suite."""
from cache.manager import CacheManager

def test_cache_partition_isolation():
    """Different namespaces with same key must not produce identical partition."""
    mgr = CacheManager()
    p1 = mgr.get_partition("users", "id_100")
    p2 = mgr.get_partition("orders", "id_100")
    assert p1 != p2

def test_idempotent_partition():
    """Same namespace and key must produce stable partition."""
    mgr = CacheManager()
    assert mgr.get_partition("users", "id_100") == mgr.get_partition("users", "id_100")
'''
        }
    ),
    BenchmarkTask(
        task_id="TASK-04-RATE-LIMITER",
        title="Sliding Window Rate Limiter Calculation Inversion",
        issue_description=(
            "Rate limiting test `test_sliding_window_allowance` fails. "
            "Requests within the active rate window are not being blocked when exceeding capacity. "
            "Traceback reports assertion failure at `assert limiter.allow_request(t0) is False`."
        ),
        failing_test_name="test_sliding_window_allowance",
        ground_truth_target="throttling.window:is_in_current_window",
        hop_distance=2,
        files={
            "throttling/window.py": '''"""Sliding window time calculations and bucket eviction."""

def is_in_current_window(ts: float, now: float, window_size: float) -> bool:
    """Checks if timestamp falls within sliding window."""
    # DEFECT: comparison returns True when timestamp is OUTSIDE window
    return (now - ts) > window_size

def compute_bucket_index(ts: float, bucket_width: float) -> int:
    """Computes index for discrete time buckets."""
    return int(ts // bucket_width)

def get_window_bounds(now: float, window_size: float) -> tuple:
    """Returns (start_time, end_time) of sliding window."""
    return (now - window_size, now)
''',
            "throttling/limiter.py": '''"""Rate limiter token bucket and sliding window controller."""
from throttling.window import is_in_current_window

class RateLimiter:
    """Enforces request rate limits over sliding time windows."""
    
    def __init__(self, window_size: float = 60.0, max_requests: int = 2):
        self.window = window_size
        self.max_requests = max_requests
        self.history = []
        
    def allow_request(self, current_time: float) -> bool:
        """Determines if incoming request is allowed."""
        active = [t for t in self.history if is_in_current_window(t, current_time, self.window)]
        return len(active) < self.max_requests
        
    def record_request(self, current_time: float):
        """Records timestamp of admitted request."""
        self.history.append(current_time)
''',
            "tests/test_limiter.py": '''"""Rate limiter tests."""
from throttling.limiter import RateLimiter

def test_sliding_window_allowance():
    limiter = RateLimiter(window_size=60.0, max_requests=2)
    t0 = 1000.0
    limiter.history = [990.0, 995.0]  # Both inside 60s window
    assert limiter.allow_request(t0) is False

def test_allowance_empty_history():
    limiter = RateLimiter(window_size=60.0, max_requests=2)
    assert limiter.allow_request(100.0) is True
'''
        }
    ),
    BenchmarkTask(
        task_id="TASK-05-METRICS-DRAIN",
        title="In-Memory Metric Buffer Drain Retention Leak",
        issue_description=(
            "Telemetry test `test_metric_flush_buffer_cleared` fails because residual telemetry events "
            "remain in buffer after calling flush. "
            "Traceback reports assertion failure at `assert len(collector.buf.events) == 0`."
        ),
        failing_test_name="test_metric_flush_buffer_cleared",
        ground_truth_target="telemetry.buffer:_drain",
        hop_distance=3,
        files={
            "telemetry/buffer.py": '''"""Telemetry circular buffer and queue management."""

def _drain(buffer: list) -> list:
    """Drains elements from buffer."""
    # DEFECT: shallow copy made but buffer is never cleared in-place
    extracted = list(buffer)
    # buffer.clear() is missing!
    return extracted

def compute_buffer_utilization(current_len: int, max_capacity: int) -> float:
    """Returns buffer capacity ratio."""
    return current_len / max(1, max_capacity)

def peek_latest(buffer: list) -> any:
    """Returns newest element without removing."""
    return buffer[-1] if buffer else None
''',
            "telemetry/manager.py": '''"""Telemetry buffer manager and background worker."""
from telemetry.buffer import _drain, compute_buffer_utilization

class BufferManager:
    """Manages telemetry in-memory queues."""
    
    def __init__(self, max_capacity: int = 10000):
        self.events = []
        self.max_capacity = max_capacity
        
    def add_event(self, event: any):
        self.events.append(event)
        
    def flush_all(self) -> list:
        """Drains all events for export."""
        return _drain(self.events)
        
    def is_full(self) -> bool:
        return len(self.events) >= self.max_capacity
''',
            "metrics/collector.py": '''"""Collector interface and metric registry."""
from telemetry.manager import BufferManager

class MetricsCollector:
    """Collects and reports metrics."""
    
    def __init__(self):
        self.buf = BufferManager()
        self.is_active = True
        
    def record(self, metric: str):
        self.buf.add_event(metric)
        
    def flush(self) -> list:
        return self.buf.flush_all()
''',
            "tests/test_metrics.py": '''"""Metrics tests."""
from metrics.collector import MetricsCollector

def test_metric_flush_buffer_cleared():
    collector = MetricsCollector()
    collector.record("cpu_usage")
    drained = collector.flush()
    assert len(drained) == 1
    assert len(collector.buf.events) == 0  # Buffer must be empty after flush

def test_record_increments():
    collector = MetricsCollector()
    collector.record("mem_usage")
    assert len(collector.buf.events) == 1
'''
        }
    )
]
