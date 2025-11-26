"""
Performance and load tests for POLYMORPH-4 Lite.

This module tests:
- API response times
- Concurrent request handling
- Database query performance
- Memory usage patterns
- Workflow execution performance
"""
import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from fastapi import FastAPI

from retrofitkit.api.auth import router as auth_router
from retrofitkit.api.workflows import router as workflow_router


@pytest.fixture
def app():
    """Create test application."""
    app = FastAPI()
    app.include_router(auth_router, prefix="/auth")
    app.include_router(workflow_router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API endpoints."""

    def test_login_response_time(self, client):
        """Test that login responds within acceptable time."""
        # This test will fail initially but documents performance expectations
        start = time.time()

        response = client.post("/auth/login", json={
            "email": "test@polymorph.com",
            "password": "TestPassword123!"
        })

        duration = time.time() - start

        # Login should complete within 500ms
        assert duration < 0.5, f"Login took {duration:.3f}s, expected < 0.5s"

    def test_workflow_list_performance(self, client):
        """Test workflow listing performance."""
        # Add some workflows first
        for i in range(10):
            workflow_yaml = f"""
id: "perf_workflow_{i}"
name: "Performance Test Workflow {i}"
entry_step: "wait"
steps:
  wait:
    kind: "wait"
    params:
      seconds: 0.01
    children: []
"""
            client.post("/workflows/", json={"yaml_content": workflow_yaml})

        # Measure list performance
        start = time.time()
        response = client.get("/workflows/")
        duration = time.time() - start

        assert response.status_code == 200
        # Should list workflows quickly (< 100ms)
        assert duration < 0.1, f"Listing took {duration:.3f}s, expected < 0.1s"

    def test_concurrent_workflow_uploads(self, client):
        """Test handling concurrent workflow uploads."""
        def upload_workflow(index):
            workflow_yaml = f"""
id: "concurrent_workflow_{index}"
name: "Concurrent Workflow {index}"
entry_step: "wait"
steps:
  wait:
    kind: "wait"
    params:
      seconds: 0.01
    children: []
"""
            return client.post("/workflows/", json={"yaml_content": workflow_yaml})

        # Upload 20 workflows concurrently
        start = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(upload_workflow, i) for i in range(20)]
            results = [f.result() for f in futures]

        duration = time.time() - start

        # All should succeed
        assert all(r.status_code == 201 for r in results)

        # Should complete in reasonable time (< 5s)
        assert duration < 5.0, f"Concurrent uploads took {duration:.3f}s"


@pytest.mark.performance
class TestWorkflowExecutionPerformance:
    """Performance tests for workflow execution."""

    def test_simple_workflow_execution_time(self, client):
        """Test execution time for simple workflow."""
        workflow_yaml = """
id: "simple_perf_workflow"
name: "Simple Performance Workflow"
entry_step: "wait"
steps:
  wait:
    kind: "wait"
    params:
      seconds: 0.1
    children: []
"""
        client.post("/workflows/", json={"yaml_content": workflow_yaml})

        start = time.time()
        response = client.post("/workflows/simple_perf_workflow/execute", json={})
        duration = time.time() - start

        assert response.status_code == 200
        assert response.json()["success"] is True

        # Should complete close to wait time + small overhead (< 0.2s total)
        assert duration < 0.2, f"Execution took {duration:.3f}s, expected < 0.2s"

    def test_multi_step_workflow_performance(self, client):
        """Test performance of workflow with multiple steps."""
        workflow_yaml = """
id: "multi_step_perf"
name: "Multi-Step Performance Test"
entry_step: "step1"
steps:
  step1:
    kind: "wait"
    params:
      seconds: 0.01
    children: ["step2"]
  step2:
    kind: "wait"
    params:
      seconds: 0.01
    children: ["step3"]
  step3:
    kind: "wait"
    params:
      seconds: 0.01
    children: ["step4"]
  step4:
    kind: "wait"
    params:
      seconds: 0.01
    children: ["step5"]
  step5:
    kind: "wait"
    params:
      seconds: 0.01
    children: []
"""
        client.post("/workflows/", json={"yaml_content": workflow_yaml})

        start = time.time()
        response = client.post("/workflows/multi_step_perf/execute", json={})
        duration = time.time() - start

        assert response.status_code == 200
        result = response.json()

        assert result["success"] is True
        assert len(result["steps_executed"]) == 5

        # 5 steps * 0.01s + overhead should be < 0.2s
        assert duration < 0.2, f"Multi-step execution took {duration:.3f}s"

    def test_workflow_execution_throughput(self, client):
        """Test workflow execution throughput."""
        workflow_yaml = """
id: "throughput_test"
name: "Throughput Test Workflow"
entry_step: "wait"
steps:
  wait:
    kind: "wait"
    params:
      seconds: 0.01
    children: []
"""
        client.post("/workflows/", json={"yaml_content": workflow_yaml})

        # Execute 50 workflows and measure throughput
        start = time.time()
        for _ in range(50):
            response = client.post("/workflows/throughput_test/execute", json={})
            assert response.status_code == 200

        duration = time.time() - start
        throughput = 50 / duration

        # Should handle at least 10 workflows per second
        assert throughput > 10, f"Throughput was {throughput:.1f} workflows/s, expected > 10/s"


@pytest.mark.performance
@pytest.mark.slow
class TestDatabasePerformance:
    """Performance tests for database operations."""

    def test_user_creation_performance(self):
        """Test performance of creating many users."""
        import tempfile
        import os
        from retrofitkit.compliance.users import Users

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["P4_DATA_DIR"] = temp_dir
            users = Users()

            start = time.time()
            for i in range(100):
                users.create(
                    email=f"user{i}@test.com",
                    name=f"User {i}",
                    role="Operator",
                    password="TestPassword123!"
                )
            duration = time.time() - start

            # Should create 100 users in < 2 seconds
            assert duration < 2.0, f"Creating 100 users took {duration:.3f}s"

    def test_approval_query_performance(self):
        """Test performance of querying approvals."""
        import tempfile
        import os
        from retrofitkit.compliance import approvals

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["P4_DATA_DIR"] = temp_dir
            approvals.DB_DIR = temp_dir
            approvals.DB = os.path.join(temp_dir, "system.db")

            # Create many approval requests
            for i in range(100):
                approvals.request(f"recipe_{i}.yaml", f"user{i}@test.com")

            # Measure query performance
            start = time.time()
            pending = approvals.list_pending(limit=200)
            duration = time.time() - start

            assert len(pending) == 100
            # Query should be fast (< 100ms)
            assert duration < 0.1, f"Query took {duration:.3f}s, expected < 0.1s"


@pytest.mark.performance
class TestMemoryUsage:
    """Tests for memory usage patterns."""

    def test_workflow_execution_memory(self, client):
        """Test memory usage during workflow execution."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        workflow_yaml = """
id: "memory_test"
name: "Memory Test Workflow"
entry_step: "wait"
steps:
  wait:
    kind: "wait"
    params:
      seconds: 0.01
    children: []
"""
        client.post("/workflows/", json={"yaml_content": workflow_yaml})

        # Execute workflow multiple times
        for _ in range(100):
            client.post("/workflows/memory_test/execute", json={})

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal (< 50MB)
        assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB, expected < 50MB"


@pytest.mark.performance
@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for performance comparison."""

    def test_benchmark_workflow_parse(self, benchmark):
        """Benchmark workflow YAML parsing."""
        from retrofitkit.core.workflows.models import WorkflowDefinition

        workflow_yaml = """
id: "benchmark_workflow"
name: "Benchmark Workflow"
entry_step: "step1"
steps:
  step1:
    kind: "wait"
    params:
      seconds: 1.0
    children: []
"""

        # Benchmark parsing
        result = benchmark(WorkflowDefinition.from_yaml, workflow_yaml)
        assert result.id == "benchmark_workflow"

    @pytest.mark.asyncio
    async def test_benchmark_workflow_execution(self, benchmark):
        """Benchmark workflow execution."""
        from retrofitkit.core.workflows.models import WorkflowDefinition, WorkflowStep
        from retrofitkit.core.workflows.engine import WorkflowEngine
        from retrofitkit.core.workflows.safety import SafetyManager

        workflow = WorkflowDefinition(
            id="benchmark",
            name="Benchmark",
            steps={
                "wait": WorkflowStep(id="wait", kind="wait", params={"seconds": 0.01})
            },
            entry_step="wait"
        )

        safety = SafetyManager()
        engine = WorkflowEngine(safety)

        # Benchmark execution
        async def run_workflow():
            return await engine.run(workflow)

        # Note: benchmark with async requires special handling
        result = await run_workflow()
        assert result.success is True


@pytest.mark.performance
class TestLoadScenarios:
    """Load testing scenarios."""

    def test_sustained_load(self, client):
        """Test system under sustained load."""
        workflow_yaml = """
id: "load_test"
name: "Load Test Workflow"
entry_step: "wait"
steps:
  wait:
    kind: "wait"
    params:
      seconds: 0.01
    children: []
"""
        client.post("/workflows/", json={"yaml_content": workflow_yaml})

        # Simulate sustained load for 5 seconds
        start = time.time()
        request_count = 0
        errors = 0

        while time.time() - start < 5.0:
            response = client.post("/workflows/load_test/execute", json={})
            request_count += 1
            if response.status_code != 200:
                errors += 1

        duration = time.time() - start
        throughput = request_count / duration
        error_rate = errors / request_count if request_count > 0 else 0

        print(f"\nLoad test results:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Requests: {request_count}")
        print(f"  Throughput: {throughput:.1f} req/s")
        print(f"  Error rate: {error_rate:.2%}")

        # Should handle reasonable load with low error rate
        assert error_rate < 0.01, f"Error rate {error_rate:.2%} too high"

    def test_burst_load(self, client):
        """Test system handling burst traffic."""
        workflow_yaml = """
id: "burst_test"
name: "Burst Test Workflow"
entry_step: "wait"
steps:
  wait:
    kind: "wait"
    params:
      seconds: 0.01
    children: []
"""
        client.post("/workflows/", json={"yaml_content": workflow_yaml})

        def execute_workflow():
            return client.post("/workflows/burst_test/execute", json={})

        # Send burst of 50 concurrent requests
        start = time.time()
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(execute_workflow) for _ in range(50)]
            results = [f.result() for f in futures]
        duration = time.time() - start

        successful = sum(1 for r in results if r.status_code == 200)
        success_rate = successful / len(results)

        print(f"\nBurst test results:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Requests: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Success rate: {success_rate:.2%}")

        # Should handle burst with high success rate
        assert success_rate > 0.95, f"Success rate {success_rate:.2%} too low"
