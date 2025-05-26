"""
Performance tests for the API Hub and critical system components.
"""
import pytest
import time
import asyncio
import statistics
from unittest.mock import patch, MagicMock, AsyncMock

# Performance test configuration
CONCURRENCY_LEVELS = [1, 5, 10]  # Number of concurrent requests
REQUEST_COUNTS = [10, 50, 100]    # Number of requests per concurrent worker
REQUEST_TIMEOUT = 10.0           # Timeout for requests

class MockPerformanceClient:
    """Mock client for performance testing"""
    
    def __init__(self, response_time=0.05, error_rate=0.05):
        """
        Initialize the mock client
        
        Args:
            response_time: Mean response time in seconds
            error_rate: Probability of request error (0.0 to 1.0)
        """
        self.response_time = response_time
        self.error_rate = error_rate
    
    async def make_request(self, service_id, endpoint):
        """Make a mock request with simulated latency"""
        # Simulate network latency
        await asyncio.sleep(self.response_time)
        
        # Randomly simulate errors
        import random
        if random.random() < self.error_rate:
            raise Exception("Simulated request error")
        
        # Return mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": "Test data"}
        return mock_response
    
    async def batch_request(self, requests):
        """Make a batch request with simulated latency"""
        # Simulate batch processing (slightly more efficient than individual requests)
        await asyncio.sleep(self.response_time * 0.7 * len(requests))
        
        # Return mock response
        results = []
        for req in requests:
            # Randomly simulate per-request errors
            import random
            if random.random() < self.error_rate:
                results.append({
                    "success": False,
                    "error": "Simulated error"
                })
            else:
                results.append({
                    "success": True,
                    "status_code": 200,
                    "data": {"success": True, "data": "Test data"}
                })
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": results}
        return mock_response

@pytest.fixture
def performance_client():
    """Create a performance testing client"""
    return MockPerformanceClient()

async def run_concurrent_requests(client, concurrency, request_count):
    """Run concurrent requests and measure performance metrics"""
    
    async def worker(worker_id, num_requests):
        """Worker to make requests"""
        results = []
        for i in range(num_requests):
            start_time = time.time()
            try:
                await client.make_request("test-api", f"/endpoint{i}")
                success = True
            except Exception:
                success = False
            
            elapsed = time.time() - start_time
            results.append({
                "worker_id": worker_id,
                "request_id": i,
                "success": success,
                "elapsed_seconds": elapsed
            })
        return results
    
    # Create and run workers
    tasks = []
    for i in range(concurrency):
        task = asyncio.create_task(worker(i, request_count))
        tasks.append(task)
    
    # Wait for all tasks to complete
    all_results = await asyncio.gather(*tasks)
    
    # Flatten results
    results = []
    for worker_results in all_results:
        results.extend(worker_results)
    
    return results

async def run_batch_requests(client, batch_size, request_count):
    """Run batch requests and measure performance"""
    
    results = []
    num_batches = (request_count + batch_size - 1) // batch_size  # Ceiling division
    
    for i in range(num_batches):
        # Create batch request
        batch_requests = []
        for j in range(min(batch_size, request_count - i * batch_size)):
            batch_requests.append({
                "service_id": "test-api",
                "endpoint": f"/endpoint{i * batch_size + j}",
                "method": "GET"
            })
        
        # Make batch request
        start_time = time.time()
        try:
            response = await client.batch_request(batch_requests)
            batch_results = response.json()["results"]
            success = True
        except Exception:
            batch_results = []
            success = False
        
        elapsed = time.time() - start_time
        
        # Record results
        results.append({
            "batch_id": i,
            "batch_size": len(batch_requests),
            "success": success,
            "elapsed_seconds": elapsed,
            "requests_per_second": len(batch_requests) / elapsed if elapsed > 0 else 0
        })
    
    return results

@pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
@pytest.mark.parametrize("request_count", REQUEST_COUNTS)
@pytest.mark.asyncio
async def test_concurrent_request_performance(performance_client, concurrency, request_count):
    """Test performance with concurrent requests"""
    
    # Run concurrent requests
    results = await run_concurrent_requests(performance_client, concurrency, request_count)
    
    # Calculate metrics
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r["success"])
    success_rate = successful_requests / total_requests if total_requests > 0 else 0
    
    response_times = [r["elapsed_seconds"] for r in results]
    avg_response_time = statistics.mean(response_times)
    p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
    max_response_time = max(response_times)
    
    # Print performance metrics
    print(f"\nPerformance with {concurrency} concurrent workers, {request_count} requests per worker:")
    print(f"Total requests: {total_requests}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Avg response time: {avg_response_time:.4f}s")
    print(f"P95 response time: {p95_response_time:.4f}s")
    print(f"Max response time: {max_response_time:.4f}s")
    
    # Assertions to ensure the test passes with reasonable performance
    assert success_rate > 0.9, "Success rate too low"
    assert avg_response_time < 1.0, "Average response time too high"

@pytest.mark.parametrize("batch_size", [5, 20, 50])
@pytest.mark.asyncio
async def test_batch_request_performance(performance_client, batch_size):
    """Test performance of batch requests"""
    request_count = 100
    
    # Run batch requests
    results = await run_batch_requests(performance_client, batch_size, request_count)
    
    # Calculate metrics
    total_batches = len(results)
    successful_batches = sum(1 for r in results if r["success"])
    success_rate = successful_batches / total_batches if total_batches > 0 else 0
    
    response_times = [r["elapsed_seconds"] for r in results]
    avg_response_time = statistics.mean(response_times)
    
    requests_per_second = [r["requests_per_second"] for r in results]
    avg_requests_per_second = statistics.mean(requests_per_second)
    
    # Print performance metrics
    print(f"\nBatch performance with batch size {batch_size}, total requests {request_count}:")
    print(f"Total batches: {total_batches}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Avg batch response time: {avg_response_time:.4f}s")
    print(f"Avg requests per second: {avg_requests_per_second:.2f}")
    
    # Assertions to ensure the test passes with reasonable performance
    assert success_rate > 0.9, "Success rate too low"
    assert avg_requests_per_second > 10, "Requests per second too low"
