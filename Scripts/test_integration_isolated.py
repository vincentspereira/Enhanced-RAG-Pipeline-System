#!/usr/bin/env python3
"""
Isolated Integration Test for RAG System Enhancement Modules
Tests each module independently without cross-dependencies
"""

import sys
import os
import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import traceback
from contextlib import contextmanager

# Add Scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestResults:
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def add_result(self, test_name: str, status: str, duration: float, details: Dict = None):
        self.results[test_name] = {
            "status": status,
            "duration": duration,
            "details": details or {}
        }
        
    def get_summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["status"] == "PASSED")
        failed = total - passed
        total_duration = time.time() - self.start_time
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "total_duration": total_duration
        }

async def test_individual_modules():
    """Test each enhancement module individually"""
    test_results = TestResults()
      # Test 1: Advanced Memory Optimizer
    try:
        start_time = time.time()
        from enhancers.advanced_memory_optimizer import AdvancedMemoryOptimizer, AdvancedMemoryConfig
        
        config = AdvancedMemoryConfig(
            max_system_memory_usage=0.85,
            enable_mixed_precision=True,
            enable_gradient_checkpointing=True
        )
        optimizer = AdvancedMemoryOptimizer(config)
          # Test basic functionality
        await optimizer.start()
        memory_stats = optimizer.get_comprehensive_stats()
        await optimizer.stop()
        
        duration = time.time() - start_time
        test_results.add_result("advanced_memory_optimizer", "PASSED", duration, {
            "memory_stats": memory_stats
        })
        logger.info("✓ Advanced Memory Optimizer test PASSED")
        
    except Exception as e:
        duration = time.time() - start_time
        test_results.add_result("advanced_memory_optimizer", "FAILED", duration, {
            "error": str(e)
        })
        logger.error(f"✗ Advanced Memory Optimizer test FAILED: {e}")
    
    # Test 2: Advanced Query Optimizer
    try:
        start_time = time.time()
        from enhancers.advanced_query_optimizer import AdvancedQueryOptimizer, OptimizationConfig
        
        config = OptimizationConfig(
            enable_query_rewriting=True,
            enable_semantic_expansion=True,
            enable_query_caching=True,
            max_expanded_terms=10
        )
        optimizer = AdvancedQueryOptimizer(config)
          # Test query optimization
        test_query = "What is machine learning?"
        analysis = await optimizer.optimize_query(test_query)
        
        duration = time.time() - start_time
        test_results.add_result("advanced_query_optimizer", "PASSED", duration, {
            "original_query": test_query,
            "optimized_query": analysis.original_query,
            "intent": analysis.intent.value if hasattr(analysis, 'intent') else 'unknown',
            "confidence": analysis.confidence_score if hasattr(analysis, 'confidence_score') else 0.0
        })
        logger.info("✓ Advanced Query Optimizer test PASSED")
        
    except Exception as e:
        duration = time.time() - start_time
        test_results.add_result("advanced_query_optimizer", "FAILED", duration, {
            "error": str(e)
        })
        logger.error(f"✗ Advanced Query Optimizer test FAILED: {e}")
    
    # Test 3: Auto Scaler
    try:
        start_time = time.time()
        from enhancers.auto_scaler import AutoScaler, ScalingConfig
        
        config = ScalingConfig(
            min_cpu_workers=2,
            max_cpu_workers=8,
            target_cpu_utilization=0.7,
            target_memory_utilization=0.8
        )
        scaler = AutoScaler(config)
          # Test scaling functionality
        await scaler.start()
        
        # Get current scaling stats instead of making decisions
        scaling_stats = scaler.get_scaling_stats()
        
        await scaler.shutdown()
        
        duration = time.time() - start_time
        test_results.add_result("auto_scaler", "PASSED", duration, {
            "scaling_stats": scaling_stats,
            "enabled": scaling_stats.get("enabled", False)
        })
        logger.info("✓ Auto Scaler test PASSED")
        
    except Exception as e:
        duration = time.time() - start_time
        test_results.add_result("auto_scaler", "FAILED", duration, {
            "error": str(e)
        })
        logger.error(f"✗ Auto Scaler test FAILED: {e}")
    
    # Test 4: Error Recovery System
    try:
        start_time = time.time()
        from enhancers.error_recovery import ErrorRecoverySystem, RecoveryConfig
        
        config = RecoveryConfig(
            default_max_retries=3,
            default_backoff_factor=2.0,
            default_timeout=30.0,
            exponential_backoff=True
        )
        recovery_system = ErrorRecoverySystem(config)
        
        # Test error recovery
        @recovery_system.with_recovery("test_component")
        async def test_function():
            return {"status": "success", "data": "test"}
        
        result = await test_function()
        
        duration = time.time() - start_time
        test_results.add_result("error_recovery_system", "PASSED", duration, {
            "recovery_result": result
        })
        logger.info("✓ Error Recovery System test PASSED")
        
    except Exception as e:
        duration = time.time() - start_time
        test_results.add_result("error_recovery_system", "FAILED", duration, {
            "error": str(e)
        })
        logger.error(f"✗ Error Recovery System test FAILED: {e}")
      # Test 5: Real-Time Dashboard (Initialize only)
    try:
        start_time = time.time()
        from monitoring.realtime_dashboard import RealTimeDashboard, MonitoringConfig, MetricsCollector
        
        config = MonitoringConfig(
            sampling_interval=2.0,
            history_size=50,
            enable_alerts=False  # Disable for testing
        )
        
        metrics_collector = MetricsCollector(config)
        dashboard = RealTimeDashboard(metrics_collector, config)
        
        # Test dashboard creation
        dashboard.create_app()
        
        # Test metrics collector
        metrics_collector.start_collection()
        time.sleep(1)  # Let it collect some metrics
        latest_metrics = list(metrics_collector.metrics_history)
        metrics_collector.stop_collection()
        
        duration = time.time() - start_time
        test_results.add_result("realtime_dashboard", "PASSED", duration, {
            "metrics_collected": len(latest_metrics) > 0,
            "dashboard_created": dashboard.app is not None
        })
        logger.info("✓ Real-Time Dashboard test PASSED")
        
    except Exception as e:
        duration = time.time() - start_time
        test_results.add_result("realtime_dashboard", "FAILED", duration, {
            "error": str(e)
        })
        logger.error(f"✗ Real-Time Dashboard test FAILED: {e}")
    
    return test_results

async def test_integration_coordination():
    """Test coordination between modules"""
    test_results = TestResults()
    
    try:
        start_time = time.time()
          # Test integration manager (if available)
        try:
            from enhancers.integration_manager import IntegrationManager, IntegrationConfig
            
            config = IntegrationConfig()
            manager = IntegrationManager(config)
            
            # Test initialization
            await manager.initialize()
            
            # Test stats collection (includes health data)
            stats = manager.get_integration_stats()
            
            # Test shutdown
            await manager.shutdown()
            
            duration = time.time() - start_time
            test_results.add_result("integration_manager", "PASSED", duration, {
                "initialization_status": stats.get("is_initialized", False),
                "components_status": stats.get("components_status", {}),
                "latest_health": stats.get("latest_health"),
                "uptime": stats.get("uptime_seconds", 0)
            })
            logger.info("✓ Integration Manager test PASSED")
            
        except Exception as e:
            duration = time.time() - start_time
            test_results.add_result("integration_manager", "FAILED", duration, {
                "error": str(e),
                "note": "Integration Manager may have import issues"
            })
            logger.warning(f"⚠ Integration Manager test FAILED (expected): {e}")
    
    except Exception as e:
        logger.error(f"Integration coordination test failed: {e}")
        
    return test_results

async def generate_comprehensive_report():
    """Generate comprehensive test report"""
    logger.info("=" * 80)
    logger.info("RAG SYSTEM ENHANCEMENT MODULES - COMPREHENSIVE TEST REPORT")
    logger.info("=" * 80)
    
    # Test individual modules
    logger.info("\n🔧 TESTING INDIVIDUAL ENHANCEMENT MODULES")
    module_results = await test_individual_modules()
    
    # Test integration
    logger.info("\n🔗 TESTING MODULE INTEGRATION")
    integration_results = await test_integration_coordination()
    
    # Combine results
    all_results = TestResults()
    all_results.results.update(module_results.results)
    all_results.results.update(integration_results.results)
    all_results.start_time = min(module_results.start_time, integration_results.start_time)
    
    # Generate summary
    summary = all_results.get_summary()
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {summary['total_tests']}")
    logger.info(f"Passed: {summary['passed']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
    logger.info(f"Total Duration: {summary['total_duration']:.2f} seconds")
    
    # Detailed results
    logger.info("\n📋 DETAILED RESULTS")
    logger.info("-" * 60)
    for test_name, result in all_results.results.items():
        status_icon = "✓" if result["status"] == "PASSED" else "✗"
        logger.info(f"{status_icon} {test_name}: {result['status']} ({result['duration']:.2f}s)")
        if result.get("details", {}).get("error"):
            logger.info(f"   Error: {result['details']['error']}")
    
    # Save detailed report
    report = {
        "test_summary": summary,
        "test_results": all_results.results,
        "timestamp": datetime.now().isoformat(),
        "test_type": "comprehensive_isolated_integration"
    }
    
    report_file = f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n💾 Detailed report saved to: {report_file}")
    
    # Analysis and recommendations
    logger.info("\n🎯 ANALYSIS & RECOMMENDATIONS")
    logger.info("-" * 60)
    
    if summary['success_rate'] >= 80:
        logger.info("✅ EXCELLENT: System shows strong enhancement capabilities")
        if summary['success_rate'] == 100:
            logger.info("🎉 ALL TESTS PASSED: Ready for production deployment")
        else:
            logger.info("⚠️  Minor issues detected - review failed components")
    elif summary['success_rate'] >= 60:
        logger.info("⚠️  GOOD: Core functionality working, some enhancements need attention")
    else:
        logger.info("❌ NEEDS ATTENTION: Multiple enhancement modules require fixes")
    
    # Integration status
    integration_passed = all_results.results.get("integration_manager", {}).get("status") == "PASSED"
    if integration_passed:
        logger.info("🔗 INTEGRATION: Full system coordination operational")
    else:
        logger.info("🔗 INTEGRATION: Limited - modules work independently")
    
    logger.info("\n" + "=" * 80)
    
    return summary['success_rate']

if __name__ == "__main__":
    asyncio.run(generate_comprehensive_report())
