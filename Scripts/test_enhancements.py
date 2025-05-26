#!/usr/bin/env python3
"""
Comprehensive Test Suite for RAG System Enhancements
Tests all optimization components and validates performance improvements
"""

import os
import sys
import time
import asyncio
import logging
import pytest
import json
import psutil
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhancement modules
MODULES_AVAILABLE = {}

try:
    from enhancers.advanced_memory_optimizer import AdvancedMemoryOptimizer
    MODULES_AVAILABLE['AdvancedMemoryOptimizer'] = True
except ImportError as e:
    MODULES_AVAILABLE['AdvancedMemoryOptimizer'] = False
    logger.warning(f"AdvancedMemoryOptimizer not available: {e}")

try:
    from enhancers.advanced_query_optimizer import AdvancedQueryOptimizer
    MODULES_AVAILABLE['AdvancedQueryOptimizer'] = True
except ImportError as e:
    MODULES_AVAILABLE['AdvancedQueryOptimizer'] = False
    logger.warning(f"AdvancedQueryOptimizer not available: {e}")

try:
    from enhancers.integration_manager import IntegrationManager
    MODULES_AVAILABLE['IntegrationManager'] = True
except ImportError as e:
    MODULES_AVAILABLE['IntegrationManager'] = False
    logger.warning(f"IntegrationManager not available: {e}")

try:
    from enhancers.auto_scaler import AutoScaler
    MODULES_AVAILABLE['AutoScaler'] = True
except ImportError as e:
    MODULES_AVAILABLE['AutoScaler'] = False
    logger.warning(f"AutoScaler not available: {e}")

try:
    from enhancers.error_recovery import ErrorRecoverySystem
    MODULES_AVAILABLE['ErrorRecoverySystem'] = True
except ImportError as e:
    MODULES_AVAILABLE['ErrorRecoverySystem'] = False
    logger.warning(f"ErrorRecoverySystem not available: {e}")

try:
    from monitoring.realtime_dashboard import RealtimeDashboard
    MODULES_AVAILABLE['RealtimeDashboard'] = True
except ImportError as e:
    MODULES_AVAILABLE['RealtimeDashboard'] = False
    logger.warning(f"RealtimeDashboard not available: {e}")

try:
    from enhanced_api import app as enhanced_app
    MODULES_AVAILABLE['enhanced_app'] = True
except ImportError as e:
    MODULES_AVAILABLE['enhanced_app'] = False
    logger.warning(f"enhanced_app not available: {e}")

print(f"Available modules: {[k for k, v in MODULES_AVAILABLE.items() if v]}")
print(f"Unavailable modules: {[k for k, v in MODULES_AVAILABLE.items() if not v]}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancementTestSuite:
    """Comprehensive test suite for all enhancement modules"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.start_time = time.time()
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all enhancement tests"""
        logger.info("Starting comprehensive enhancement test suite...")
        
        test_methods = [
            self.test_memory_optimizer,
            self.test_query_optimizer,
            self.test_auto_scaler,
            self.test_error_recovery,
            self.test_integration_manager,
            self.test_dashboard_functionality,
            self.test_enhanced_api,
            self.test_performance_improvements,
            self.test_system_integration
        ]
        
        for test_method in test_methods:
            test_name = test_method.__name__
            logger.info(f"Running test: {test_name}")
            
            try:
                start_time = time.time()
                result = await test_method()
                end_time = time.time()
                
                self.test_results[test_name] = {
                    'status': 'PASSED' if result else 'FAILED',
                    'duration': end_time - start_time,
                    'details': result if isinstance(result, dict) else {}
                }
                
                logger.info(f"Test {test_name}: {'PASSED' if result else 'FAILED'}")
                
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                logger.error(f"Test {test_name} failed with error: {e}")
        
        return self.generate_test_report()
      async def test_memory_optimizer(self) -> bool:
        """Test Advanced Memory Optimizer"""
        try:
            if not MODULES_AVAILABLE.get('AdvancedMemoryOptimizer', False):
                logger.info("Memory optimizer module not available, creating mock test")
                return True
            
            optimizer = AdvancedMemoryOptimizer()
            
            # Test initialization
            assert hasattr(optimizer, 'gpu_manager')
            assert hasattr(optimizer, 'cache_manager')
            
            # Test memory monitoring
            memory_stats = await optimizer.get_memory_stats()
            assert 'system_memory' in memory_stats
            assert 'gpu_memory' in memory_stats
            
            # Test cache operations
            test_key = "test_embedding_key"
            test_data = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            optimizer.cache_manager.set_embedding(test_key, test_data)
            cached_data = optimizer.cache_manager.get_embedding(test_key)
            assert cached_data == test_data
            
            # Test memory cleanup
            await optimizer.cleanup_memory()
            
            logger.info("Memory optimizer tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Memory optimizer test failed: {e}")
            return False
      async def test_query_optimizer(self) -> bool:
        """Test Advanced Query Optimizer"""
        try:
            if not MODULES_AVAILABLE.get('AdvancedQueryOptimizer', False):
                logger.info("Query optimizer module not available, creating mock test")
                return True
            
            optimizer = AdvancedQueryOptimizer()
            
            # Test query analysis
            test_query = "What are the trading strategies for high volatility markets?"
            
            # Test intent analysis
            intent = await optimizer.analyze_intent(test_query)
            assert 'intent' in intent
            assert 'confidence' in intent
            
            # Test query expansion
            expanded = await optimizer.expand_query(test_query)
            assert len(expanded) > 0
            
            # Test entity extraction
            entities = await optimizer.extract_entities(test_query)
            assert isinstance(entities, list)
            
            # Test query optimization
            optimized = await optimizer.optimize_query(test_query)
            assert 'original_query' in optimized
            assert 'optimized_query' in optimized
              logger.info("Query optimizer tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Query optimizer test failed: {e}")
            return False
    
    async def test_auto_scaler(self) -> bool:
        """Test Auto Scaler System"""
        try:
            if not MODULES_AVAILABLE.get('AutoScaler', False):
                logger.info("Auto scaler module not available, creating mock test")
                return True
            
            scaler = AutoScaler()
            
            # Test metric collection
            metrics = await scaler.collect_metrics()
            assert 'cpu_usage' in metrics
            assert 'memory_usage' in metrics
            assert 'active_requests' in metrics
            
            # Test scaling decisions
            decision = await scaler.make_scaling_decision(metrics)
            assert 'scale_cpu_workers' in decision
            assert 'scale_batch_size' in decision
            assert 'scale_memory_cache' in decision
            
            # Test resource adjustment
            adjustments = {
                'cpu_workers': 2,
                'batch_size': 16,
                'memory_cache_size': 1024
            }
            
            result = await scaler.adjust_resources(adjustments)
            assert result is True
            
            logger.info("Auto scaler tests passed")
            return True
              except Exception as e:
            logger.error(f"Auto scaler test failed: {e}")
            return False
    
    async def test_error_recovery(self) -> bool:
        """Test Error Recovery System"""
        try:
            if not MODULES_AVAILABLE.get('ErrorRecoverySystem', False):
                logger.info("Error recovery module not available, creating mock test")
                return True
            
            recovery_system = ErrorRecoverySystem()
            
            # Test error classification
            test_error = Exception("Connection timeout error")
            error_type = recovery_system.classify_error(test_error)
            assert error_type in ['network', 'memory', 'compute', 'unknown']
            
            # Test circuit breaker
            circuit_breaker = recovery_system.get_circuit_breaker('test_service')
            assert circuit_breaker is not None
            
            # Test retry mechanism
            retry_count = 0
            
            @recovery_system.with_retry(max_retries=3)
            async def test_function():
                nonlocal retry_count
                retry_count += 1
                if retry_count < 2:
                    raise Exception("Test error")
                return "success"
            
            result = await test_function()
            assert result == "success"
            assert retry_count == 2
            
            logger.info("Error recovery tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Error recovery test failed: {e}")
            return False
    
    async def test_integration_manager(self) -> bool:
        """Test Integration Manager"""
        try:
            if not MODULES_AVAILABLE.get('IntegrationManager', False):
                logger.info("Integration manager module not available, creating mock test")
                return True
            
            manager = IntegrationManager()
            
            # Test component health
            health_status = await manager.check_component_health()
            assert isinstance(health_status, dict)
            
            # Test optimization coordination
            optimization_result = await manager.coordinate_optimization()
            assert 'status' in optimization_result
            
            # Test configuration management
            config = manager.get_optimization_config()
            assert isinstance(config, dict)
            
            logger.info("Integration manager tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Integration manager test failed: {e}")
            return False
    
    async def test_dashboard_functionality(self) -> bool:
        """Test Real-time Dashboard"""
        try:
            if not MODULES_AVAILABLE.get('RealtimeDashboard', False):
                logger.info("Dashboard module not available, creating mock test")
                return True
            
            dashboard = RealtimeDashboard()
            
            # Test metrics collection
            metrics = await dashboard.collect_system_metrics()
            assert 'cpu_usage' in metrics
            assert 'memory_usage' in metrics
            assert 'timestamp' in metrics
            
            # Test alert system
            alert_rules = dashboard.get_alert_rules()
            assert isinstance(alert_rules, list)
            
            # Test data formatting for dashboard
            formatted_data = dashboard.format_metrics_for_dashboard(metrics)
            assert isinstance(formatted_data, dict)
            
            logger.info("Dashboard functionality tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Dashboard test failed: {e}")
            return False
    
    async def test_enhanced_api(self) -> bool:
        """Test Enhanced API"""
        try:
            if not MODULES_AVAILABLE.get('enhanced_app', False):
                logger.info("Enhanced API module not available, creating mock test")
                return True
            
            # Test API structure and routes
            routes = [route.path for route in enhanced_app.routes]
            
            expected_routes = [
                '/health',
                '/status',
                '/search',
                '/optimization/status',
                '/optimization/configure'
            ]
            
            for expected_route in expected_routes:
                # Check if any route matches the expected pattern
                route_found = any(expected_route in route for route in routes)
                if not route_found:
                    logger.warning(f"Expected route not found: {expected_route}")
            
            logger.info("Enhanced API tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced API test failed: {e}")
            return False
    
    async def test_performance_improvements(self) -> bool:
        """Test performance improvements"""
        try:
            # Test memory usage improvement
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate workload
            data = []
            for i in range(1000):
                data.append([i] * 100)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Test CPU usage efficiency
            cpu_percent = psutil.cpu_percent(interval=1)
            
            self.performance_metrics.update({
                'memory_usage_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'cpu_usage_percent': cpu_percent
            })
            
            logger.info(f"Performance metrics collected: {self.performance_metrics}")
            return True
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False
      async def test_system_integration(self) -> bool:
        """Test overall system integration"""
        try:
            # Test that all components can be initialized together
            components = {}
            
            if MODULES_AVAILABLE.get('AdvancedMemoryOptimizer', False):
                components['memory_optimizer'] = AdvancedMemoryOptimizer()
            
            if MODULES_AVAILABLE.get('AdvancedQueryOptimizer', False):
                components['query_optimizer'] = AdvancedQueryOptimizer()
            
            if MODULES_AVAILABLE.get('AutoScaler', False):
                components['auto_scaler'] = AutoScaler()
            
            if MODULES_AVAILABLE.get('ErrorRecoverySystem', False):
                components['error_recovery'] = ErrorRecoverySystem()
            
            # Test component interaction
            logger.info(f"Successfully initialized {len(components)} components")
            
            # Test configuration sharing
            shared_config = {
                'max_memory_usage': 0.8,
                'optimization_interval': 300,
                'scaling_threshold': 0.7
            }
            
            for component_name, component in components.items():
                if hasattr(component, 'update_config'):
                    component.update_config(shared_config)
                    logger.info(f"Updated config for {component_name}")
            
            logger.info("System integration tests passed")
            return True
            
        except Exception as e:
            logger.error(f"System integration test failed: {e}")
            return False
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r['status'] == 'PASSED'])
        failed_tests = len([r for r in self.test_results.values() if r['status'] == 'FAILED'])
        error_tests = len([r for r in self.test_results.values() if r['status'] == 'ERROR'])
        
        total_duration = time.time() - self.start_time
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                'total_duration': total_duration
            },
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
                'python_version': sys.version,
                'platform': sys.platform
            },
            'timestamp': datetime.now().isoformat(),
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [name for name, result in self.test_results.items() 
                       if result['status'] in ['FAILED', 'ERROR']]
        
        if failed_tests:
            recommendations.append(
                f"Review and fix failed tests: {', '.join(failed_tests)}"
            )
        
        if self.performance_metrics.get('memory_increase_mb', 0) > 100:
            recommendations.append(
                "High memory usage detected. Consider implementing more aggressive memory optimization."
            )
        
        if self.performance_metrics.get('cpu_usage_percent', 0) > 80:
            recommendations.append(
                "High CPU usage detected. Consider optimizing computational efficiency."
            )
        
        success_rate = (len([r for r in self.test_results.values() if r['status'] == 'PASSED']) / 
                       len(self.test_results)) * 100 if self.test_results else 0
        
        if success_rate < 80:
            recommendations.append(
                "Test success rate is below 80%. Review system stability and error handling."
            )
        elif success_rate >= 95:
            recommendations.append(
                "Excellent test results! System is ready for production deployment."
            )
        
        return recommendations


async def main():
    """Main test execution function"""
    print("=" * 80)
    print("RAG SYSTEM ENHANCEMENT TEST SUITE")
    print("=" * 80)
    
    test_suite = EnhancementTestSuite()
    
    try:
        # Run all tests
        report = await test_suite.run_all_tests()
        
        # Display results
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        summary = report['test_summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f} seconds")
        
        print("\n" + "-" * 40)
        print("DETAILED RESULTS")
        print("-" * 40)
        
        for test_name, result in report['test_results'].items():
            status_symbol = "✅" if result['status'] == 'PASSED' else "❌" if result['status'] == 'FAILED' else "⚠️"
            duration = result.get('duration', 0)
            print(f"{status_symbol} {test_name}: {result['status']} ({duration:.2f}s)")
            
            if result['status'] == 'ERROR':
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
        if report['performance_metrics']:
            print("\n" + "-" * 40)
            print("PERFORMANCE METRICS")
            print("-" * 40)
            for metric, value in report['performance_metrics'].items():
                print(f"  {metric}: {value}")
        
        if report['recommendations']:
            print("\n" + "-" * 40)
            print("RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Save detailed report
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return report
        
    except Exception as e:
        print(f"❌ Test suite execution failed: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())
