#!/usr/bin/env python3
"""
OpenCode LRS Integration: Benchmark Testing Module

Adds benchmark testing capabilities to the web interface.
Integrates Chaos Scriptorium and GAIA lightweight benchmarks.
"""

import json
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import our lightweight benchmarks
from lrs_agents.lrs.benchmarking.lightweight_benchmarks import (
    LightweightChaosBenchmark,
    LightweightGAIABenchmark,
    run_lightweight_benchmarks,
)

# Create router for benchmark endpoints
benchmark_router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


# Data models
class BenchmarkRequest(BaseModel):
    benchmark_type: str  # "chaos", "gaia", or "all"
    num_trials: int = 5
    max_steps: int = 20


class BenchmarkResponse(BaseModel):
    success: bool
    benchmark_type: str
    results: Dict[str, Any]
    execution_time: float
    message: str


@benchmark_router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest):
    """Run specified benchmark"""
    import time

    start_time = time.time()

    try:
        if request.benchmark_type == "chaos":
            # Run Chaos Scriptorium benchmark
            benchmark = LightweightChaosBenchmark()
            results = benchmark.run(num_trials=request.num_trials)

            return BenchmarkResponse(
                success=True,
                benchmark_type="chaos",
                results=results,
                execution_time=time.time() - start_time,
                message=f"Chaos Scriptorium benchmark completed: {results['success_rate']:.1%} success rate",
            )

        elif request.benchmark_type == "gaia":
            # Run GAIA benchmark
            benchmark = LightweightGAIABenchmark()
            results = benchmark.run(num_tasks=request.num_trials)

            return BenchmarkResponse(
                success=True,
                benchmark_type="gaia",
                results=results,
                execution_time=time.time() - start_time,
                message=f"GAIA benchmark completed: {results['success_rate']:.1%} success rate",
            )

        elif request.benchmark_type == "all":
            # Run all benchmarks
            results = run_lightweight_benchmarks()

            return BenchmarkResponse(
                success=True,
                benchmark_type="all",
                results=results,
                execution_time=time.time() - start_time,
                message="All benchmarks completed successfully",
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown benchmark type: {request.benchmark_type}",
            )

    except Exception as e:
        return BenchmarkResponse(
            success=False,
            benchmark_type=request.benchmark_type,
            results={},
            execution_time=time.time() - start_time,
            message=f"Benchmark failed: {str(e)}",
        )


@benchmark_router.get("/status")
async def get_benchmark_status():
    """Get benchmark system status"""
    return {
        "status": "ready",
        "available_benchmarks": ["chaos", "gaia", "all"],
        "lightweight_implementation": True,
        "numpy_independent": True,
        "description": "Lightweight LRS benchmark system without NumPy dependencies",
    }


@benchmark_router.get("/results")
async def get_latest_results():
    """Get latest benchmark results if available"""
    try:
        with open("lightweight_benchmark_results.json", "r") as f:
            results = json.load(f)
        return {"results": results, "cached": True}
    except FileNotFoundError:
        return {
            "results": None,
            "cached": False,
            "message": "No cached results available",
        }


# Integration functions for main application
def integrate_benchmarks_into_main(app):
    """Integrate benchmark endpoints into main FastAPI app"""
    app.include_router(benchmark_router)
    print("✅ Benchmark endpoints integrated")


def add_benchmark_ui_to_main_html():
    """Add benchmark testing UI to the main HTML interface"""

    benchmark_html = """
    <!-- Benchmark Testing Section -->
    <div class="bg-white rounded-xl shadow-lg p-8 mb-8">
        <h2 class="text-3xl font-bold text-center text-gray-900 mb-8">Benchmark Testing Suite</h2>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Benchmark Controls -->
            <div>
                <h3 class="text-xl font-semibold text-gray-900 mb-4">Run Benchmarks</h3>

                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Benchmark Type</label>
                        <select id="benchmarkType" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="chaos">Chaos Scriptorium</option>
                            <option value="gaia">GAIA Tasks</option>
                            <option value="all">All Benchmarks</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Number of Trials/Tasks</label>
                        <input type="number" id="numTrials" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" value="5" min="1" max="20">
                    </div>

                    <button onclick="runBenchmark()" class="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 px-4 rounded-md hover:from-purple-700 hover:to-pink-700 transition-all duration-300">
                        Run Benchmark Test
                    </button>
                </div>

                <!-- Benchmark Info -->
                <div class="mt-6 p-4 bg-gray-50 rounded-lg">
                    <h4 class="font-semibold text-gray-900 mb-2">Benchmark Types</h4>
                    <div class="space-y-2 text-sm text-gray-600">
                        <div><strong>Chaos Scriptorium:</strong> Tests resilience in volatile environments with changing permissions</div>
                        <div><strong>GAIA:</strong> Tests multi-step reasoning on diverse tasks</div>
                        <div><strong>All:</strong> Runs both benchmarks sequentially</div>
                    </div>
                </div>
            </div>

            <!-- Benchmark Results -->
            <div>
                <h3 class="text-xl font-semibold text-gray-900 mb-4">Benchmark Results</h3>

                <div id="benchmarkResults" class="bg-gray-50 rounded-lg p-4 min-h-64">
                    <div class="text-center text-gray-500 py-8">
                        <svg class="w-12 h-12 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                        </svg>
                        <p>Run a benchmark to see results</p>
                        <p class="text-xs mt-2">Lightweight implementation - no NumPy required</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """

    benchmark_js = """
    async function runBenchmark() {
        const benchmarkType = document.getElementById('benchmarkType').value;
        const numTrials = parseInt(document.getElementById('numTrials').value);

        const resultsContainer = document.getElementById('benchmarkResults');
        resultsContainer.innerHTML = '<div class="text-center py-8"><div class="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div><p class="text-gray-600">Running benchmark...</p></div>';

        try {
            const response = await axios.post('/benchmarks/run', {
                benchmark_type: benchmarkType,
                num_trials: numTrials,
                max_steps: 20
            });

            const data = response.data;
            displayBenchmarkResults(data);

        } catch (error) {
            console.error('Benchmark error:', error);
            resultsContainer.innerHTML = '<div class="text-center py-8 text-red-600"><p class="font-semibold">Benchmark failed</p><p class="text-sm mt-2">' + (error.response?.data?.detail || error.message) + '</p></div>';
        }
    }

    function displayBenchmarkResults(data) {
        const resultsContainer = document.getElementById('benchmarkResults');

        if (data.success) {
            const results = data.results;

            let html = '<div class="space-y-4">';

            // Success header
            html += '<div class="flex items-center justify-between"><h4 class="font-semibold text-green-600">✅ Benchmark Completed</h4><span class="text-sm bg-green-100 text-green-800 px-2 py-1 rounded">' + data.execution_time.toFixed(2) + 's</span></div>';

            // Results based on benchmark type
            if (data.benchmark_type === 'chaos' || (data.benchmark_type === 'all' && results.chaos)) {
                const chaos = data.benchmark_type === 'all' ? results.chaos : results;
                html += '<div class="bg-white p-4 rounded border-l-4 border-purple-500"><h5 class="font-medium text-purple-900 mb-2">🏭 Chaos Scriptorium</h5>';
                html += '<div class="grid grid-cols-2 gap-4 text-sm"><div><strong>Success Rate:</strong> ' + (chaos.success_rate * 100).toFixed(1) + '%</div>';
                html += '<div><strong>Avg Steps:</strong> ' + chaos.avg_steps.toFixed(1) + '</div>';
                html += '<div><strong>Avg Time:</strong> ' + chaos.avg_execution_time.toFixed(2) + 's</div>';
                html += '<div><strong>Trials:</strong> ' + chaos.total_trials + '</div></div></div>';
            }

            if (data.benchmark_type === 'gaia' || (data.benchmark_type === 'all' && results.gaia)) {
                const gaia = data.benchmark_type === 'all' ? results.gaia : results;
                html += '<div class="bg-white p-4 rounded border-l-4 border-blue-500"><h5 class="font-medium text-blue-900 mb-2">🌍 GAIA Tasks</h5>';
                html += '<div class="grid grid-cols-2 gap-4 text-sm"><div><strong>Success Rate:</strong> ' + (gaia.success_rate * 100).toFixed(1) + '%</div>';
                html += '<div><strong>Avg Time:</strong> ' + gaia.avg_execution_time.toFixed(2) + 's</div>';
                html += '<div><strong>Tasks:</strong> ' + gaia.total_tasks + '</div>';
                html += '<div><strong>Final Precision:</strong> ' + gaia.final_precision.execution.toFixed(3) + '</div></div></div>';
            }

            // Integration notes
            html += '<div class="text-xs text-gray-500 mt-4 p-3 bg-gray-100 rounded">';
            html += '<strong>LRS Integration Notes:</strong><br>';
            html += '• Lightweight implementation (no NumPy dependencies)<br>';
            html += '• Precision tracking with hierarchical adaptation<br>';
            html += '• Free energy calculations for decision optimization<br>';
            html += '• Real-time performance monitoring and adaptation';
            html += '</div>';

            html += '</div>';
            resultsContainer.innerHTML = html;

        } else {
            resultsContainer.innerHTML = '<div class="text-center py-8 text-red-600"><p class="font-semibold">Benchmark Failed</p><p class="text-sm mt-2">' + data.message + '</p></div>';
        }
    }
    """

    return benchmark_html, benchmark_js


# Export integration functions
__all__ = [
    "benchmark_router",
    "integrate_benchmarks_into_main",
    "add_benchmark_ui_to_main_html",
    "run_lightweight_benchmarks",
]
