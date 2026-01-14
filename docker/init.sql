-- Initialize database for LRS-Agents
-- This script sets up the schema for storing agent execution history

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table for storing agent execution runs
CREATE TABLE IF NOT EXISTS agent_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    task TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(50) NOT NULL DEFAULT 'running',
    final_precision JSONB,
    total_steps INTEGER,
    adaptations INTEGER,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Table for storing precision history
CREATE TABLE IF NOT EXISTS precision_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    level VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL CHECK (value >= 0 AND value <= 1),
    prediction_error FLOAT CHECK (prediction_error >= 0 AND prediction_error <= 1),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Table for storing tool execution history
CREATE TABLE IF NOT EXISTS tool_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    tool_name VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    success BOOLEAN NOT NULL,
    prediction_error FLOAT NOT NULL CHECK (prediction_error >= 0 AND prediction_error <= 1),
    execution_time_ms INTEGER,
    error_message TEXT,
    input_data JSONB,
    output_data JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Table for storing adaptation events
CREATE TABLE IF NOT EXISTS adaptation_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    trigger_tool VARCHAR(255),
    trigger_error FLOAT,
    old_precision FLOAT,
    new_precision FLOAT,
    action_taken TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Table for storing benchmark results
CREATE TABLE IF NOT EXISTS benchmark_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    benchmark_name VARCHAR(255) NOT NULL,
    run_date DATE NOT NULL DEFAULT CURRENT_DATE,
    num_trials INTEGER NOT NULL,
    success_rate FLOAT CHECK (success_rate >= 0 AND success_rate <= 1),
    avg_steps FLOAT,
    avg_adaptations FLOAT,
    avg_execution_time FLOAT,
    results JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Table for multi-agent coordination logs
CREATE TABLE IF NOT EXISTS coordination_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    action VARCHAR(255) NOT NULL,
    social_precision JSONB,
    message_sent TEXT,
    message_received TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_agent_runs_agent_id ON agent_runs(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status);
CREATE INDEX IF NOT EXISTS idx_agent_runs_started_at ON agent_runs(started_at);

CREATE INDEX IF NOT EXISTS idx_precision_history_run_id ON precision_history(run_id);
CREATE INDEX IF NOT EXISTS idx_precision_history_level ON precision_history(level);

CREATE INDEX IF NOT EXISTS idx_tool_executions_run_id ON tool_executions(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_executions_tool_name ON tool_executions(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_executions_success ON tool_executions(success);

CREATE INDEX IF NOT EXISTS idx_adaptation_events_run_id ON adaptation_events(run_id);

CREATE INDEX IF NOT EXISTS idx_benchmark_results_name ON benchmark_results(benchmark_name);
CREATE INDEX IF NOT EXISTS idx_benchmark_results_date ON benchmark_results(run_date);

CREATE INDEX IF NOT EXISTS idx_coordination_logs_session ON coordination_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_coordination_logs_agent ON coordination_logs(agent_id);

-- Views for common queries
CREATE OR REPLACE VIEW agent_performance_summary AS
SELECT 
    agent_id,
    COUNT(*) as total_runs,
    AVG(total_steps) as avg_steps,
    AVG(adaptations) as avg_adaptations,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds
FROM agent_runs
WHERE completed_at IS NOT NULL
GROUP BY agent_id;

CREATE OR REPLACE VIEW tool_reliability AS
SELECT 
    tool_name,
    COUNT(*) as total_executions,
    SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate,
    AVG(prediction_error) as avg_prediction_error,
    AVG(execution_time_ms) as avg_execution_time_ms
FROM tool_executions
GROUP BY tool_name;

-- Function to clean old data
CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 90)
RETURNS void AS $$
BEGIN
    DELETE FROM agent_runs 
    WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;
    
    -- Related tables will cascade delete
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO lrs_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO lrs_user;

-- Insert sample data for testing
INSERT INTO agent_runs (agent_id, task, status, final_precision, total_steps, adaptations, metadata)
VALUES 
    ('test_agent_1', 'Sample task', 'completed', '{"execution": 0.75, "planning": 0.68}', 10, 2, '{"test": true}'),
    ('test_agent_2', 'Another task', 'completed', '{"execution": 0.82, "planning": 0.79}', 8, 1, '{"test": true}');

COMMIT;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'LRS-Agents database initialized successfully!';
END $$;

