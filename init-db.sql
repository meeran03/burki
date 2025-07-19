-- =============================================================================
-- Burki Voice AI Database Initialization
-- =============================================================================
-- This script is automatically executed when the PostgreSQL container starts
-- It creates the database, user, and installs required extensions

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE burki'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'burki')\gexec

-- Connect to the burki database
\c burki;

-- Create user if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'burki_user') THEN
        CREATE USER burki_user WITH PASSWORD 'burki_password';
    END IF;
END
$$;

-- Grant all privileges on database to user
GRANT ALL PRIVILEGES ON DATABASE burki TO burki_user;

-- Grant all privileges on schema public to user
GRANT ALL ON SCHEMA public TO burki_user;

-- Grant default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO burki_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO burki_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO burki_user;

-- Install required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Install pgvector extension for RAG functionality (if available)
-- This will fail silently if pgvector is not installed
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
    RAISE NOTICE 'pgvector extension installed successfully';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector extension not available - RAG functionality will use fallback';
END
$$;

-- Create a function to generate random IDs
CREATE OR REPLACE FUNCTION generate_random_id() RETURNS TEXT AS $$
BEGIN
    RETURN encode(gen_random_bytes(16), 'hex');
END;
$$ LANGUAGE plpgsql;

-- Set up proper permissions for the user
GRANT USAGE ON SCHEMA public TO burki_user;
GRANT CREATE ON SCHEMA public TO burki_user;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Burki Voice AI database initialized successfully';
    RAISE NOTICE 'Database: burki';
    RAISE NOTICE 'User: burki_user';
    RAISE NOTICE 'Extensions installed: uuid-ossp, pgcrypto, vector (if available)';
END
$$; 