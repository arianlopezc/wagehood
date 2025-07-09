#!/bin/bash
# Stop Wagehood service

echo "🛑 Stopping Wagehood service..."

if [[ -f "wagehood.pid" ]]; then
    PID=$(cat wagehood.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "Stopping service (PID: $PID)..."
        kill $PID
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 $PID 2>/dev/null; then
                echo "✅ Service stopped gracefully"
                rm -f wagehood.pid
                exit 0
            fi
            sleep 1
        done
        
        # Force kill if necessary
        echo "⚠️  Forcing service shutdown..."
        kill -9 $PID 2>/dev/null || true
        rm -f wagehood.pid
        echo "✅ Service force stopped"
    else
        echo "⚠️  Service not running"
        rm -f wagehood.pid
    fi
else
    echo "⚠️  No PID file found"
fi

# Also kill any stray processes
pkill -f "start_production_service" 2>/dev/null || true
echo "✅ Cleanup complete"
