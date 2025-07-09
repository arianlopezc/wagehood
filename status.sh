#!/bin/bash
# Check Wagehood service status

echo "🔍 Wagehood Service Status"
echo "========================="

if [[ -f "wagehood.pid" ]]; then
    PID=$(cat wagehood.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✅ Service running (PID: $PID)"
        
        # Check Redis
        if redis-cli ping >/dev/null 2>&1; then
            echo "✅ Redis connection OK"
            
            # Check streams
            STREAM_LEN=$(redis-cli XLEN market_data_stream 2>/dev/null || echo "0")
            echo "📊 Market data stream: $STREAM_LEN messages"
            
            # Check recent data
            if [[ $STREAM_LEN -gt 0 ]]; then
                echo "🔄 Data flowing - service operational"
            else
                echo "⚠️  No data yet - service may be starting"
            fi
        else
            echo "❌ Redis connection failed"
        fi
    else
        echo "❌ Service not running"
        rm -f wagehood.pid
    fi
else
    echo "❌ No PID file found - service not started"
fi

echo
echo "📋 Log files:"
echo "   • Production log: wagehood_production.log"
echo "   • Startup log: wagehood_startup.log"
echo "   • Installation log: installation.log"
