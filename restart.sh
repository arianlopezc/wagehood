#!/bin/bash
# Restart Wagehood service

echo "🔄 Restarting Wagehood service..."
./stop.sh
sleep 2
echo "🚀 Starting service..."
nohup python3 start_production_service.py > wagehood_startup.log 2>&1 &
WAGEHOOD_PID=$!
echo $WAGEHOOD_PID > wagehood.pid
echo "✅ Service restarted with PID: $WAGEHOOD_PID"
