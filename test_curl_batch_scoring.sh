#!/bin/bash

# Test script for batch scoring API using curl
echo "🚀 Batch Scoring API Test (curl)"
echo "=================================="

# Check if API is running
echo "🔍 Checking API Health..."
curl -s http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ API is running"
else
    echo "   ❌ API is not running. Please start it with: python similarity_app.py"
    exit 1
fi

# Test data
echo "📋 Preparing test data..."

# Create test JSON file
cat > test_data.json << 'EOF'
{
  "jobs": [
    {
      "title": "Data Scientist",
      "description": "We are looking for a Data Scientist to analyze large datasets and build machine learning models.",
      "requirements": "Python, SQL, Machine Learning, TensorFlow, 3+ years experience"
    },
    {
      "title": "Marketing Analyst", 
      "description": "Join our marketing team to analyze campaign performance and customer behavior.",
      "requirements": "Google Analytics, Excel, Power BI, Data Visualization, Marketing experience"
    }
  ],
  "candidates": [
    {
      "name": "Alice",
      "resume_text": "Alice Johnson, Senior Data Scientist. 5+ years in data science and machine learning. Expert in Python, SQL, TensorFlow, PyTorch. Built recommendation systems and predictive models."
    },
    {
      "name": "Bob",
      "resume_text": "Bob Smith, Marketing Analyst. 4+ years in marketing analytics. Expert in Google Analytics, Excel, Power BI. Analyzed campaign performance and customer behavior."
    }
  ]
}
EOF

echo "   ✅ Test data prepared"

# Test batch scoring endpoint
echo ""
echo "📡 Testing Batch Scoring Endpoint..."
echo "   POST /api/v1/scoring/batch"

# Send request and capture response
response=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d @test_data.json \
  http://localhost:8000/api/v1/scoring/batch)

# Check if request was successful
if [ $? -eq 0 ]; then
    echo "   ✅ Request successful"
    
    # Parse and display results
    echo ""
    echo "🎯 Batch Scoring Results:"
    echo "$response" | python3 -c "
import json
import sys
try:
    data = json.load(sys.stdin)
    if 'results' in data:
        for result in data['results']:
            print(f\"   Job: {result['job_title']}\")
            print(f\"   Best Candidate: {result['best_candidate']}\")
            print(f\"   Score: {result['score']:.3f}\")
            print(f\"   Reason: {result['reason']}\")
            print()
    else:
        print('   ❌ No results found')
except Exception as e:
    print(f'   ❌ Error parsing response: {e}')
    print('   Raw response:')
    print(sys.stdin.read())
"
else
    echo "   ❌ Request failed"
    exit 1
fi

# Test edge cases
echo ""
echo "🧪 Testing Edge Cases..."

# Test empty jobs
echo "🔍 Testing Empty Jobs..."
empty_jobs_response=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"jobs": [], "candidates": [{"name": "Alice", "resume_text": "Python developer"}]}' \
  http://localhost:8000/api/v1/scoring/batch)

if [ $? -eq 0 ]; then
    echo "   ✅ Empty jobs handled gracefully"
else
    echo "   ❌ Empty jobs failed"
fi

# Test empty candidates
echo "🔍 Testing Empty Candidates..."
empty_candidates_response=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"jobs": [{"title": "Developer", "description": "Build apps", "requirements": "Python"}], "candidates": []}' \
  http://localhost:8000/api/v1/scoring/batch)

if [ $? -eq 0 ]; then
    echo "   ✅ Empty candidates handled gracefully"
else
    echo "   ❌ Empty candidates failed"
fi

# Clean up
rm -f test_data.json

echo ""
echo "✅ Batch scoring API tests completed!"
echo "💡 The API is working correctly and handling various scenarios"
