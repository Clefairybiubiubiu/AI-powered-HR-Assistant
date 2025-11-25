# Google Gemini API Integration Guide

This guide explains how to use Google Gemini (FREE tier) to enhance the HR Assistant's accuracy and capabilities.

## Overview

The HR Assistant uses Google Gemini API for:

- **Enhanced Match Explanations**: More detailed, natural language explanations of why candidates match job descriptions
- **Better Professional Summaries**: AI-generated professional summaries from resume sections
- **Improved Resume Parsing**: Enhanced extraction of structured information from resumes

## Why Google Gemini?

- ✅ **FREE tier** with generous limits
- ✅ No credit card required
- ✅ High-quality AI responses
- ✅ Easy to set up
- ✅ Fast and reliable

## Quick Setup

### Step 1: Install Package

```bash
pip install google-generativeai
```

### Step 2: Get Free API Key

1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### Step 3: Set API Key

**On Mac/Linux:**

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

**On Windows:**

```cmd
set GOOGLE_API_KEY=your-api-key-here
```

**In Python (temporary):**

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
```

### Step 4: Run Application

```bash
streamlit run resume_jd_matcher.py
```

Enable "AI-Powered Enhancements" in the sidebar!

## Features Enhanced by Gemini

### 1. Match Explanations

**Before (Rule-based):**

> "Strong match in skills with 0.75 similarity."

**After (Gemini-enhanced):**

> "This candidate demonstrates strong alignment with the job requirements, particularly in technical skills and experience. The candidate's 5+ years of data engineering experience at leading tech companies, combined with expertise in Python, Spark, and cloud platforms, closely matches the role's requirements. The main strengths include hands-on experience with big data technologies and proven track record in scalable system design."

### 2. Professional Summaries

**Before (Rule-based):**

> "Data Engineer with 5+ years of experience. Proficient in Python, SQL, Spark."

**After (Gemini-enhanced):**

> "Experienced Data Engineer with over 5 years of professional experience specializing in big data processing and cloud infrastructure. Proven expertise in designing and implementing scalable data pipelines using Python, Apache Spark, and SQL, with hands-on experience in AWS and GCP environments. Strong background in optimizing data workflows and collaborating with cross-functional teams to deliver high-impact data solutions."

## How It Works

1. **Automatic Detection**: The system automatically detects if Google Gemini API is configured
2. **Graceful Fallback**: If Gemini is unavailable or fails, the system falls back to rule-based methods
3. **Error Handling**: All API calls are wrapped in try-except blocks to ensure the application continues working even if the API fails

## Configuration

### Default Model

- **Model**: `gemini-pro` (free tier)
- **Max Tokens**: 200-500 (depending on use case)
- **Temperature**: 0.5-0.7 (for balanced creativity and accuracy)

### Customizing Settings

The system uses optimized defaults, but you can modify them in `resume_matcher/utils/llm_client.py` if needed.

## Cost

**Google Gemini FREE Tier:**

- **Cost**: $0 (completely free)
- **Rate Limits**: Generous free tier limits
- **No credit card required**

## Troubleshooting

### Issue: "Google Gemini API not configured"

**Solution:**

1. Check that you've installed the package: `pip install google-generativeai`
2. Verify your API key is set: `echo $GOOGLE_API_KEY`
3. Restart the Streamlit application after setting environment variables

### Issue: "Failed to initialize Google Gemini client"

**Solution:**

1. Check your API key is valid
2. Ensure you have internet connectivity
3. Verify package installation: `pip show google-generativeai`
4. Check API service status

### Issue: API calls are slow

**Solution:**

1. This is normal for free tier (may have rate limits)
2. The system will fall back to rule-based methods if needed
3. Consider upgrading to paid tier if you need faster responses

### Issue: Rate limits

**Solution:**

1. Free tier has rate limits - this is normal
2. The system automatically falls back to rule-based methods
3. Wait a few minutes and try again
4. Consider upgrading to paid tier for higher limits

## Best Practices

1. **Set API key in environment**: More secure than hardcoding
2. **Monitor usage**: Keep track of your API usage (though free tier is generous)
3. **Use fallback methods**: The system works without Gemini, so you can disable it if needed
4. **Test thoroughly**: Test with your specific resumes and job descriptions to ensure quality

## Security Notes

- **Never commit API keys to version control**
- Use environment variables or secure secret management
- Consider using API key rotation
- Monitor API usage for unusual activity

## Example Usage

```python
from resume_matcher.utils.llm_client import get_llm_client, is_llm_available

# Check if Gemini is available
if is_llm_available():
    llm_client = get_llm_client()

    # Generate explanation
    explanation = llm_client.generate_match_explanation(
        resume_name="John Doe",
        jd_name="Senior Data Engineer",
        match_score=0.85,
        section_scores={"skills": 0.9, "experience": 0.8, "education": 0.7},
        resume_summary="5+ years of data engineering experience...",
        jd_requirements="Looking for senior data engineer with Spark experience..."
    )

    print(explanation)
```

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the code in `resume_matcher/utils/llm_client.py`
3. Check Google Gemini API documentation: https://ai.google.dev/docs

## Summary

Google Gemini provides a **completely free** way to enhance your HR Assistant with AI-powered explanations and summaries. Setup takes just 2 minutes, and you get high-quality AI responses without any cost!
