# 🚀 Deployment Guide: HR Assistant to GitHub Pages

Since GitHub Pages only hosts static files and your HR Assistant is a Streamlit (Python) application, you'll need to deploy it separately and link it from your GitHub Pages site.

## Option 1: Streamlit Cloud (Recommended - FREE)

Streamlit Cloud offers free hosting for Streamlit applications and is the easiest option.

### Step 1: Push Your Code to GitHub

1. If you haven't already, create a repository for your HR Assistant:
   ```bash
   cd "/Users/junfeibai/Desktop/网站/Hr Assistant"
   git init
   git add .
   git commit -m "Initial commit: HR Assistant"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Set the **Main file path** to: `resume_jd_matcher.py`
6. Click "Deploy"

### Step 3: Get Your App URL

After deployment, Streamlit Cloud will provide you with a URL like:
```
https://YOUR_APP_NAME.streamlit.app
```

### Step 4: Update Your GitHub Pages Site

1. Open `Clefairybiubiubiu.github.io/index.html`
2. Replace the placeholder link with your Streamlit Cloud URL
3. Update the `script.js` file to set the correct URL

## Option 2: Other Free Hosting Options

### Railway.app
- Free tier available
- Supports Python applications
- Easy deployment from GitHub

### Render.com
- Free tier available
- Supports Streamlit apps
- Automatic deployments from GitHub

### Heroku (Paid now, but alternatives exist)
- Consider Railway or Render instead

## Configuration Files Created

I've created the following files to help with deployment:

1. **`.streamlit/config.toml`** - Streamlit configuration for production
2. **`DEPLOYMENT.md`** - This deployment guide

## Important Notes

### Environment Variables
If your app uses API keys (like Google Gemini), you'll need to set them as environment variables in your hosting platform:

- **Streamlit Cloud**: Settings → Secrets → Add secrets
- **Railway**: Variables tab
- **Render**: Environment section

### Dependencies
Make sure `resume_matcher_requirements.txt` is up to date. Streamlit Cloud will automatically install dependencies from this file.

### File Structure
Your app should work as-is, but ensure:
- All Python files are in the correct locations
- The main file is `resume_jd_matcher.py`
- Cache directory exists (will be created automatically)

## Testing Locally Before Deployment

Before deploying, test locally:
```bash
cd "/Users/junfeibai/Desktop/网站/Hr Assistant"
streamlit run resume_jd_matcher.py
```

## Updating Your GitHub Pages Link

Once deployed, update the link in your GitHub Pages site by editing:
- `Clefairybiubiubiu.github.io/index.html` - Update the href
- `Clefairybiubiubiu.github.io/script.js` - Update the URL variable

## Troubleshooting

### App won't start
- Check that `resume_jd_matcher.py` is the correct main file
- Verify all dependencies are in `resume_matcher_requirements.txt`
- Check logs in Streamlit Cloud dashboard

### Import errors
- Ensure all Python files are in the repository
- Check that the `resume_matcher` package structure is correct

### API key issues
- Set environment variables in your hosting platform
- Don't commit API keys to GitHub

---

**Need help?** Check the [Streamlit Cloud documentation](https://docs.streamlit.io/streamlit-community-cloud) or open an issue on GitHub.

