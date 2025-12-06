# Quick Start: Deploy HR Assistant

## 🎯 What You Need to Do

### 1. Push HR Assistant to GitHub
```bash
cd "/Users/junfeibai/Desktop/网站/Hr Assistant"
git init
git add .
git commit -m "HR Assistant ready for deployment"
git remote add origin https://github.com/YOUR_USERNAME/hr-assistant.git
git push -u origin main
```

### 2. Deploy to Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file: `resume_jd_matcher.py`
6. Click "Deploy"

### 3. Update GitHub Pages Link
After deployment, you'll get a URL like: `https://your-app.streamlit.app`

Edit `Clefairybiubiubiu.github.io/script.js`:
- Find `const HR_ASSISTANT_URL = "#";`
- Replace `"#"` with your Streamlit URL: `"https://your-app.streamlit.app"`

Then commit and push to your GitHub Pages repository.

## ✅ Files Created/Updated

- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `requirements.txt` - Dependencies (copied from resume_matcher_requirements.txt)
- ✅ `DEPLOYMENT.md` - Detailed deployment guide
- ✅ Updated GitHub Pages `index.html` - Added HR Assistant link
- ✅ Updated GitHub Pages `script.js` - Added link handler

## 📝 Notes

- Streamlit Cloud is **FREE** and perfect for Streamlit apps
- Your app will be accessible at a `.streamlit.app` URL
- API keys can be set in Streamlit Cloud's "Secrets" section
- Updates: Just push to GitHub, Streamlit Cloud auto-deploys

