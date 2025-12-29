# imgshape UI - Vercel Deployment

## Environment Variables

Set these in Vercel project settings:

```
VITE_API_BASE_URL=https://imgshape-947614788790.asia-south1.run.app
```

## Deploy Steps

1. **Connect GitHub Repository**
   - Go to [vercel.com](https://vercel.com)
   - Click "Add New..." → "Project"
   - Select your GitHub repo: `STiFLeR7/imgshape`
   - Import the project

2. **Configure Settings**
   - Root Directory: `ui`
   - Framework: `Vite`
   - Build Command: `npm run build`
   - Output Directory: `dist`

3. **Add Environment Variable**
   - In Vercel project settings → Environment Variables
   - Add: `VITE_API_BASE_URL` = `https://imgshape-947614788790.asia-south1.run.app`

4. **Deploy**
   - Click "Deploy"
   - Vercel will build and deploy automatically

## Live URLs

- **UI**: https://imgshape.vercel.app (or your custom domain)
- **API**: https://imgshape-947614788790.asia-south1.run.app
  - Health: `/health`
  - Analyze v4: `/v4/analyze`
  - Augment: `/augment`
  - Report: `/generate_report`

## Local Development

```bash
cd ui
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Notes

- UI is deployed on free Vercel tier
- Backend API runs on Cloud Run (also cost-efficient)
- Environment variable automatically passed to build
- The API base URL can be overridden at build time
