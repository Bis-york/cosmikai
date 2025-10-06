🌌 CosmikAI – Automated Exoplanet Detection

CosmikAI is an open-source platform for automated exoplanet detection and visualization.
It integrates a FastAPI backend for running detection pipelines and a React/Vite frontend for visualization and interactive exploration of planetary systems.

✨ Features

📂 Upload light curve data (CSV, FITS, TSV) or query by star name (Kepler, K2, TESS).

⚡ FastAPI backend with MongoDB caching for results and statistics.

📊 Detection dashboard:

Orbital period, transit duration, transit depth

Confidence score and model performance stats

🪐 3D Orbit Visualizer (React Three Fiber + Three.js) to view planetary orbits interactively.

📖 Detection history stored locally and in MongoDB.

🎛️ Adjustable model parameters (bins, thresholds, feature selection).

📂 Repository Structure
cosmikai/
├── backend/             # FastAPI backend (main app: newmain.py, DB, prediction pipeline)
├── base_frontend/       # Primary React frontend (Vite + Tailwind)
├── visual_frontend/     # Extra frontend visual prototypes
├── data_test/           # Example datasets (CSV/FITS for testing)
├── model code/          # Model training + analysis code
├── outputs/             # Generated outputs & logs
├── photod/              # Image assets (logos, etc.)
├── requirements.txt     # Python dependencies
├── server_setup.py      # Backend setup helper
├── package-lock.json    # NPM lockfile
├── .env                 # Environment variables (API URLs, DB connection)
└── README.md

🚀 Getting Started
1. Clone the repository
git clone https://github.com/<your-org>/cosmikai.git
cd cosmikai

3. Backend Setup (FastAPI)

Requires Python 3.9+

cd backend
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows

pip install --upgrade pip
pip install -r ../requirements.txt


Run the backend:

uvicorn newmain:app --reload --port 8000


Backend runs on → http://127.0.0.1:8000

3. Frontend Setup (React + Vite)

Requires Node.js 18+

cd base_frontend
npm install
1) Base app (base_frontend/)

Copy env template and set URLs:

cd base_frontend
cp .env.example .env


Set:

VITE_API_BASE_URL=https://api.yourdomain.tld

VITE_VISUAL_BASE_URL=https://visuals.yourdomain.tld (where the visual explorer is hosted)

Build and deploy the static site:

npm install
npm run build


Host the dist/ folder on your domain (Vercel/Netlify/Cloudflare Pages/Nginx/etc).

2) Visual explorer (visual_frontend/)

Copy env template and set:

cd visual_frontend
cp .env.example .env


VITE_API_BASE_URL=https://api.yourdomain.tld

Build and deploy:

npm install
npm run build


Host the dist/ folder on your chosen domain.

 

🧪 Usage

Open the frontend UI.

Choose:

Upload mode → upload your CSV/FITS/TSV light curve file.

Query mode → enter a star name or ID (Kepler, K2, TESS, etc.).

Adjust advanced options:

Number of bins

Confidence threshold

Click Detect Exoplanets to run analysis.

Explore results:

Detection cards

Confidence scores

Interactive 3D orbit visualizer

History of past detections

🛠️ Tech Stack

Frontend: React, Vite, TailwindCSS, Recharts, React Three Fiber, Three.js

Backend: FastAPI, Uvicorn, Pydantic, MongoDB

Data: NASA Kepler, K2, TESS missions

🤝 Contributing

Contributions are welcome!

Fork this repo

Create a feature branch (git checkout -b feat/my-feature)

Commit changes (git commit -m "feat: add something")

Push branch & open a Pull Request

📜 License

Licensed under the MIT License.
See LICENSE
 for details.

🌠 Acknowledgements

NASA Exoplanet Archive (Kepler, K2, TESS)

FastAPI, React, and Vite ecosystem

Inspiration from the ongoing search for habitable worlds
