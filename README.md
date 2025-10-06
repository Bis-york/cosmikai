<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** CosmiKai README Template based on the Best-README-Template.
*** Replace placeholder values (owner, repo, contact info) with your project details.
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="96" height="96">
  </a>

<h3 align="center">CosmiKai Exoplanet Detection Suite</h3>

  <p align="center">
    A hybrid research stack for detecting exoplanet candidates from Kepler and TESS light curves.
    <br />
    <a href="https://github.com/github_username/repo_name"><strong>Explore the docs &raquo;</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name#usage">View Demo</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

[![CosmiKai dashboard preview][product-screenshot]](https://example.com)

CosmiKai combines a PyTorch-based 1D convolutional network with astronomy tooling to identify transit signatures in stellar light curves. The repository packages:

- A FastAPI service (`backend/api.py`) that exposes `/score-target` and `/process-config` inference endpoints.
- A data analysis toolkit (`backend/data_analyzer.py`) that loads checkpoints, folds light curves, and produces diagnostic metrics.
- Predictive utilities (`backend/predict.py`) that integrate with NASA's Lightkurve API for on-demand Kepler/TESS downloads.
- React + Vite frontends (`Frontend/`, `visual_frontend/`, `base_frontend/`) for data visualization and operator workflows.
- Reusable testing harnesses (`backend/test_predict.py`, `backend/test_db_connection.py`) and configuration examples for CSV- and JSON-based ingestion.

Use this template to spin up a fresh CosmiKai deployment, adapt it for new missions, or document downstream applications.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python-badge]][Python-url]
* [![FastAPI][FastAPI-badge]][FastAPI-url]
* [![PyTorch][PyTorch-badge]][PyTorch-url]
* [![Lightkurve][Lightkurve-badge]][Lightkurve-url]
* [![Pydantic][Pydantic-badge]][Pydantic-url]
* [![React][React.js]][React-url]
* [![Vite][Vite-badge]][Vite-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

Follow the steps below to spin up the full stack locally. Use the provided `example_*` configs to sanity-check your setup.

### Prerequisites

- Python 3.11+ with `pip`
- Node.js 20+ with `npm` or `pnpm`
- Git LFS (recommended for large checkpoints)

```sh
# install project-level tooling
pip install --upgrade pip
python -m venv .venv && .\\.venv\\Scripts\\activate    # Windows example
pip install -r requirements.txt
npm install -g npm@latest                              # optional: front-end tooling
```

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/github_username/repo_name.git
   cd repo_name
   ```
2. Activate your virtual environment and install backend requirements
   ```sh
   python -m venv .venv
   source .venv/bin/activate        # use .venv\\Scripts\\activate on Windows
   pip install -r requirements.txt
   ```
3. (Optional) Install front-end dependencies
   ```sh
   cd Frontend
   npm install
   cd ..
   ```
4. Configure secrets and checkpoints
   - Place pretrained models in `models/` (defaults to `trained_exoplanet_detector.pth`).
   - Copy sample configs (`backend/example_*`) and adjust mission/author/threshold values.
5. Update Git remotes if you forked the template
   ```sh
   git remote set-url origin git@github.com:github_username/repo_name.git
   git remote -v
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

- **Run the API**
  ```sh
  uvicorn backend.api:app --reload
  ```
  - Health check: `GET http://localhost:8000/health`
  - Score a target: `POST http://localhost:8000/score-target`
    ```json
    {
      "target": "Kepler-10",
      "mission": "Kepler",
      "threshold": 0.6
    }
    ```
- **Batch process configs** with the CLI
  ```sh
  python -m backend.predict --config backend/example_lightkurve_config.json
  ```
- **Visualize results** by launching the Vite front-end
  ```sh
  cd Frontend
  npm run dev
  ```
- **Run tests**
  ```sh
  pytest backend
  ```

_For more examples, explore the `/guides` directory and the sample notebooks in `models/`._

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [ ] Expand mission presets (TESS, K2, PLATO)
- [ ] Surface periodogram diagnostics in the UI
- [ ] Add async job orchestration for large target batches
    - [ ] Plug into Lightkurve bulk download pipeline
- [ ] Publish Docker images for backend + frontend

See the [open issues](https://github.com/github_username/repo_name/issues) for the full list of proposed features.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions power CosmiKai's research velocity-whether you are polishing the UI, tuning models, or adding pipelines. To get started:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a pull request with screenshots, test runs, and context

Please review the `guides/CONTRIBUTING.md` (coming soon) for coding standards, data governance rules, and model release notes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/Bis-york/cosmikai/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Bis-york/cosmikai" alt="contrib.rocks image" />
</a>

## License

Distributed under the `project_license`. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

CosmiKai Team - [@twitter_handle](https://twitter.com/twitter_handle) - team@email_client.com

Project Link: [https://github.com/Bis-york/cosmikai](https://github.com/Bis-york/cosmikai)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

* [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
* [Lightkurve](https://docs.lightkurve.org/)
* [BATMAN Transit Model](https://www.cfa.harvard.edu/~lkreidberg/batman/)
* [Astropy Project](https://www.astropy.org/)
* [PyTorch](https://pytorch.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png

[Python-badge]: https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[FastAPI-badge]: https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white
[FastAPI-url]: https://fastapi.tiangolo.com/
[PyTorch-badge]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Lightkurve-badge]: https://img.shields.io/badge/Lightkurve-1B2A4B?style=for-the-badge
[Lightkurve-url]: https://docs.lightkurve.org/
[Pydantic-badge]: https://img.shields.io/badge/Pydantic-0082C9?style=for-the-badge&logo=pydantic&logoColor=white
[Pydantic-url]: https://docs.pydantic.dev/
[Vite-badge]: https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vite-url]: https://vitejs.dev/




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

If you want to run on your device/domain.

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
