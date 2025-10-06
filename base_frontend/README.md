# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.


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

<a href="https://github.com/github_username/repo_name/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=github_username/repo_name" alt="contrib.rocks image" />
</a>

## License

Distributed under the `project_license`. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

CosmiKai Team - [@twitter_handle](https://twitter.com/twitter_handle) - team@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

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


