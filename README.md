# Nithin V. Gopal - Personal Website

[![Built with MkDocs](https://img.shields.io/badge/Built%20with-MkDocs-blue)](https://www.mkdocs.org/)
[![Material Theme](https://img.shields.io/badge/Theme-Material-green)](https://squidfunk.github.io/mkdocs-material/)
[![GitHub Pages](https://img.shields.io/badge/Deployed%20on-GitHub%20Pages-orange)](https://nvg14.github.io/)

A modern, minimalistic personal website and technical blog showcasing my work in media technology, machine learning, and software engineering.

🌐 **Live Site:** [nvg14.github.io](https://nvg14.github.io)

## About

This is the personal website of **Nithin V. Gopal**, Lead Engineer - Product Development at Amagi Media Labs. The site features a clean, minimalistic design with a focus on readability and user experience.

## Technology Stack

- **Static Site Generator**: [MkDocs](https://www.mkdocs.org/)
- **Theme**: [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- **Styling**: Custom CSS with CSS Variables
- **Typography**: [Inter](https://rsms.me/inter/) + [JetBrains Mono](https://www.jetbrains.com/lp/mono/)
- **Deployment**: GitHub Pages

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nvg14/nvg14.github.io.git
   cd nvg14.github.io
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start development server**
   ```bash
   mkdocs serve
   ```

5. **Open your browser**
   Navigate to `http://127.0.0.1:8000`

## Project Structure

```
nvg14.github.io/
├── docs/                          # Source content
│   ├── assets/                    # Images and static files
│   │   └── images/               # Profile pictures, logos
│   ├── css/                      # Custom stylesheets
│   │   └── neoteroi-mkdocs.css  # Main stylesheet
│   ├── posts/                    # Blog posts
│   │   └── 2025/                # Posts organized by year
│   ├── timeline/                 # Career timeline data
│   │   └── career.json          # Professional experience
│   └── index.md                 # Homepage content
├── overrides/                    # Theme customizations
│   ├── main.html               # Base template override
│   └── partials/               # Partial templates
│       └── footer.html         # Custom footer
├── mkdocs.yaml                  # MkDocs configuration
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Development Workflow

### Adding a New Blog Post

1. Create a new Markdown file in `docs/posts/YEAR/`
2. Add the post to the navigation in `mkdocs.yaml`
3. Test locally with `mkdocs serve`
4. Commit and push to deploy

### Updating Content

- **Homepage**: Edit `docs/index.md`
- **Styling**: Modify `docs/css/neoteroi-mkdocs.css`
- **Theme**: Update files in `overrides/`
- **Configuration**: Edit `mkdocs.yaml`

### Local Development

```bash
# Setup development environment
./scripts/dev.sh setup

# Start development server with auto-reload
./scripts/dev.sh serve
# OR: mkdocs serve

# Build static site for production
./scripts/dev.sh build
# OR: mkdocs build

# Deploy to GitHub Pages (automatically handled by GitHub Actions)
./scripts/dev.sh deploy
# OR: git push origin main (triggers automatic deployment)
```

### GitHub Actions Deployment

This project uses GitHub Actions for automatic deployment to GitHub Pages:

- **Deploy Workflow**: Automatically builds and deploys on push to `main` branch
- **Test Workflow**: Tests builds on pull requests without deploying
- **Live Site**: [https://nvg14.github.io](https://nvg14.github.io)

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed setup instructions.

## Contact

- **Email**: [nithinvg1495@amagi.com](mailto:nithinvg1495@amagi.com)
- **LinkedIn**: [nithin_v_gopal](http://linkedin.com/nithin_v_gopal)
- **GitHub**: [nvg14](https://github.com/nvg14)