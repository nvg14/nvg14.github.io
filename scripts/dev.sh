#!/bin/bash

# Personal Website Development Script
# Usage: ./scripts/dev.sh [command]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

show_help() {
    echo "Personal Website Development Script"
    echo ""
    echo "Usage: ./scripts/dev.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup     - Set up development environment"
    echo "  serve     - Start development server"
    echo "  build     - Build static site"
    echo "  clean     - Clean build artifacts"
    echo "  deploy    - Deploy to GitHub Pages (via git push)"
    echo "  help      - Show this help message"
    echo ""
}

setup_env() {
    echo "🚀 Setting up development environment..."
    
    # Check if virtual environment exists
    if [ ! -d "env" ]; then
        echo "Creating virtual environment..."
        python3 -m venv env
    fi
    
    # Activate virtual environment
    source env/bin/activate
    
    # Upgrade pip and install dependencies
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo "✅ Setup complete! Run './scripts/dev.sh serve' to start development server."
}

serve_site() {
    echo "🌐 Starting development server..."
    source env/bin/activate
    mkdocs serve --dev-addr=127.0.0.1:8000
}

build_site() {
    echo "🔨 Building static site..."
    source env/bin/activate
    mkdocs build --verbose --clean --strict
    echo "✅ Build complete! Static files are in ./site directory."
}

clean_build() {
    echo "🧹 Cleaning build artifacts..."
    rm -rf site/
    rm -rf .mkdocs_cache/
    echo "✅ Clean complete!"
}

deploy_site() {
    echo "🚀 Deploying to GitHub Pages..."
    echo "Building site first..."
    build_site
    
    echo "Pushing to GitHub (this will trigger GitHub Actions)..."
    git add .
    read -p "Enter commit message: " commit_msg
    git commit -m "$commit_msg"
    git push origin main
    
    echo "✅ Deployed! GitHub Actions will build and deploy your site."
    echo "Check https://github.com/nvg14/nvg14.github.io/actions for deployment status."
}

# Main script logic
case "${1:-}" in
    setup)
        setup_env
        ;;
    serve)
        serve_site
        ;;
    build)
        build_site
        ;;
    clean)
        clean_build
        ;;
    deploy)
        deploy_site
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        echo "❌ No command specified."
        show_help
        exit 1
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_help
        exit 1
        ;;
esac 