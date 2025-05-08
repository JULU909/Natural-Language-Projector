#!/bin/bash

create_env_file() {
    echo "Creating .env file..."
    cat <<EOF > .env
HUGGINGFACEHUB_API_TOKEN=
OPENAI_API_KEY=
EOF
    echo ".env file created successfully."
}

# Function to install pyenv if not present
install_pyenv() {
    if ! command -v pyenv &> /dev/null; then
        echo "Pyenv not found. Installing pyenv..."
        
        # Install dependencies
        sudo apt update && sudo apt install -y \
            make build-essential libssl-dev zlib1g-dev \
            libbz2-dev libreadline-dev libsqlite3-dev \
            wget curl llvm libncurses5-dev xz-utils tk-dev \
            libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

        # Install pyenv
        curl https://pyenv.run | bash

        # Add pyenv to bash configuration
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

        # Apply changes
        source ~/.bashrc
    else
        echo "Pyenv is already installed."
    fi
}

# Function to install Python 3.9 using pyenv
install_python() {
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"

    if ! pyenv versions | grep -q "3.9"; then
        echo "Installing Python 3.9..."
        pyenv install 3.9
    else
        echo "Python 3.9 is already installed."
    fi
}

# Function to create and activate the virtual environment
setup_virtualenv() {

    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"

    if ! pyenv virtualenvs | grep -q "microled"; then
        echo "Creating virtual environment..."
        pyenv virtualenv 3.9 microled
    else
        echo "Virtual environment already exists."
    fi

    echo "Activating virtual environment..."
    pyenv activate microled
}

# Function to install dependencies and run the app
setup_and_run_app() {
    pyenv activate microled
    echo "🐍 Using Python version: $(python --version)"
    echo "📍 Python path: $(which python)"
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "Error: requirements.txt not found!"
        exit 1
    fi

    echo "Running app.py..."
    if [ -f "app.py" ]; then
        python app.py
    else
        echo "Error: app.py not found!"
        exit 1
    fi
}

# Run functions
install_pyenv
install_python
setup_virtualenv
setup_and_run_app
