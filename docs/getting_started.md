# Getting Started

This document contains instructions to get a fully working development environment for running this repo.
Currently, these instructions cover macOS, Ubuntu, zsh and bash.


## 1. pyenv

### Installation on macOS

Follow instructions on: https://github.com/pyenv/pyenv#homebrew-on-macos

### Installation on Ubuntu

```sh
# Install pre-requisites
sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash
```

Configure by adding the following to your `~/.zshrc` or `~/.bashrc`:

```sh
# Pyenv environment variables
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

# Pyenv initialization
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

Basic usage:

```sh
# Check Python versions
pyenv install --list

# Install the Python version defined in this repo
pyenv install $(cat .python-version)

# See installed Python versions
pyenv versions
```


## 2. [pyenv-virtualenvwrapper](https://github.com/pyenv/pyenv-virtualenvwrapper)

### Installation on macOS

```sh
# Install with homebrew (recommended if you installed pyenv with homebrew)
brew install pyenv-virtualenvwrapper
```

### Installation on Ubuntu
```sh
# Clone git repo into the pyenv plugins directory. This will make the pyenv virtualenvwrapper
# and pyenv virtualenvwrapper_lazy commands available.
git clone https://github.com/pyenv/pyenv-virtualenvwrapper.git $(pyenv root)/plugins/pyenv-virtualenvwrapper
```

Configure by adding the following to your `~/.zshrc` or `~/.bashrc`:

```sh
# pyenv-virtualenvwrapper
export PYENV_VIRTUALENVWRAPPER_PREFER_PYVENV="true"
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/code  # <- change this to wherever you store your repos
export VIRTUALENVWRAPPER_PYTHON=$HOME/.pyenv/shims/python
pyenv virtualenvwrapper_lazy
```

Test everything is working by opening a new shell (e.g. new Terminal window):

```sh
# Change to the Python version you just installed
pyenv shell $(cat .python-version)
# This only needs to be run once after installing a new Python version through pyenv
# in order to initialise virtualenvwrapper for this Python version
python -m pip install --upgrade pip
python -m pip install virtualenvwrapper
pyenv virtualenvwrapper_lazy

# Create test virtualenv (if this doesn't work, try sourcing ~/.zshrc or opening new shell)
mkvirtualenv venv_test
which python
python -V

# Deactivate & remove test virtualenv
deactivate
rmvirtualenv venv_test
```


## 3. Get the repo & initialise the repo environment

⚠️ N.B. You should replace `REPO_GIT_URL` here with your actual URL to your GitHub repo.

```sh
git clone ${REPO_GIT_URL}
pyenv shell $(cat .python-version)

# Make a new virtual environment using the Python version & environment name specified in the repo
mkvirtualenv -p python$(cat .python-version) $(cat .venv)
python -V  # check this is the correct version of Python
python -m pip install --upgrade pip
```


## 4. Install Python requirements into the virtual environment using [Poetry](https://python-poetry.org/docs/)

Install Poetry onto your system by following the instructions here: [https://python-poetry.org/docs/]

Note that Poetry "lives" outside of project/environment, and if you follow the recommended install
process it will be installed isolated from the rest of your system.

```sh
# Update Poetry regularly as you would any other system-level tool. Poetry is environment agnostic,
# it doesn't matter if you run this command inside/outside the virtualenv.
poetry self update

# This command should be run inside the virtualenv.
poetry install --no-root --sync
```


## 5. [zsh-autoswitch-virtualenv](https://github.com/MichaelAquilina/zsh-autoswitch-virtualenv)

### zsh

Download with `git clone "https://github.com/MichaelAquilina/zsh-autoswitch-virtualenv.git" "$ZSH_CUSTOM/plugins/autoswitch_virtualenv"`

Configure by adding the following to your `~/.zshrc` or equivalent:

```sh
# Find line containing plugins=(git) and replace with below
plugins=(git autoswitch_virtualenv)
```

### bash

There is no nice equivalent to `zsh-autoswitch-virtualenv` for bash.
I (LL) created my own hacky little function to reproduce the same behaviour.

Add these lines to your `~/.bashrc` file:

```sh
cd() {
    builtin cd "$@"  # Call the original cd command
    activate_virtualenv_if_exists  # Activate virtual environment if .venv exists
}

activate_virtualenv_if_exists() {
    local venv_file=".venv"
    local pyenv_file=".python-version"

    # Check if the .python-version file exists and activate the pyenv shell if yes
    if [ -f "$pyenv_file" ]; then
        pyenv shell $(cat "$pyenv_file")
    fi


    # Check if .venv file exists in the directory
    if [ -f "$venv_file" ]; then
        # Check if not already in the same virtual environment
        if [ -z "$CURRENT_VIRTUAL_ENV" ]; then
            workon "$(cat $venv_file)"  # Activate virtual environment
            export CURRENT_VIRTUAL_ENV=$(pwd)
        fi
    else
        # Deactivate virtual environment if leaving a directory with .venv file
        parentdir="$(dirname "$CURRENT_VIRTUAL_ENV")"
        if [[ "$PWD" != "$parentdir"/* ]] ; then
          deactivate
          unset CURRENT_VIRTUAL_ENV
        fi
    fi
}
```

This will overload you `cd` command with the addition, that it checks for the existence of a `.venv` and a `.python-version` file.
If found, it will try to activate the corresponding pyenv version and virtualenv.
When you cd out of the directory, it will deactivate the virtualenv.
It works for my setup, but it's possible that you may experience some problems with it.


Check it's working by cd-ing into & out of the repo. The environment should load & unload respectively.


## 6. Install [Poetry Up](https://github.com/MousaZeidBaker/poetry-plugin-up)

This is a useful Poetry plugin that updates dependencies and bumps their versions in the
pyproject.toml file. The version constraints are respected, unless the `--latest` flag is passed, in
which case dependencies are updated to latest available compatible versions.

```sh
# Installation
poetry self add poetry-plugin-up

# Usage
poetry up
poetry up --latest
```


## 7. Add secrets into .env

  - Run `cp .env.template .env` and update the secrets.
  - [Install direnv](https://direnv.net/) to autoload environment variables specified in `.env`
  - Run `direnv allow` to authorise direnv to load the secrets from `.env` into the environment
    (these will unload when you `cd` out of the repo; note that you will need to re-run this
    command whenever you change `.env`)


## 8. Initialise the `detect-secrets` pre-commit hook

We use [`detect-secrets`](https://github.com/Yelp/detect-secrets) to check that no secrets are
accidentally committed. Please read [docs/detect_secrets.md](docs/detect_secrets.md) for more information.


```shell
# Generate a baseline
detect-secrets scan > .secrets.baseline

# You may want to check/amend the exclusions in `.pre-commit-config.yaml` e.g.
detect-secrets --verbose scan \
    --exclude-files 'poetry\.lock' \
    --exclude-files '\.secrets\.baseline' \
    --exclude-files '\.env\.template' \
    --exclude-secrets 'password|ENTER_PASSWORD_HERE|INSERT_API_KEY_HERE' \
    --exclude-lines 'integrity=*sha' \
    > .secrets.baseline

# Audit the generated baseline
detect-secrets audit .secrets.baseline
```

When you run this command, you'll enter an interactive console. This will present you with a list
of high-entropy string and/or anything which could be a secret. It will then ask you to verify
whether this is the case. This allows the hook to remember false positives in the future, and alert
you to new secrets.


## 9. Project-specific setup

Please check [docs/project_specific_setup.md](docs/project_specific_setup.md) for further instructions.
