This repository implements an AI-based chatbot to answer questions about the Vestibular Unicamp 2025.

# Instructions for ChatBot Installation

## Clone repository

```sh
git clone git@github.com:jovi2000/NeuralMind_Summer.git
```

## Python Version:

It is recommended to use Python 3 in version 3.13.0.

### How to install using pyenv:

1. **Install pyenv**:
   - On Ubuntu/Debian:
     ```sh
     curl https://pyenv.run | bash
     ```

   - On macOS (using Homebrew):
     ```sh
     brew install pyenv
     ```

2. **Update your shell configuration**:

   ```sh
   export PATH="$HOME/.pyenv/bin:$PATH"
   eval "$(pyenv init --path)"
   eval "$(pyenv virtualenv-init -)"
   ```
   Reload the configuration with:
   ```sh
   source ~/.bashrc
   ```

3. **Install Python 3.13.0**:
   ```sh
   pyenv install 3.13.0
   ```

4. **Set Python 3.13.0 as the global version**:
   ```sh
   pyenv global 3.13.0
   ```

## Install Dependencies

```sh
pip install -r requirements.txt
```

## Set OpenAI Environment Key

As the Chatbot is using the `gpt-4o-mini` model and `OpenAIEmbeddings`, the use of the OpenAI key is required.

```sh
export OPENAI_API_KEY="{your_key}"
```

# Instructions for ChatBot Use

## Run

```sh
python3 chatbot.py
```

After the terminal displays "Digite sua dúvida:", type your question related to Vestibular Unicamp 2025.

## Stop

To stop the Chatbot, press `Ctrl+C`.

