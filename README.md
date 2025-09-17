# Inspect AI examples
Some examples of using [Inspect AI](https://inspect.aisi.org.uk/) to evaluate task performance:
* [Binary classification](./classifier.py)
* [Intent classifier](./intent_classifier.py)

## Project Structure
```
├── data/                    # Evaluate datasets
├── logs/                    # Evaluation results and logs
├── classifier.py            # Binary classification evaluation
├── intent_classifier.py     # Intent classification evaluation
├── .env                     # API configuration
└── pyproject.toml           # Project dependencies
```

## Setup

### Requirements
- Tested on Python 3.13
- uv package manager

### Installation
1. Install dependencies:
   ```bash
   uv sync
   ```

### Configuration
1. Create a `.env` file in the project root
2. Add your API configuration:
   ```bash
   export BEDROCK_API_KEY=<bedrock_api_key>
   export BEDROCK_BASE_URL=https://bedrock-runtime.<aws_region>.amazonaws.com/openai/v1
   ```

## Evaluation
1. Source environment variables:
   ```bash
   source .env
   ```

2. Run an evaluation:
   ```bash
   python <evaluation_file.py>
   ```

3. View results:
    ```bash
    inspect view
    ```