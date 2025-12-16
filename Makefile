# This file lets us run complex commands with simple shortcuts.

# 1. Setup the environment
install:
	pip install -e .
	pip install -r requirements.txt

# 2. formatting code (optional but good for teams)
format:
	black src/

# 3. Clean up junk files (pycache, etc)
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# 4. Run the Training (Baseline)
train-mcq:
	python -m src.train --config configs/mcq_baseline.yaml

# 5. Run the Prediction
predict-mcq:
	python -m src.predict --task mcq --checkpoint artifacts/latest_model
