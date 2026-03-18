.PHONY: venv

help:
	@echo "📋 Asthma Detection Dataset - Snowflake Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make venv              - Create virtual environment and install dependencies"
	@echo ""

venv:
	python3.11 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
