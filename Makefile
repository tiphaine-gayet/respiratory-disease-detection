.PHONY: help venv download_dataset raw app ingested infra streamlit

help:
	@echo "📋 Asthma Detection Dataset - Snowflake Commands"
	@echo ""
	@echo "Setup local environment:"
	@echo "  make venv              - Create virtual environment and install dependencies"
	@echo "  make download_dataset  - Download and prepare the asthma detection dataset"
	@echo ""
	@echo "Setup Snowflake environment:"
	@echo "  make infra             - Run all infrastructure setup steps (raw, ingested, app)"
	@echo "  make raw               - Transfer raw audio files and metadata to Snowflake"
	@echo "  make app               - Set up predictions table and load data used in app (pharmacies in France) into Snowflake"
	@echo "  make ingested          - Create INGESTED schema (ingested audio files, extracted features, and metadata)"
	@echo ""
	@echo "Run Streamlit app:"
	@echo "  make streamlit          - Start the Streamlit app"

venv:
	python3.11 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

download_dataset:
	@echo "📥 Downloading dataset from Kaggle..."
	@kaggle datasets download -d mohammedtawfikmusaed/asthma-detection-dataset-version-2
	@echo "📂 Unzipping dataset..."
	@unzip asthma-detection-dataset-version-2.zip -d asthma_detection_dataset
	@mkdir asthma_detection_dataset/audio/
	@mv "asthma_detection_dataset/Asthma Detection Dataset Version 2/Asthma Detection Dataset Version 2/"* asthma_detection_dataset/audio/
	@rm -rf "asthma_detection_dataset/Asthma Detection Dataset Version 2/"
	@rm asthma-detection-dataset-version-2.zip
	@echo "✅ Dataset downloaded and extracted to 'asthma_detection_dataset/'"

raw:
	@echo "🚀 Transferring raw audio files to Snowflake stage..."
	@python -m backend.infra.raw.stg_respiratory_sounds
	@echo "✅ Files transferred successfully!"
	@echo "🚀 Extract audio metadata..."
	@python -m backend.infra.raw.table_respiratory_sounds_metadata
	@echo "✅ Metadata ingested successfully!"

app:
	@echo "🚀 Deploying APP schema resources..."
	@python -m backend.infra.app.load_pharmacies_france
	@python -m backend.infra.app.table_predictions
	@echo "✅ Done!"

ingested:
	@echo "🚀 Deploying INGESTED schema resources..."
	@python -m backend.infra.ingested.stg_ingested_sounds
	@python -m backend.infra.ingested.stg_processed_sounds
	@python -m backend.infra.ingested.table_ingested_sounds_metadata
	@python -m backend.infra.ingested.table_processed_sounds_metadata
	@echo "✅ Done!"
	
infra:
	@python -m backend.infra.schemas
	@make raw
	@make ingested
	@make app

streamlit:
	@echo "🚀 Starting Streamlit app..."
	@streamlit run frontend/app.py
