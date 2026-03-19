.PHONY: help venv download_dataset stage ingest

help:
	@echo "📋 Asthma Detection Dataset - Snowflake Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make venv              - Create virtual environment and install dependencies"
	@echo "  make download_dataset  - Download and prepare the asthma detection dataset"
	@echo "  make stage             - Transfer dataset to Snowflake stage"
	@echo "  make ingest            - Ingest audio metadata to Snowflake table"
	@echo ""

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

stage:
	@echo "🚀 Transferring audio files to Snowflake stage..."
	@python -m backend.db.stage.stg_respiratory_sounds
	@echo "✅ Files transferred successfully!"
	@echo "🚀 Extract audio..."
	@python -m backend.db.table.respiratory_sounds_metadata
	@echo "✅ Metadata ingested successfully!"

