.PHONY: data
.PHONY: create-env

create-env:
		python -m venv env
		source env/bin/activate
		pip install -r requirements.txt

get-data:
		mkdir -p data && cd data && \
		kaggle datasets download -d mkechinov/ecommerce-events-history-in-electronics-store && \
		unzip *.zip && rm *.zip
		echo "Downloaded and inflated dataset successfully"
