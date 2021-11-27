init-airflow:
	echo "AIRFLOW_UID=$$(id -u)" > .env
	docker-compose -f docker-compose_airflow.yaml build
	docker-compose -f docker-compose_airflow.yaml up airflow-init

start-airflow:
	docker-compose -f docker-compose_airflow.yaml up -d

stop-airflow:
	docker-compose -f docker-compose_airflow.yaml down

init-pipenv:
	pipenv sync --dev

unit-tests:
	pipenv run pytest tests --pdb

generate-dags:
	docker-compose -f docker-compose_airflow.yaml run --rm airflow-cli python scripts/generate_dags.py

follow-logs:
	docker-compose -f docker-compose_airflow.yaml logs -f

run-jupyter:
	pipenv run jupyter-notebook

lint:
	pipenv run pylint --rcfile=setup.cfg categories_classification

clean-up:
	docker-compose -f docker-compose_airflow.yaml down && docker-compose -f docker-compose_airflow.yaml rm -v
