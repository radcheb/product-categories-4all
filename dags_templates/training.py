from datetime import datetime, timedelta
from textwrap import dedent

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.bash import BashOperator

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
CLIENT_ID = "{{ client_id }}"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['radcheb@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}
with DAG(
        f'{CLIENT_ID}-training',
        default_args=default_args,
        description=f'Training DAG for client {CLIENT_ID}',
        schedule_interval=timedelta(days=30),
        start_date=datetime(2021, 1, 1),
        catchup=False,
        tags=['training', CLIENT_ID],
) as dag:
    # Trainer command to be run in pipenv env
    command = f"""pipenv run python categories_classification_cli.py trainer --client_id={CLIENT_ID} """ \
              """--features='{{ features|tojson|safe }}' """ \
              """{% raw %}--training_date='{{ ts }}' {% endraw %}""" \
              """--model_params='{{ params|tojson|safe }}'"""
    trainer_task = BashOperator(
        task_id='run_trainer',
        bash_command=command,
        cwd="/opt/better_categories_4all"
    )
