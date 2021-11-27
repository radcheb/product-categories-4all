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
        f'{CLIENT_ID}-prediction',
        default_args=default_args,
        description=f'Top categories prediction DAG for client {CLIENT_ID}',
        schedule_interval=timedelta(days=1),
        start_date=datetime(2021, 1, 1),
        catchup=False,
        tags=['prediction', CLIENT_ID],
) as dag:
    # Trainer command to be run in pipenv env
    command = f"""pipenv run python categories_classification_cli.py predictor --client_id={CLIENT_ID} """ \
              """{% raw %}--inference_date='{{ ts }}' {% endraw %}"""

    trainer_task = BashOperator(
        task_id='run_predictor',
        bash_command=command,
        cwd="/opt/better_categories_4all"
    )
