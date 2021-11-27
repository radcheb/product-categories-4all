import logging
import os

import yaml
from jinja2 import Template

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


TEMPLATES_DIR = './dags_templates'
CONFIG_FILE = os.getenv("CONFIG_FILE", "conf/clients_config.yaml")
TEMPLATES = ["training", "prediction"]
GENERATED_DAGS_DIR = "dags"

with open(CONFIG_FILE, 'r') as yaml_file:
    clients_config = yaml.load(yaml_file, Loader=yaml.FullLoader)

for dag_template_name in TEMPLATES:
    with open(os.path.join(TEMPLATES_DIR, f'{dag_template_name}.py'), 'r') as template:
        dag_template = Template(template.read())
    for client_id, client_data in clients_config.items():
        LOGGER.info(f"Generating {dag_template_name} DAG for client {client_id}")
        dag_payload: str = dag_template.render(client_id=client_id, **client_data)
        with open(os.path.join(GENERATED_DAGS_DIR, f'{client_id}_{dag_template_name}.py'), 'w') as dag:
            dag.write(dag_payload)
