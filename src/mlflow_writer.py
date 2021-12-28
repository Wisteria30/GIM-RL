# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import json
import mlflow
from mlflow import pytorch
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, ListConfig
import requests
import yaml


def parse_override_yaml(overrides_path):
    with open(overrides_path) as f:
        override_params = yaml.safe_load(f)
    override_params = {op.split("=")[0]: op.split("=")[1] for op in override_params}
    # delete content, user, commit
    _ = override_params.pop("content", None)
    _ = override_params.pop("user", None)
    _ = override_params.pop("commit", None)
    return override_params


class MlflowWriter:
    def __init__(self, experiment_name, uri, **kwargs):
        self.client = MlflowClient(tracking_uri=uri, **kwargs)
        mlflow.set_tracking_uri(uri)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except BaseException:
            self.experiment_id = self.client.get_experiment_by_name(
                experiment_name
            ).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f"{parent_name}.{k}", v)
                else:
                    self.client.log_param(self.run_id, f"{parent_name}.{k}", v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f"{parent_name}.{i}", v)
        else:
            self.client.log_param(self.run_id, f"{parent_name}", element)

    def log_torch_model(self, model):
        with mlflow.start_run(self.run_id):
            pytorch.log_model(model, "models")

    def load_torch_model(self, uri):
        with mlflow.start_run(self.run_id):
            pytorch.load_model(uri)

    def set_tag(self, key, value):
        self.client.set_tag(self.run_id, key, value)

    def set_runname(self, overrides_path):
        override_params = parse_override_yaml(overrides_path)
        run_name = ",".join(["{}:{}".format(k, v) for k, v in override_params.items()])
        self.client.set_tag(self.run_id, "mlflow.runName", run_name)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, **kwargs):
        self.client.log_metric(self.run_id, key, value, **kwargs)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)

    def post_slack(self, type, overrides_path):
        """
        Send a notification to slack when finished
        """
        try:
            with open("/work/secret.json") as f:
                secret = json.load(f)
        except FileNotFoundError as e:
            print(e)
            return

        run_result_dict = self.client.get_run(self.run_id).to_dictionary()
        override_params = parse_override_yaml(overrides_path)

        pretty_result_dict = {
            "info": {
                "run_id": run_result_dict["info"]["run_id"],
                "type": type,
                "status": run_result_dict["info"]["status"],
                "start_time": str(
                    datetime.fromtimestamp(run_result_dict["info"]["start_time"] / 1000)
                ),
                "endtime": str(
                    datetime.fromtimestamp(run_result_dict["info"]["end_time"] / 1000)
                ),
                "exe_time": str(
                    timedelta(
                        milliseconds=run_result_dict["info"]["end_time"]
                        - run_result_dict["info"]["start_time"]
                    )
                ),
                "dataset": run_result_dict["data"]["params"][
                    f"{type}.file_path"
                ]
                .replace("data/", "")
                .replace(".txt", ""),
            },
            "override_params": override_params,
            "metrics": run_result_dict["data"]["metrics"],
        }
        run_result_str = json.dumps(pretty_result_dict, indent=4)

        requests.post(
            secret["slack_webhook_uri"], data=json.dumps({"text": run_result_str})
        )
