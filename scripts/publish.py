import os
import json

import requests
from loguru import logger

MainDir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


class Config:
    Endpoint = None
    HeaderAuthName = None
    HeaderAuthValue = None
    ModelDistDir = os.path.join(MainDir, "dist/api/v1")
    ModelJsonPath = "models.json"
    ModelDetailPath = "models"


def load_config():
    Config.Endpoint = os.environ["ENDPOINT"]
    Config.HeaderAuthName = os.environ["HEADERAUTHNAME"]
    Config.HeaderAuthValue = os.environ["HEADERAUTHVALUE"]


def load_models():
    info = json.loads(open(os.path.join(Config.ModelDistDir, Config.ModelJsonPath), "rb").read())
    tags = [
        {
            "key": key,
            "value": info["tags"][key],
        } for key in info["tags"]
    ]

    model_ids = [
        _["id"] for _ in info["models"]
    ]
    return model_ids, tags


def load_model_details():
    models_ids, tags = load_models()
    dir = os.path.join(Config.ModelDistDir, Config.ModelDetailPath)
    models = []
    for model_id in models_ids:
        models.append(json.loads(open(os.path.join(dir, f"{model_id}.json"), "rb").read()))

    return models, tags


def sync_server():
    models, tags = load_model_details()
    headers = {
        Config.HeaderAuthName: Config.HeaderAuthValue,
    }

    for model in models:
        logger.info(f"model: {json.dumps(model)}")
        resp = requests.post(f"{Config.Endpoint}/api/v1/models", json=model, headers=headers)
        logger.info(f"resp.code: {resp.status_code}, resp.content: {resp.content}")

    for tag in tags:
        logger.info(f"tag: {json.dumps(tag)}")
        resp = requests.post(f"{Config.Endpoint}/api/v1/tags", json=tag, headers=headers)
        logger.info(f"resp.code: {resp.status_code}, resp.content: {resp.content}")


if __name__ == "__main__":
    load_config()
    sync_server()
