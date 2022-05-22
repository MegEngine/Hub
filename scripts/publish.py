import os
import json
from pathlib import Path

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


def sync_server():
    filelist = [Config.ModelJsonPath]
    filelist.extend(
        map(
            lambda fname: os.path.join(Config.ModelDetailPath, fname),
            os.listdir(
                os.path.join(Config.ModelDistDir, Config.ModelDetailPath))))

    headers = {Config.HeaderAuthName: Config.HeaderAuthValue}
    for fname in filelist:
        data = Path(os.path.join(Config.ModelDistDir, fname)).read_text()
        resp = requests.put(f"{Config.Endpoint}{fname}",
                            headers=headers,
                            json=json.loads(data))
        logger.info(
            f"resp.code: {resp.status_code}, resp.content: {resp.content}")


if __name__ == "__main__":
    load_config()
    sync_server()
