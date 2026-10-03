"""AWS Lambda entrypoint. Loads the LLM provider key from SSM Parameter Store once per cold start,
then delegates to the Mangum handler. The key never lives in an environment variable."""

import os

import boto3

_param = os.environ.get("LLM_KEY_SSM_PARAM")
_key_env = os.environ.get("LLM_KEY_ENV", "ANTHROPIC_API_KEY")
if _param and not os.environ.get(_key_env):
    os.environ[_key_env] = boto3.client("ssm").get_parameter(Name=_param, WithDecryption=True)[
        "Parameter"
    ]["Value"]

from .api import handler  # noqa: F401
