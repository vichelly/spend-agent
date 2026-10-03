"""AWS Lambda entrypoint. Loads the Anthropic key from SSM Parameter Store once per cold start,
then delegates to the Mangum handler. The key never lives in an environment variable."""

import os

import boto3

_param = os.environ.get("ANTHROPIC_KEY_SSM_PARAM")
if _param and not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = boto3.client("ssm").get_parameter(
        Name=_param, WithDecryption=True
    )["Parameter"]["Value"]

from .api import handler  # noqa: F401
