#!/usr/bin/env python3
import os

import aws_cdk as cdk
from aws_cdk import Aspects
from ecs_fargate_app.permission_boundary_aspect import PermissionBoundaryAspect

from ecs_fargate_app.wa_genai_stack import WAGenAIStack

app = cdk.App()

# Try to get the region from an environment variable
REGION = os.environ.get("CDK_DEPLOY_REGION")

# If REGION is still None, it will use the default region when deployed
env = cdk.Environment(region=REGION) if REGION else None

APP_PREFIX = f"WA-IaC-Analyzer-{REGION or 'default'}"

# Create the front-end Stack
wa_genai_stack = WAGenAIStack(
    app,
    f"{APP_PREFIX}-GenAIStack",
    env=env,
)

app.synth()
