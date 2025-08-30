# ecs_fargate_app/permission_boundary_aspect.py
from aws_cdk import IAspect
from aws_cdk import aws_iam as iam
from constructs import IConstruct

class PermissionBoundaryAspect(IAspect):
    def __init__(self, boundary_arn: str) -> None:
        self.boundary_arn = boundary_arn

    def visit(self, node: IConstruct) -> None:
        # when encountering high-level iam.Role L2, patch its default child
        if isinstance(node, iam.Role):
            cfn = node.node.default_child
            if cfn and hasattr(cfn, "add_property_override"):
                cfn.add_property_override("PermissionsBoundary", self.boundary_arn)
