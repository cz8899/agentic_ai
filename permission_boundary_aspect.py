from aws_cdk import Aspects, aws_iam as iam
from aws_cdk import IAspect
from constructs import IConstruct


class PermissionBoundaryAspect(IAspect):
    def __init__(self, boundary_arn: str) -> None:
        self.boundary_arn = boundary_arn

    def visit(self, node: IConstruct) -> None:
        # Apply PermissionsBoundary by patching the low-level CfnRole properties
        if isinstance(node, iam.Role):
            cfn = node.node.default_child
            try:
                # If default child is a CfnRole, add the PermissionsBoundary property
                if hasattr(cfn, "add_property_override"):
                    cfn.add_property_override("PermissionsBoundary", self.boundary_arn)
            except Exception as e:
                # best-effort; avoid failing synth if we can't patch a role
                print(f"PermissionBoundaryAspect: failed to apply to {node}: {e}")
