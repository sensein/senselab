# Data Model: AWS GPU Test Orchestration Setup

## Entities

### GitHub Secrets (per repository)

| Name | Source | Purpose |
|------|--------|---------|
| `AWS_KEY_ID` | IAM user access key | EC2 RunInstances/TerminateInstances |
| `AWS_KEY_SECRET` | IAM user secret key | Paired with AWS_KEY_ID |
| `GH_TOKEN` | GitHub PAT (repo scope) | Register self-hosted runner |

### GitHub Variables (per repository)

| Name | Example | Purpose |
|------|---------|---------|
| `AWS_REGION` | `us-east-1` | Region for all EC2 operations |
| `AWS_IMAGE_ID` | `ami-0abc123` | Pre-built AMI with GPU drivers + venv |
| `AWS_INSTANCE_TYPE` | `g4dn.xlarge` | Default GPU instance type |
| `AWS_SUBNET` | `subnet-0abc123` | Primary subnet (must have internet) |
| `AWS_SECURITY_GROUP` | `sg-0abc123` | Outbound HTTPS only |
| `AWS_AZ_CONFIG` | JSON array | Multi-AZ failover configuration |
| `WORKING_DIR` | `/tmp/senselab` | Cache directory on instance |

### AWS Resources (per AWS account, shared across repos)

| Resource | Purpose | Created By |
|----------|---------|------------|
| IAM User | EC2 API access for CI | Setup script |
| IAM Policy | Minimum EC2 permissions | Setup script |
| Security Group | Outbound HTTPS for runner registration | Setup script (or reuse existing) |
| AMI | Pre-built image with GPU stack | Manual (documented process) |

### AZ Configuration Entry

```json
{
  "imageId": "ami-xxx",
  "subnetId": "subnet-xxx",
  "securityGroupId": "sg-xxx"
}
```

Multiple entries form the `AWS_AZ_CONFIG` array for failover.

### PR Labels (for GPU instance selection)

| Label | Effect | Priority |
|-------|--------|----------|
| `to-test` | Triggers GPU test workflow | Required |
| `gpu-instance:<type>` | Override exact instance type | Highest |
| `gpu-multi` | Select multi-GPU instance (g5.12xlarge) | Medium |
| `gpu-family:<family>` | Select family default (g4dn, g5, g6, p3, p4d, p5) | Low |
| `gpu-ondemand:true` | Use on-demand instead of spot | Modifier |

### Setup Script Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--profile` | Yes | — | AWS CLI profile name |
| `--repo` | Yes | — | GitHub repo (owner/repo) |
| `--region` | No | us-east-1 | AWS region |
| `--instance-type` | No | g4dn.xlarge | Default GPU instance |
| `--ami` | Yes | — | Pre-built AMI ID |
| `--vpc` | No | default VPC | VPC for security group |
