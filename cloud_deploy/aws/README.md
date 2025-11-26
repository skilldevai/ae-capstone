# AWS Deployment Guide

## Option 1: AWS App Runner (Recommended - Simplest)

App Runner is the easiest way to deploy containers on AWS. It handles
load balancing, auto-scaling, and HTTPS automatically.

### Prerequisites
- AWS CLI installed and configured
- Docker installed locally
- AWS account with ECR access

### Steps

1. **Create ECR Repository**
   ```bash
   aws ecr create-repository --repository-name omnitech-support
   ```

2. **Build and Push Docker Image**
   ```bash
   # Get ECR login
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com

   # Build image
   docker build -t omnitech-support .

   # Tag for ECR
   docker tag omnitech-support:latest <account>.dkr.ecr.<region>.amazonaws.com/omnitech-support:latest

   # Push to ECR
   docker push <account>.dkr.ecr.<region>.amazonaws.com/omnitech-support:latest
   ```

3. **Store HF_TOKEN in Secrets Manager**
   ```bash
   aws secretsmanager create-secret \
     --name hf-token \
     --secret-string '{"HF_TOKEN":"your_huggingface_token"}'
   ```

4. **Deploy with App Runner**
   ```bash
   # Edit apprunner.yaml with your ECR image URI and secret ARN
   aws apprunner create-service --cli-input-yaml file://apprunner.yaml
   ```

5. **Get the URL**
   ```bash
   aws apprunner describe-service --service-arn <service-arn> --query 'Service.ServiceUrl'
   ```

### Estimated Cost
- App Runner: ~$5-15/month for light usage
- ECR: ~$0.10/GB/month for storage

---

## Option 2: Amazon ECS with Fargate

For more control over networking and scaling.

### Steps

1. **Create ECS Cluster**
   ```bash
   aws ecs create-cluster --cluster-name omnitech-cluster
   ```

2. **Create Task Definition**
   ```bash
   # Edit ecs-task-definition.json with your values
   aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
   ```

3. **Create Service**
   ```bash
   aws ecs create-service \
     --cluster omnitech-cluster \
     --service-name omnitech-support \
     --task-definition omnitech-support \
     --desired-count 1 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
   ```

4. **Set up Application Load Balancer** (for HTTPS)
   - Create ALB in AWS Console
   - Add target group pointing to ECS service
   - Configure HTTPS listener with ACM certificate

---

## Option 3: Elastic Beanstalk

For a more traditional PaaS experience.

1. Install EB CLI: `pip install awsebcli`
2. Initialize: `eb init -p docker omnitech-support`
3. Create environment: `eb create omnitech-env`
4. Set environment variable: `eb setenv HF_TOKEN=your_token`
5. Deploy: `eb deploy`

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| HF_TOKEN | HuggingFace API token | Yes |

## Troubleshooting

- **Container won't start**: Check CloudWatch logs
- **Health check failing**: Ensure port 7860 is exposed and app starts within 60s
- **Out of memory**: Increase task memory in task definition
