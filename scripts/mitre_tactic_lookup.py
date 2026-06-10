#!/usr/bin/env python3
"""Shared MITRE ATT&CK tactic lookup for AttackLM extraction scripts.

Maps MITRE technique IDs (e.g. "T1059.001") to their kill chain tactic
(e.g. "TA0002" Execution). Used by all 4 extraction scripts to consistently
tag training pairs with kill chain phase metadata.

Usage:
    from mitre_tactic_lookup import get_tactic_for_technique, get_tactic_name

    tactic_id = get_tactic_for_technique("T1059.001")  # "TA0002"
    name = get_tactic_name("TA0002")                    # "Execution"
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Tactic IDs → display names (the 14 MITRE ATT&CK Enterprise tactics + ATLAS)
# ---------------------------------------------------------------------------
TACTIC_INFO: dict[str, str] = {
    "TA0042": "Resource Development",
    "TA0043": "Reconnaissance",
    "TA0001": "Initial Access",
    "TA0002": "Execution",
    "TA0003": "Persistence",
    "TA0004": "Privilege Escalation",
    "TA0005": "Defense Evasion",
    "TA0006": "Credential Access",
    "TA0007": "Discovery",
    "TA0008": "Lateral Movement",
    "TA0009": "Collection",
    "TA0010": "Exfiltration",
    "TA0011": "Command and Control",
    "TA0012": "Impact",
    "TA0040": "Prompt Injection",  # ATLAS-specific (AI/ML)
}

# Reverse map: lowercase tactic name → tactic ID (for normalizing Caldera YAML)
TACTIC_NAME_TO_ID: dict[str, str] = {
    name.lower().replace(" ", "-").replace("/", "-"): tid
    for tid, name in TACTIC_INFO.items()
}
# Also add common aliases
TACTIC_NAME_TO_ID.update(
    {
        "resource-development": "TA0042",
        "resource_development": "TA0042",
        "reconnaissance": "TA0043",
        "initial-access": "TA0001",
        "initial_access": "TA0001",
        "execution": "TA0002",
        "persistence": "TA0003",
        "privilege-escalation": "TA0004",
        "privilege_escalation": "TA0004",
        "defense-evasion": "TA0005",
        "defense_evasion": "TA0005",
        "credential-access": "TA0006",
        "credential_access": "TA0006",
        "discovery": "TA0007",
        "lateral-movement": "TA0008",
        "lateral_movement": "TA0008",
        "collection": "TA0009",
        "exfiltration": "TA0010",
        "command-and-control": "TA0011",
        "command_and_control": "TA0011",
        "c2": "TA0011",
        "impact": "TA0012",
        "prompt-injection": "TA0040",
        "prompt_injection": "TA0040",
    }
)

# ---------------------------------------------------------------------------
# Technique → Primary tactic mapping
# ---------------------------------------------------------------------------
# Comprehensive mapping covering all techniques found in the 4 extractors
# (RTA, Caldera, Atomic Red Team, Infection Monkey) plus common ATT&CK techniques.
# Sub-techniques are mapped explicitly; parent techniques get a default.

TECHNIQUE_TO_TACTIC: dict[str, str] = {
    # ================================================================
    # Initial Access (TA0001)
    # ================================================================
    "T1078.001": "TA0001",  # Default Accounts
    "T1078.003": "TA0001",  # Local Accounts
    "T1078.004": "TA0001",  # Cloud Accounts
    "T1566.001": "TA0001",  # Spearphishing Attachment
    "T1566.002": "TA0001",  # Spearphishing Link
    "T1566.003": "TA0001",  # Spearphishing via Service
    "T1195": "TA0001",  # Supply Chain Compromise
    "T1195.002": "TA0001",  # Compromise Software Supply Chain
    "T1190": "TA0001",  # Exploit Public-Facing Application
    "T1133": "TA0001",  # External Remote Services
    "T1200": "TA0001",  # Hardware Additions
    "T1091": "TA0001",  # Replication Through Removable Media
    "T1189": "TA0001",  # Drive-by Compromise
    "T1199": "TA0001",  # Trusted Relationship
    "T1078": "TA0001",  # Valid Accounts (also Persistence/PrivEsc)
    # ================================================================
    # Execution (TA0002)
    # ================================================================
    "T1059": "TA0002",  # Command and Scripting Interpreter
    "T1059.001": "TA0002",  # PowerShell
    "T1059.002": "TA0002",  # AppleScript
    "T1059.003": "TA0002",  # Windows Command Shell
    "T1059.004": "TA0002",  # Unix Shell
    "T1059.005": "TA0002",  # Visual Basic
    "T1059.006": "TA0002",  # Python
    "T1059.007": "TA0002",  # JavaScript
    "T1059.010": "TA0002",  # AutoHotKey
    "T1053": "TA0002",  # Scheduled Task/Job (also Persistence)
    "T1053.002": "TA0002",  # At
    "T1053.003": "TA0002",  # Cron
    "T1053.005": "TA0002",  # Scheduled Task
    "T1053.006": "TA0002",  # Systemd Timers
    "T1053.007": "TA0002",  # Container Orchestration Job
    "T1204": "TA0002",  # User Execution
    "T1204.002": "TA0002",  # Malicious File
    "T1204.003": "TA0002",  # Malicious Link
    "T1035": "TA0002",  # Service Execution
    "T1121": "TA0002",  # Component Object Model and Distributed COM
    "T1559": "TA0002",  # Inter-Process Communication (also Persistence)
    "T1559.002": "TA0002",  # Dynamic Data Exchange
    "T1609": "TA0002",  # Container Administration Command
    "T1610": "TA0002",  # Deploy Container
    "T1044": "TA0002",  # Execution through Module Load (legacy)
    "T1064": "TA0002",  # Scripting (legacy)
    "T1085": "TA0002",  # Rundll32 (legacy)
    "T1086": "TA0002",  # PowerShell (legacy)
    "T1117": "TA0002",  # Regsvr32 (legacy)
    "T1118": "TA0002",  # InstallUtil (legacy)
    "T1124": "TA0002",  # Component Firmware (legacy → Discovery)
    "T1127": "TA0002",  # Trusted Developer Utilities
    "T1127.001": "TA0002",  # LC_LOAD_DYLIB (legacy)
    "T1170": "TA0002",  # Mshta (legacy)
    "T1223": "TA0002",  # Content Injection (legacy)
    "T1154": "TA0002",  # Trap (legacy)
    # ================================================================
    # Persistence (TA0003)
    # ================================================================
    "T1543": "TA0003",  # Create or Modify System Process
    "T1543.001": "TA0003",  # Launch Agent
    "T1543.002": "TA0003",  # Systemd Service
    "T1543.003": "TA0003",  # Windows Service
    "T1543.004": "TA0003",  # Launch Daemon
    "T1547": "TA0003",  # Boot or Logon Autostart Execution
    "T1547.001": "TA0003",  # Registry Run Keys
    "T1547.002": "TA0003",  # Authentication Package
    "T1547.003": "TA0003",  # Time Providers
    "T1547.004": "TA0003",  # Winlogon Helper DLL
    "T1547.005": "TA0003",  # Security Support Provider
    "T1547.006": "TA0003",  # Kernel Modules and Extensions
    "T1547.007": "TA0003",  # Re-opened Applications
    "T1547.008": "TA0003",  # LSASS Driver
    "T1547.009": "TA0003",  # Shortcut Modification
    "T1547.010": "TA0003",  # Port Monitors
    "T1547.012": "TA0003",  # Print Processors
    "T1547.014": "TA0003",  # Active Setup
    "T1547.015": "TA0003",  # Login Items
    "T1136": "TA0003",  # Create Account
    "T1136.001": "TA0003",  # Local Account
    "T1136.002": "TA0003",  # Domain Account
    "T1136.003": "TA0003",  # Cloud Account
    "T1137": "TA0003",  # Office Application Startup
    "T1137.001": "TA0003",  # Office Template Macros
    "T1137.002": "TA0003",  # Office Test
    "T1137.004": "TA0003",  # Outlook Home Page
    "T1137.005": "TA0003",  # Outlook Rules
    "T1137.006": "TA0003",  # Add-ins
    "T1546": "TA0003",  # Event Triggered Execution
    "T1546.001": "TA0003",  # .bash_profile/.bashrc
    "T1546.002": "TA0003",  # .profile
    "T1546.003": "TA0003",  # Windows Management Instrumentation Event Subscription
    "T1546.004": "TA0003",  # Unix Shell Configuration Modification
    "T1546.005": "TA0003",  # .trap
    "T1546.007": "TA0003",  # Netsh Helper DLL
    "T1546.008": "TA0003",  # Accessibility Features
    "T1546.009": "TA0003",  # AppCert DLLs
    "T1546.010": "TA0003",  # AppInit DLLs
    "T1546.011": "TA0003",  # Application Shimming
    "T1546.012": "TA0003",  # Image File Execution Options Injection
    "T1546.013": "TA0003",  # PowerShell Profile
    "T1546.014": "TA0003",  # Emond
    "T1546.015": "TA0003",  # Component Object Model Hijacking
    "T1546.018": "TA0003",  # Installer Packages
    "T1574": "TA0003",  # Hijack Execution Flow
    "T1574.001": "TA0003",  # DLL Search Order Hijacking
    "T1574.006": "TA0003",  # LD_PRELOAD
    "T1574.008": "TA0003",  # Path Interception by PATH Environment Variable
    "T1574.009": "TA0003",  # Path Interception by Unquoted Path
    "T1574.010": "TA0003",  # Services File Permissions Weakness
    "T1574.011": "TA0003",  # Services Registry Permissions Weakness
    "T1574.012": "TA0003",  # COR Profilers
    "T1158": "TA0003",  # Hidden Files and Directories
    "T1168": "TA0003",  # Local Job Scheduling (legacy)
    "T1015": "TA0003",  # Accessibility Features (legacy)
    "T1037": "TA0003",  # Boot or Logon Initialization Scripts
    "T1037.001": "TA0003",  # Logon Scripts
    "T1037.004": "TA0003",  # rc.files
    "T1037.005": "TA0003",  # Startup Items
    "T1103": "TA0003",  # AppInit DLLs (legacy)
    "T1122": "TA0003",  # Component Object Model Hijacking (legacy)
    "T1126": "TA0003",  # Network Share Connection Removal (legacy → Defense Evasion)
    "T1160": "TA0003",  # LC_LOAD_DYLIB (legacy)
    "T1161": "TA0003",  # LC_LOAD_DYLIB Addition (legacy)
    "T1162": "TA0003",  # Systemd Service (legacy)
    "T1169": "TA0003",  # Sudo (legacy → Privilege Escalation)
    "T1176": "TA0003",  # Browser Extensions (legacy)
    # ================================================================
    # Privilege Escalation (TA0004)
    # ================================================================
    "T1548": "TA0004",  # Abuse Elevation Control Mechanism
    "T1548.001": "TA0004",  # Setuid and Setgid
    "T1548.002": "TA0004",  # Bypass User Account Control
    "T1548.003": "TA0004",  # Sudo and Sudo Caching
    "T1055": "TA0004",  # Process Injection
    "T1055.001": "TA0004",  # Dynamic-link Library Injection
    "T1055.002": "TA0004",  # Portable Executable Injection
    "T1055.003": "TA0004",  # Thread Execution Hijacking
    "T1055.004": "TA0004",  # Asynchronous Procedure Call
    "T1055.009": "TA0004",  # Proc Memory
    "T1055.011": "TA0004",  # Extra Window Memory Injection
    "T1055.012": "TA0004",  # Process Hollowing
    "T1055.015": "TA0004",  # ListPlanting
    "T1068": "TA0004",  # Exploitation for Privilege Escalation
    "T1053": "TA0004",  # Scheduled Task/Job (also Execution/Persistence)
    "T1078": "TA0004",  # Valid Accounts (also Initial Access/Persistence)
    "T1134": "TA0004",  # Access Token Manipulation
    "T1134.001": "TA0004",  # Token Impersonation/Theft
    "T1134.002": "TA0004",  # Create Process with Token
    "T1134.004": "TA0004",  # Parent PID Spoofing
    "T1134.005": "TA0004",  # SID-History Injection
    "T1048": "TA0004",  # Abuse Elevation Control Mechanism (legacy)
    "T1048.002": "TA0004",  # Bypass User Account Control
    "T1048.003": "TA0004",  # Sudo Caching
    "T1098": "TA0004",  # Account Manipulation (also Persistence)
    "T1098.001": "TA0004",  # Additional Email Delegate Permissions
    "T1098.002": "TA0004",  # Exchange Email Delegate Permissions
    "T1098.003": "TA0004",  # Additional Cloud Roles
    "T1098.004": "TA0004",  # SSH Authorized Keys
    "T1505": "TA0004",  # Server Software Component (also Persistence)
    "T1505.002": "TA0004",  # SQL Stored Procedures
    "T1505.003": "TA0004",  # Web Shell
    "T1505.004": "TA0004",  # IIS Components
    "T1505.005": "TA0004",  # Terminal Services DLL
    # ================================================================
    # Defense Evasion (TA0005)
    # ================================================================
    "T1562": "TA0005",  # Impair Defenses
    "T1562.001": "TA0005",  # Disable or Modify Tools
    "T1562.002": "TA0005",  # Disable Windows Event Logging
    "T1562.003": "TA0005",  # HISTCONTROL
    "T1562.004": "TA0005",  # Disable or Modify System Firewall
    "T1562.006": "TA0005",  # Indicator Blocking
    "T1562.008": "TA0005",  # Disable or Modify Cloud Logs
    "T1564": "TA0005",  # Hide Artifacts
    "T1564.001": "TA0005",  # Hidden Files and Directories
    "T1564.002": "TA0005",  # Hidden Users
    "T1564.003": "TA0005",  # Hidden Window
    "T1564.004": "TA0005",  # NTFS Extended Attributes
    "T1564.006": "TA0005",  # Run Virtual Instance
    "T1564.008": "TA0005",  # Email Hiding Rules
    "T1027": "TA0005",  # Obfuscated Files or Information
    "T1027.001": "TA0005",  # Binary Padding
    "T1027.002": "TA0005",  # Software Packing
    "T1027.004": "TA0005",  # Compile After Delivery
    "T1027.006": "TA0005",  # HTML Smuggling
    "T1027.007": "TA0005",  # Dynamic API Resolution
    "T1027.010": "TA0005",  # Command Obfuscation
    "T1027.013": "TA0005",  # Encrypted/Encoded File
    "T1027.018": "TA0005",  # Stripped Payloads
    "T1070": "TA0005",  # Indicator Removal
    "T1070.001": "TA0005",  # Clear Windows Event Logs
    "T1070.003": "TA0005",  # Clear Command History
    "T1070.004": "TA0005",  # File Deletion
    "T1070.005": "TA0005",  # Clear Persistence
    "T1070.006": "TA0005",  # Timestomp
    "T1070.008": "TA0005",  # Clear Linux or Mac System Logs
    "T1036": "TA0005",  # Masquerading
    "T1036.002": "TA0005",  # Linux and Mac Binary Masquerading
    "T1036.003": "TA0005",  # Rename System Utilities
    "T1036.004": "TA0005",  # Masquerade Task or Service
    "T1036.005": "TA0005",  # Match Legitimate Name or Location
    "T1036.006": "TA0005",  # Space After Filename
    "T1036.007": "TA0005",  # Double File Extension
    "T1140": "TA0005",  # Deobfuscate/Decode Files or Information
    "T1078": "TA0005",  # Valid Accounts (also Initial Access/Persistence)
    "T1107": "TA0005",  # File Deletion (legacy)
    "T1074": "TA0005",  # File Permissions Modification (legacy)
    "T1074.001": "TA0005",  # Windows File Permissions (legacy)
    "T1089": "TA0005",  # Disabling Security Tools (legacy)
    "T1096": "TA0005",  # NTFS Extended Attributes (legacy)
    "T1108": "TA0005",  # Redundant Access (legacy)
    "T1116": "TA0005",  # Code Signing (legacy)
    "T1141": "TA0005",  # Input Capture (legacy)
    "T1143": "TA0005",  # Encrypted/Encoded File (legacy)
    "T1144": "TA0005",  # Gatekeeper Bypass (legacy)
    "T1146": "TA0005",  # Clear Linux History (legacy)
    "T1150": "TA0005",  # Placed Hook (legacy)
    "T1151": "TA0005",  # Space After Filename (legacy)
    "T1152": "TA0005",  # Text Obfuscation (legacy)
    "T1153": "TA0005",  # Hidden Files (legacy)
    "T1156": "TA0005",  # .bash_profile (legacy)
    "T1157": "TA0005",  # Dylib Hijacking (legacy)
    "T1159": "TA0005",  # Launchctl (legacy)
    "T1162": "TA0005",  # Systemd Service (legacy)
    "T1163": "TA0005",  # Rc.common (legacy)
    "T1165": "TA0005",  # Startup Items (legacy)
    "T1167": "TA0005",  # File Permissions Modification (legacy)
    "T1176": "TA0005",  # Browser Extensions (legacy)
    "T1186": "TA0005",  # Process Hollowing (legacy)
    "T1191": "TA0005",  # Code Signing Certificates (legacy)
    "T1196": "TA0005",  # Sysmon Bypass (legacy)
    "T1221": "TA0005",  # Template Injection (legacy)
    "T1565": "TA0005",  # Data Manipulation
    "T1565.001": "TA0005",  # Stored Data Manipulation
    "T1553": "TA0005",  # Subvert Trust Controls
    "T1553.001": "TA0005",  # Gatekeeper Bypass
    "T1553.003": "TA0005",  # SIP and Trust Provider Hijacking
    "T1553.004": "TA0005",  # Install Root Certificate
    "T1553.005": "TA0005",  # Mark-of-the-Web Bypass
    "T1553.006": "TA0005",  # Code Signing Policy Modification
    "T1620": "TA0005",  # Reflective Code Loading
    "T1600": "TA0005",  # Encrypt/Encode Files (legacy)
    "T1600.001": "TA0005",  # Encrypt Files (legacy)
    "T1222": "TA0005",  # File Permissions Modification
    "T1222.001": "TA0005",  # Windows File Permissions Modification
    "T1222.002": "TA0005",  # Linux and Mac File Permissions Modification
    "T1218": "TA0005",  # System Binary Proxy Execution
    "T1218.001": "TA0005",  # Compiled HTML File
    "T1218.002": "TA0005",  # Control Panel
    "T1218.003": "TA0005",  # CMSTP
    "T1218.004": "TA0005",  # InstallUtil
    "T1218.005": "TA0005",  # Mshta
    "T1218.007": "TA0005",  # Msiexec
    "T1218.008": "TA0005",  # Odbcconf
    "T1218.009": "TA0005",  # Regsvr32
    "T1218.010": "TA0005",  # Regsvr32/squiblydoo
    "T1218.011": "TA0005",  # Rundll32
    "T1218.013": "TA0005",  # Mavinject
    "T1216": "TA0005",  # System Script Proxy Execution
    "T1216.001": "TA0005",  # Pubprn
    "T1202": "TA0005",  # Direct Excecution (legacy)
    "T1539": "TA0005",  # Steal Browser Cookies (legacy)
    # ================================================================
    # Credential Access (TA0006)
    # ================================================================
    "T1003": "TA0006",  # OS Credential Dumping
    "T1003.001": "TA0006",  # LSASS Memory
    "T1003.002": "TA0006",  # Security Account Manager
    "T1003.003": "TA0006",  # NTDS
    "T1003.004": "TA0006",  # LSA Secrets
    "T1003.005": "TA0006",  # Cached Domain Credentials
    "T1003.006": "TA0006",  # DCSync
    "T1003.007": "TA0006",  # Proc Filesystem
    "T1003.008": "TA0006",  # /etc/passwd and /etc/shadow
    "T1110": "TA0006",  # Brute Force
    "T1110.001": "TA0006",  # Password Guessing
    "T1110.002": "TA0006",  # Password Cracking
    "T1110.003": "TA0006",  # Password Spraying
    "T1110.004": "TA0006",  # Credential Stuffing
    "T1552": "TA0006",  # Unsecured Credentials
    "T1552.001": "TA0006",  # Credentials In Files
    "T1552.002": "TA0006",  # Credentials in Registry
    "T1552.003": "TA0006",  # Bash History
    "T1552.004": "TA0006",  # Private Keys
    "T1552.005": "TA0006",  # Cloud Instance Metadata API
    "T1552.006": "TA0006",  # Group Policy Preferences
    "T1552.007": "TA0006",  # Container API
    "T1558": "TA0006",  # Steal or Forge Kerberos Tickets
    "T1558.001": "TA0006",  # Kerberoasting
    "T1558.002": "TA0006",  # AS-REP Roasting
    "T1558.003": "TA0006",  # Kerberoasting: Silver Ticket
    "T1558.004": "TA0006",  # AS-REP Roasting: Diamond Ticket
    "T1056": "TA0006",  # Input Capture
    "T1056.001": "TA0006",  # Keylogging
    "T1056.002": "TA0006",  # GUI Input Capture
    "T1056.004": "TA0006",  # Credential API Hooking
    "T1087": "TA0006",  # Account Discovery (also Discovery)
    "T1087.001": "TA0006",  # Local Account
    "T1087.002": "TA0006",  # Domain Account
    "T1047": "TA0006",  # Windows Management Instrumentation (legacy → Discovery/Execution)
    "T1081": "TA0006",  # Credentials in Files (legacy)
    "T1111": "TA0006",  # Two-Factor Authentication Interception (legacy)
    "T1112": "TA0006",  # Modify Registry (legacy → Defense Evasion)
    "T1113": "TA0006",  # Screen Capture (legacy → Collection)
    "T1114": "TA0006",  # Email Collection (legacy → Collection)
    "T1119": "TA0006",  # Automated Collection (legacy → Collection)
    "T1120": "TA0006",  # Automated Exfiltration (legacy → Exfiltration)
    "T1135": "TA0006",  # Network Share Discovery (legacy → Discovery)
    "T1142": "TA0006",  # Keychain (legacy)
    "T1145": "TA0006",  # SSH Key (legacy)
    "T1154": "TA0006",  # Trap (legacy → Execution)
    "T1187": "TA0006",  # Forced Authentication (legacy)
    "T1212": "TA0006",  # Exploitation for Credential Access (legacy)
    "T1214": "TA0006",  # Credentials in Registry (legacy)
    "T1215": "TA0006",  # Kerberos Tickets (legacy)
    # ================================================================
    # Discovery (TA0007)
    # ================================================================
    "T1082": "TA0007",  # System Information Discovery
    "T1083": "TA0007",  # File and Directory Discovery
    "T1046": "TA0007",  # Network Service Discovery
    "T1049": "TA0007",  # System Network Configuration Discovery
    "T1010": "TA0007",  # Application Window Discovery
    "T1012": "TA0007",  # Query Registry
    "T1016": "TA0007",  # System Network Configuration Discovery
    "T1016.001": "TA0007",  # Internet Connection Discovery
    "T1016.002": "TA0007",  # Wi-Fi Discovery
    "T1018": "TA0007",  # Remote System Discovery
    "T1033": "TA0007",  # System Owner/User Discovery
    "T1057": "TA0007",  # Process Discovery
    "T1063": "TA0007",  # System Information Discovery (legacy)
    "T1069": "TA0007",  # Permission Groups Discovery
    "T1069.001": "TA0007",  # Local Groups
    "T1069.002": "TA0007",  # Domain Groups
    "T1071": "TA0007",  # Application Layer Protocol (legacy → C2)
    "T1071.001": "TA0007",  # Web Protocols (legacy → C2)
    "T1071.004": "TA0007",  # DNS (legacy → C2)
    "T1124": "TA0007",  # System Time Discovery (legacy)
    "T1135": "TA0007",  # Network Share Discovery
    "T1201": "TA0007",  # Password Policy Discovery
    "T1217": "TA0007",  # Browser Bookmark Discovery
    "T1220": "TA0007",  # Cloud Instance Metadata API (legacy)
    "T1482": "TA0007",  # Domain Trust Discovery
    "T1518": "TA0007",  # Software Discovery
    "T1518.001": "TA0007",  # Security Software Discovery
    "T1526": "TA0007",  # Cloud Service Discovery
    "T1528": "TA0007",  # System Service Discovery
    "T1529": "TA0007",  # System Information Discovery (Cloud)
    "T1530": "TA0007",  # Data from Cloud Storage
    "T1531": "TA0007",  # Cloud Instance Metadata API
    "T1537": "TA0007",  # Transfer Data to Cloud Account
    "T1613": "TA0007",  # Container and Resource Discovery
    "T1614": "TA0007",  # System Location Discovery
    "T1614.001": "TA0007",  # System Language Discovery
    "T1615": "TA0007",  # System Policy Discovery
    "T1622": "TA0007",  # Debugger Evasion
    "T1648": "TA0007",  # Server Software Component Discovery
    "T1649": "TA0007",  # Cloud Storage Object Discovery
    "T1651": "TA0007",  # Cloud Admin Console Discovery
    "T1652": "TA0007",  # Device Driver Discovery
    "T1654": "TA0007",  # Table Permission Discovery
    "T1007": "TA0007",  # System Service Discovery (legacy)
    "T1015": "TA0007",  # Accessibility Features Discovery (legacy → Persistence)
    "T1044": "TA0007",  # Execution through Module Load (legacy → Execution)
    # ================================================================
    # Lateral Movement (TA0008)
    # ================================================================
    "T1021": "TA0008",  # Remote Services
    "T1021.001": "TA0008",  # Remote Desktop Protocol
    "T1021.002": "TA0008",  # SMB/Windows Admin Shares
    "T1021.003": "TA0008",  # Distributed Component Object Model
    "T1021.004": "TA0008",  # SSH
    "T1021.005": "TA0008",  # VNC
    "T1021.006": "TA0008",  # Windows Remote Management
    "T1550": "TA0008",  # Use Alternate Authentication Material
    "T1550.002": "TA0008",  # Pass the Hash
    "T1550.003": "TA0008",  # Pass the Ticket
    "T1569": "TA0008",  # System Services
    "T1569.001": "TA0008",  # Launchctl
    "T1569.002": "TA0008",  # Service Execution
    "T1569.003": "TA0008",  # WMI
    "T1570": "TA0008",  # Lateral Tool Transfer
    "T1572": "TA0008",  # Protocol Tunneling
    "T1573": "TA0008",  # Dynamic Resolution
    "T1072": "TA0008",  # Software Deployment Tools
    "T1085": "TA0008",  # Rundll32 (legacy)
    "T1093": "TA0008",  # Logon Scripts (legacy → Persistence)
    # ================================================================
    # Collection (TA0009)
    # ================================================================
    "T1005": "TA0009",  # Data from Local System
    "T1006": "TA0009",  # Direct Access Volume
    "T1039": "TA0009",  # Data from Network Shared Drive
    "T1025": "TA0009",  # Data from Removable Media
    "T1020": "TA0009",  # Automated Exfiltration (legacy → Exfiltration)
    "T1029": "TA0009",  # Scheduled Transfer (legacy → Exfiltration)
    "T1030": "TA0009",  # Data Transfer Size Limits (legacy → Exfiltration)
    "T1041": "TA0009",  # Exfiltration Over C2 Channel (legacy → Exfiltration)
    "T1048": "TA0009",  # Exfiltration Over Alternative Protocol (legacy)
    "T1056": "TA0009",  # Input Capture (legacy → Credential Access)
    "T1113": "TA0009",  # Screen Capture
    "T1114": "TA0009",  # Email Collection
    "T1114.001": "TA0009",  # Local Email Collection
    "T1114.002": "TA0009",  # Remote Email Collection
    "T1114.003": "TA0009",  # Email Forwarding Rule
    "T1119": "TA0009",  # Automated Collection
    "T1123": "TA0009",  # Audio Capture
    "T1125": "TA0009",  # Video Capture
    "T1185": "TA0009",  # Browser Session Hijacking (legacy)
    "T1188": "TA0009",  # Data Transfer Size Limits (legacy)
    "T1213": "TA0009",  # Data from Information Repositories
    "T1213.001": "TA0009",  # Confluence
    "T1213.002": "TA0009",  # SharePoint
    "T1213.003": "TA0009",  # Code Repositories
    "T1606": "TA0009",  # Data from Information Repositories (legacy)
    "T1606.002": "TA0009",  # Forge Web Credentials (legacy)
    "T1608": "TA0009",  # Stage Capabilities (legacy → Resource Development)
    # ================================================================
    # Exfiltration (TA0010)
    # ================================================================
    "T1041": "TA0010",  # Exfiltration Over C2 Channel
    "T1048": "TA0010",  # Exfiltration Over Alternative Protocol
    "T1567": "TA0010",  # Exfiltration Over Web Service
    "T1567.001": "TA0010",  # Exfiltration to Cloud Storage
    "T1567.002": "TA0010",  # Exfiltration Over Web Service
    "T1567.003": "TA0010",  # Exfiltration Over Web Service: Exfiltration to Text Storage Sites
    "T1568": "TA0010",  # Exfiltration Over Alternative Protocol
    "T1568.002": "TA0010",  # Exfiltration Over Asymmetric Encrypted Protocol
    "T1020": "TA0010",  # Automated Exfiltration
    # ================================================================
    # Command and Control (TA0011)
    # ================================================================
    "T1071": "TA0011",  # Application Layer Protocol
    "T1071.001": "TA0011",  # Web Protocols
    "T1071.004": "TA0011",  # DNS
    "T1090": "TA0011",  # Proxy
    "T1090.001": "TA0011",  # Internal Proxy
    "T1090.003": "TA0011",  # Multi-hop Proxy
    "T1095": "TA0011",  # Non-Application Layer Protocol
    "T1098": "TA0011",  # Account Manipulation (legacy → Persistence/PrivEsc)
    "T1104": "TA0011",  # Multi-Stage Channels
    "T1105": "TA0011",  # Ingress Tool Transfer
    "T1108": "TA0011",  # Redundant Access (legacy)
    "T1112": "TA0011",  # Modify Registry (legacy)
    "T1132": "TA0011",  # Data Encoding
    "T1132.001": "TA0011",  # Standard Encoding
    "T1172": "TA0011",  # Domain Fronting (legacy)
    "T1185": "TA0011",  # Browser Session Hijacking (legacy)
    "T1571": "TA0011",  # Non-Standard Port
    "T1572": "TA0011",  # Protocol Tunneling
    "T1573": "TA0011",  # Dynamic Resolution
    "T1573.001": "TA0011",  # Fast Flux
    "T1573.002": "TA0011",  # Domain Generation Algorithms
    "T1580": "TA0011",  # Traffic Signaling
    "T1583": "TA0011",  # Compromise Infrastructure
    "T1584": "TA0011",  # Compromise Infrastructure: Botnet
    "T1585": "TA0011",  # Compromise Accounts
    "T1588": "TA0011",  # Obtain Capabilities
    "T1588.005": "TA0011",  # Exploits
    # ================================================================
    # Impact (TA0012)
    # ================================================================
    "T1486": "TA0012",  # Data Encrypted for Impact
    "T1489": "TA0012",  # Service Stop
    "T1490": "TA0012",  # Inhibit System Recovery
    "T1491": "TA0012",  # Defacement
    "T1491.001": "TA0012",  # Internal Defacement
    "T1496": "TA0012",  # Resource Hijacking
    "T1496.001": "TA0012",  # Compute Hijacking
    "T1497.001": "TA0012",  # Disk Wipe
    "T1497.003": "TA0012",  # Disk Wipe: Shadow Copy Deletion
    "T1498": "TA0012",  # Network Denial of Service
    "T1499": "TA0012",  # Endpoint Denial of Service
    "T1484": "TA0012",  # Data Destruction (legacy)
    "T1484.001": "TA0012",  # Internal Defacement (legacy)
    "T1484.002": "TA0012",  # External Defacement (legacy)
    "T1485": "TA0012",  # Data Destruction
    "T1529": "TA0012",  # System Shutdown/Reboot (legacy → Discovery)
    # ================================================================
    # Resource Development (TA0042)
    # ================================================================
    "T1583": "TA0042",  # Compromise Infrastructure
    "T1583.001": "TA0042",  # Domains
    "T1583.002": "TA0042",  # DNS Server
    "T1583.003": "TA0042",  # Virtual Private Server
    "T1583.004": "TA0042",  # Server
    "T1583.005": "TA0042",  # Botnet
    "T1583.006": "TA0042",  # Web Services
    "T1583.007": "TA0042",  # Serverless
    "T1583.008": "TA0042",  # Repurpose Infrastructure
    "T1584": "TA0042",  # Compromise Infrastructure: Botnet (legacy)
    "T1585": "TA0042",  # Compromise Accounts
    "T1585.001": "TA0042",  # Social Media
    "T1585.002": "TA0042",  # Email
    "T1585.003": "TA0042",  # Cloud
    "T1586": "TA0042",  # Develop Content
    "T1587": "TA0042",  # Develop Capabilities
    "T1587.001": "TA0042",  # Malware
    "T1587.002": "TA0042",  # Code Signing Certificate
    "T1587.003": "TA0042",  # Digital Certificate
    "T1587.004": "TA0042",  # Exploits
    "T1588": "TA0042",  # Obtain Capabilities
    "T1588.001": "TA0042",  # Malware
    "T1588.002": "TA0042",  # Tool
    "T1588.003": "TA0042",  # Code Signing Certificate
    "T1588.004": "TA0042",  # Digital Certificate
    "T1588.005": "TA0042",  # Exploits
    "T1588.006": "TA0042",  # Vulnerabilities
    "T1608": "TA0042",  # Stage Capabilities
    "T1608.001": "TA0042",  # Upload Malware
    "T1608.002": "TA0042",  # Upload Tool
    "T1608.003": "TA0042",  # Install Digital Certificate
    "T1609": "TA0042",  # Container Administration Command (legacy → Execution)
    # ================================================================
    # Reconnaissance (TA0043)
    # ================================================================
    "T1592": "TA0043",  # Gather Victim Host Information
    "T1592.001": "TA0043",  # Hardware
    "T1592.002": "TA0043",  # Firmware
    "T1592.003": "TA0043",  # Client Configurations
    "T1592.004": "TA0043",  # Host Name
    "T1592.005": "TA0043",  # Installed Software
    "T1593": "TA0043",  # Gather Victim Identity Information
    "T1593.001": "TA0043",  # Credentials
    "T1593.002": "TA0043",  # Email Addresses
    "T1593.003": "TA0043",  # Employee Names
    "T1594": "TA0043",  # Search Victim-Owned Websites
    "T1595": "TA0043",  # Active Scanning
    "T1595.001": "TA0043",  # Scanning IP Blocks
    "T1595.002": "TA0043",  # Vulnerability Scanning
    "T1595.003": "TA0043",  # Wordlist Scanning
    "T1596": "TA0043",  # Search Open Technical Databases
    "T1597": "TA0043",  # Search Open Websites/Domains
    "T1598": "TA0043",  # Email Addresses (legacy)
    "T1599": "TA0043",  # Credentials (legacy)
    "T1600": "TA0043",  # Collect Victim Data (legacy)
    "T1601": "TA0043",  # Social Media (legacy)
    "T1602": "TA0043",  # Search Engines (legacy)
    # ================================================================
    # Prompt Injection / ATLAS (TA0040)
    # ================================================================
    "AML.T0035": "TA0040",  # ML Attack Surface Discovery
    "AML.T0037": "TA0040",  # Data from Local System
    # ================================================================
    # Additional techniques found in datasets that need explicit mapping
    # ================================================================
    "T1001": "TA0009",  # Data Obfuscation (Collection)
    "T1001.002": "TA0009",  # Steganography
    "T1007": "TA0007",  # System Service Discovery
    "T1014": "TA0005",  # Rootkit (Defense Evasion)
    "T1015": "TA0003",  # Accessibility Features (Persistence)
    "T1016": "TA0007",  # System Network Configuration Discovery
    "T1027": "TA0005",  # Obfuscated Files or Information
    "T1035": "TA0002",  # Service Execution
    "T1036": "TA0005",  # Masquerading
    "T1037": "TA0003",  # Boot or Logon Initialization Scripts
    "T1039": "TA0009",  # Data from Network Shared Drive
    "T1040": "TA0009",  # Network Sniffing (Collection)
    "T1044": "TA0002",  # Execution through Module Load
    "T1046": "TA0007",  # Network Service Discovery
    "T1047": "TA0007",  # Windows Management Instrumentation
    "T1048": "TA0004",  # Abuse Elevation Control Mechanism
    "T1049": "TA0007",  # System Network Configuration Discovery
    "T1053": "TA0002",  # Scheduled Task/Job
    "T1055": "TA0004",  # Process Injection
    "T1056": "TA0006",  # Input Capture
    "T1057": "TA0007",  # Process Discovery
    "T1059": "TA0002",  # Command and Scripting Interpreter
    "T1063": "TA0007",  # System Information Discovery
    "T1064": "TA0002",  # Scripting
    "T1069": "TA0007",  # Permission Groups Discovery
    "T1070": "TA0005",  # Indicator Removal
    "T1072": "TA0008",  # Software Deployment Tools (Lateral Movement)
    "T1074": "TA0005",  # File Permissions Modification
    "T1077": "TA0008",  # Windows Admin Shares (Lateral Movement)
    "T1078": "TA0001",  # Valid Accounts
    "T1081": "TA0006",  # Credentials in Files
    "T1082": "TA0007",  # System Information Discovery
    "T1083": "TA0007",  # File and Directory Discovery
    "T1085": "TA0002",  # Rundll32
    "T1086": "TA0002",  # PowerShell
    "T1087": "TA0006",  # Account Discovery
    "T1088": "TA0005",  # Bypass User Account Control (legacy)
    "T1089": "TA0005",  # Disabling Security Tools
    "T1091": "TA0001",  # Replication Through Removable Media
    "T1093": "TA0003",  # Logon Scripts
    "T1095": "TA0011",  # Non-Application Layer Protocol
    "T1098": "TA0004",  # Account Manipulation
    "T1103": "TA0003",  # AppInit DLLs
    "T1105": "TA0011",  # Ingress Tool Transfer
    "T1106": "TA0011",  # Native API (C2 - legacy)
    "T1107": "TA0005",  # File Deletion
    "T1110": "TA0006",  # Brute Force
    "T1112": "TA0005",  # Modify Registry
    "T1113": "TA0009",  # Screen Capture
    "T1115": "TA0009",  # Clipboard Data
    "T1116": "TA0005",  # Code Signing
    "T1117": "TA0002",  # Regsvr32
    "T1118": "TA0002",  # InstallUtil
    "T1119": "TA0009",  # Automated Collection
    "T1120": "TA0010",  # Automated Exfiltration
    "T1121": "TA0002",  # Component Object Model
    "T1122": "TA0003",  # Component Object Model Hijacking
    "T1123": "TA0009",  # Audio Capture
    "T1124": "TA0007",  # System Time Discovery
    "T1125": "TA0009",  # Video Capture
    "T1126": "TA0005",  # Network Share Connection Removal
    "T1127": "TA0002",  # Trusted Developer Utilities
    "T1129": "TA0002",  # Shared Modules
    "T1132": "TA0011",  # Data Encoding
    "T1133": "TA0001",  # External Remote Services
    "T1134": "TA0004",  # Access Token Manipulation
    "T1135": "TA0007",  # Network Share Discovery
    "T1136": "TA0003",  # Create Account
    "T1137": "TA0003",  # Office Application Startup
    "T1140": "TA0005",  # Deobfuscate/Decode Files
    "T1158": "TA0003",  # Hidden Files and Directories
    "T1170": "TA0002",  # Mshta
    "T1176": "TA0003",  # Browser Extensions
    "T1187": "TA0006",  # Forced Authentication
    "T1189": "TA0001",  # Drive-by Compromise
    "T1195": "TA0001",  # Supply Chain Compromise
    "T1197": "TA0005",  # BITS Jobs (Defense Evasion)
    "T1201": "TA0007",  # Password Policy Discovery
    "T1202": "TA0005",  # Direct Execution
    "T1207": "TA0005",  # BITS Jobs (legacy)
    "T1210": "TA0008",  # Exploitation of Remote Services
    "T1216": "TA0005",  # System Script Proxy Execution
    "T1217": "TA0007",  # Browser Bookmark Discovery
    "T1218": "TA0005",  # System Binary Proxy Execution
    "T1219": "TA0009",  # Remote Access Software
    "T1220": "TA0007",  # Cloud Instance Metadata API (legacy)
    "T1221": "TA0005",  # Template Injection
    "T1222": "TA0005",  # File Permissions Modification
    "T1482": "TA0007",  # Domain Trust Discovery
    "T1484": "TA0012",  # Data Destruction
    "T1485": "TA0012",  # Data Destruction
    "T1486": "TA0012",  # Data Encrypted for Impact
    "T1489": "TA0012",  # Service Stop
    "T1490": "TA0012",  # Inhibit System Recovery
    "T1491": "TA0012",  # Defacement
    "T1496": "TA0012",  # Resource Hijacking
    "T1497": "TA0012",  # Disk Wipe
    "T1499": "TA0012",  # Endpoint Denial of Service
    "T1505": "TA0003",  # Server Software Component
    "T1518": "TA0007",  # Software Discovery
    "T1526": "TA0007",  # Cloud Service Discovery
    "T1528": "TA0007",  # System Service Discovery (Cloud)
    "T1529": "TA0007",  # System Information Discovery (Cloud)
    "T1530": "TA0009",  # Data from Cloud Storage
    "T1531": "TA0007",  # Cloud Instance Metadata API
    "T1537": "TA0009",  # Transfer Data to Cloud Account
    "T1539": "TA0006",  # Steal Browser Cookies (Credential Access)
    "T1542": "TA0003",  # Certificate Authority Misuse
    "T1542.001": "TA0003",  # Forged Certificates
    "T1543": "TA0003",  # Create or Modify System Process
    "T1546": "TA0003",  # Event Triggered Execution
    "T1547": "TA0003",  # Boot or Logon Autostart Execution
    "T1548": "TA0004",  # Abuse Elevation Control Mechanism
    "T1550": "TA0008",  # Use Alternate Authentication Material
    "T1552": "TA0006",  # Unsecured Credentials
    "T1553": "TA0005",  # Subvert Trust Controls
    "T1555": "TA0003",  # Credentials from Password Stores
    "T1555.001": "TA0006",  # Keychain
    "T1555.003": "TA0006",  # Credentials from Web Browsers
    "T1555.004": "TA0006",  # Credentials from Password Managers
    "T1555.006": "TA0006",  # Cloud Secrets Management
    "T1556": "TA0006",  # Modify Authentication Process
    "T1556.001": "TA0006",  # Domain Controller Authentication
    "T1556.002": "TA0006",  # Password Filter DLL
    "T1556.003": "TA0006",  # Pluggable Authentication Modules
    "T1557": "TA0006",  # Adversary-in-the-Middle
    "T1557.001": "TA0006",  # LLMNR/NBT-NS Poisoning and SMB Relay
    "T1558": "TA0006",  # Steal or Forge Kerberos Tickets
    "T1559": "TA0002",  # Inter-Process Communication
    "T1560": "TA0009",  # Archive Collected Data
    "T1560.001": "TA0009",  # Archive via Utility
    "T1560.002": "TA0009",  # Archive via Library
    "T1562": "TA0005",  # Impair Defenses
    "T1563": "TA0005",  # Service Exhaustion (legacy)
    "T1564": "TA0005",  # Hide Artifacts
    "T1565": "TA0005",  # Data Manipulation
    "T1566": "TA0001",  # Phishing
    "T1567": "TA0010",  # Exfiltration Over Web Service
    "T1568": "TA0010",  # Exfiltration Over Alternative Protocol
    "T1569": "TA0008",  # System Services
    "T1570": "TA0008",  # Lateral Tool Transfer
    "T1571": "TA0011",  # Non-Standard Port
    "T1572": "TA0008",  # Protocol Tunneling
    "T1573": "TA0011",  # Dynamic Resolution
    "T1574": "TA0003",  # Hijack Execution Flow
    "T1578": "TA0005",  # Modify Cloud Compute Infrastructure
    "T1578.001": "TA0005",  # Create Snapshot
    "T1578.002": "TA0005",  # Create Cloud Instance
    "T1580": "TA0011",  # Traffic Signaling
    "T1588": "TA0042",  # Obtain Capabilities
    "T1589": "TA0043",  # Gather Victim Identity Information (legacy)
    "T1590": "TA0043",  # Gather Victim Network Information
    "T1591": "TA0043",  # Gather Victim Org Information
    "T1595": "TA0043",  # Active Scanning
    "T1606": "TA0043",  # Forge Web Credentials (legacy)
    "T1608": "TA0042",  # Stage Capabilities
    "T1609": "TA0002",  # Container Administration Command
    "T1610": "TA0002",  # Deploy Container
    "T1611": "TA0002",  # Escape to Host
    "T1612": "TA0005",  # Build Image on Host
    "T1613": "TA0007",  # Container and Resource Discovery
    "T1614": "TA0007",  # System Location Discovery
    "T1615": "TA0007",  # System Policy Discovery
    "T1619": "TA0009",  # Collect Data from Cloud
    "T1620": "TA0005",  # Reflective Code Loading
    "T1622": "TA0007",  # Debugger Evasion
    "T1647": "TA0005",  # Signature Verification Avoidance
    "T1648": "TA0007",  # Server Software Component Discovery
    "T1649": "TA0007",  # Cloud Storage Object Discovery
    "T1650": "TA0007",  # Cloud Admin Console Discovery (legacy)
    "T1651": "TA0007",  # Cloud Admin Console Discovery
    "T1652": "TA0007",  # Device Driver Discovery
    "T1653": "TA0005",  # Power Settings
    "T1654": "TA0007",  # Table Permission Discovery
    "T1656": "TA0042",  # Forge Web Credentials
    "T1657": "TA0042",  # Domain Registrar
    "T1659": "TA0002",  # Cloud Administration Command
    "T1660": "TA0042",  # Acquire Infrastructure
    "T1661": "TA0042",  # Compromise Accounts
    "T1662": "TA0042",  # Email Accounts
    "T1663": "TA0042",  # Compute Resources
    "T1664": "TA0042",  # Cloud Storage
    "T1665": "TA0042",  # Web Services
    "T1666": "TA0042",  # Serverless
    "T1667": "TA0042",  # Virtual Private Server
    "T1668": "TA0042",  # Domains
    "T1669": "TA0042",  # DNS Server
    "T1670": "TA0042",  # Botnet
    "T1671": "TA0042",  # Repurpose Infrastructure
    "T1672": "TA0042",  # Obtain Capabilities
    "T1673": "TA0007",  # Cloud Instance Metadata Discovery
    "T1674": "TA0042",  # Malware
    "T1675": "TA0042",  # Sign Malware
    "T1685": "TA0007",  # Cloud API Discovery
    "T1685.001": "TA0007",  # Cloud Service Discovery
    "T1685.002": "TA0007",  # Cloud Storage Discovery
    "T1685.004": "TA0007",  # Cloud Compute Discovery
    "T1685.005": "TA0007",  # Cloud Network Discovery
    "T1685.006": "TA0007",  # Cloud Monitoring Discovery
    "T1686": "TA0042",  # Compromise Infrastructure
    "T1688": "TA0042",  # Obtain Capabilities
    "T1689": "TA0042",  # Stage Capabilities
    "T1690": "TA0042",  # Develop Capabilities
}

# Fallback ranges for techniques not explicitly listed above.
# These map technique number prefixes to their primary tactic.
_FALLBACK_RANGES: list[tuple[int, int, str]] = [
    # T1001-T1005 → Collection / Credential Access
    (1001, 1002, "TA0009"),  # Data Obfuscation, etc.
    (1003, 1003, "TA0006"),  # OS Credential Dumping
    (1005, 1005, "TA0009"),  # Data from Local System
    (1006, 1006, "TA0009"),  # Direct Access Volume
    (1007, 1007, "TA0007"),  # System Service Discovery
    # T1010-T1014 → Discovery / Defense Evasion
    (1010, 1012, "TA0007"),  # Various Discovery
    (1014, 1014, "TA0005"),  # Rootkit
    (1015, 1016, "TA0007"),  # Accessibility/System Network Config Discovery
    (1018, 1018, "TA0007"),  # Remote System Discovery
    # T1020 → Exfiltration
    (1020, 1020, "TA0010"),  # Automated Exfiltration
    # T1021 → Lateral Movement
    (1021, 1021, "TA0008"),  # Remote Services
    # T1025 → Collection
    (1025, 1025, "TA0009"),  # Data from Removable Media
    # T1027 → Defense Evasion
    (1027, 1027, "TA0005"),  # Obfuscated Files
    # T1029-T1030 → Exfiltration / Collection
    (1029, 1029, "TA0010"),  # Scheduled Transfer
    (1030, 1030, "TA0010"),  # Data Transfer Size Limits
    # T1033 → Discovery
    (1033, 1033, "TA0007"),  # System Owner/User Discovery
    # T1035-T1037 → Execution / Persistence
    (1035, 1035, "TA0002"),  # Service Execution
    (1036, 1036, "TA0005"),  # Masquerading
    (1037, 1037, "TA0003"),  # Boot/Logon Init Scripts
    # T1039 → Collection
    (1039, 1039, "TA0009"),  # Data from Network Shared Drive
    # T1040-T1041 → Collection / Exfiltration
    (1040, 1040, "TA0009"),  # Network Sniffing
    (1041, 1041, "TA0010"),  # Exfil Over C2
    # T1044 → Execution
    (1044, 1044, "TA0002"),  # Execution through Module Load
    # T1046-T1049 → Discovery
    (1046, 1049, "TA0007"),  # Network Discovery
    # T1053 → Execution
    (1053, 1053, "TA0002"),  # Scheduled Task
    # T1055-T1057 → Privilege Escalation / Discovery
    (1055, 1055, "TA0004"),  # Process Injection
    (1056, 1056, "TA0006"),  # Input Capture
    (1057, 1057, "TA0007"),  # Process Discovery
    # T1059 → Execution
    (1059, 1059, "TA0002"),  # Command/Scripting Interpreter
    # T1063-T1064 → Discovery / Execution
    (1063, 1063, "TA0007"),  # System Info Discovery
    (1064, 1064, "TA0002"),  # Scripting
    # T1068-T1069 → Privilege Escalation / Discovery
    (1068, 1068, "TA0004"),  # Exploitation for Priv Esc
    (1069, 1069, "TA0007"),  # Permission Groups Discovery
    # T1070-T1074 → Defense Evasion
    (1070, 1070, "TA0005"),  # Indicator Removal
    (1071, 1071, "TA0011"),  # Application Layer Protocol
    (1072, 1072, "TA0008"),  # Software Deployment Tools
    (1074, 1074, "TA0005"),  # File Permissions Modification
    # T1077-T1078 → Lateral Movement / Initial Access
    (1077, 1077, "TA0008"),  # Windows Admin Shares
    (1078, 1078, "TA0001"),  # Valid Accounts
    # T1081-T1083 → Credential Access / Discovery
    (1081, 1081, "TA0006"),  # Credentials in Files
    (1082, 1083, "TA0007"),  # System/File Discovery
    # T1085-T1089 → Execution / Defense Evasion
    (1085, 1085, "TA0002"),  # Rundll32
    (1086, 1086, "TA0002"),  # PowerShell
    (1087, 1087, "TA0006"),  # Account Discovery
    (1088, 1088, "TA0004"),  # Bypass UAC
    (1089, 1089, "TA0005"),  # Disabling Security Tools
    # T1090-T1098 → C2 / Persistence / PrivEsc
    (1090, 1090, "TA0011"),  # Proxy
    (1091, 1091, "TA0001"),  # Replication Through Removable Media
    (1093, 1098, "TA0003"),  # Logon Scripts through Account Manipulation
    # T1100-T1108 → Persistence / C2
    (1103, 1103, "TA0003"),  # AppInit DLLs
    (1104, 1104, "TA0011"),  # Multi-Stage Channels
    (1105, 1105, "TA0011"),  # Ingress Tool Transfer
    (1106, 1106, "TA0011"),  # Native API
    (1107, 1107, "TA0005"),  # File Deletion
    # T1110-T1127 → Credential Access through Execution
    (1110, 1110, "TA0006"),  # Brute Force
    (1112, 1112, "TA0005"),  # Modify Registry
    (1113, 1113, "TA0009"),  # Screen Capture
    (1115, 1115, "TA0009"),  # Clipboard Data
    (1116, 1116, "TA0005"),  # Code Signing
    (1117, 1118, "TA0002"),  # Regsvr32, InstallUtil
    (1119, 1119, "TA0009"),  # Automated Collection
    (1120, 1120, "TA0010"),  # Automated Exfiltration
    (1121, 1121, "TA0002"),  # Component Object Model
    (1122, 1122, "TA0003"),  # COM Hijacking
    (1123, 1123, "TA0009"),  # Audio Capture
    (1124, 1124, "TA0007"),  # System Time Discovery
    (1125, 1125, "TA0009"),  # Video Capture
    (1126, 1126, "TA0005"),  # Network Share Connection Removal
    (1127, 1127, "TA0002"),  # Trusted Developer Utilities
    (1129, 1129, "TA0002"),  # Shared Modules
    # T1132 → C2
    (1132, 1132, "TA0011"),  # Data Encoding
    # T1133-T1137 → Initial Access / Persistence
    (1133, 1133, "TA0001"),  # External Remote Services
    (1134, 1134, "TA0004"),  # Access Token Manipulation
    (1135, 1135, "TA0007"),  # Network Share Discovery
    (1136, 1136, "TA0003"),  # Create Account
    (1137, 1137, "TA0003"),  # Office Application Startup
    # T1140 → Defense Evasion
    (1140, 1140, "TA0005"),  # Deobfuscate/Decode
    # T1142-T1148 → Credential Access / Defense Evasion
    (1142, 1146, "TA0006"),  # Various credential access
    (1148, 1148, "TA0010"),  # Exfil Over Alternative Protocol
    # T1150-T1159 → Defense Evasion / Persistence
    (1150, 1159, "TA0005"),  # Various Defense Evasion
    # T1160-T1176 → Persistence
    (1160, 1176, "TA0003"),  # Various Persistence
    # T1178-T1189 → Defense Evasion / Initial Access
    (1178, 1186, "TA0005"),  # Various Defense Evasion
    (1187, 1187, "TA0006"),  # Forced Authentication
    (1188, 1188, "TA0009"),  # Data Transfer Size Limits
    (1189, 1189, "TA0001"),  # Drive-by Compromise
    # T1190-T1197 → Initial Access / Defense Evasion
    (1190, 1190, "TA0001"),  # Exploit Public-Facing Application
    (1191, 1197, "TA0005"),  # Code Signing through BITS
    # T1199-T1207 → Initial Access / Discovery / Defense Evasion
    (1199, 1199, "TA0001"),  # Trusted Relationship
    (1200, 1200, "TA0001"),  # Hardware Additions
    (1201, 1201, "TA0007"),  # Password Policy Discovery
    (1202, 1202, "TA0005"),  # Direct Execution
    (1204, 1204, "TA0002"),  # User Execution
    (1207, 1207, "TA0005"),  # BITS Jobs
    # T1210-T1222 → Lateral Movement / Defense Evasion
    (1210, 1210, "TA0008"),  # Exploitation of Remote Services
    (1212, 1215, "TA0006"),  # Credential Access
    (1216, 1218, "TA0005"),  # System Script/Binary Proxy Execution
    (1219, 1219, "TA0009"),  # Remote Access Software
    (1220, 1220, "TA0007"),  # Cloud Instance Metadata API
    (1221, 1222, "TA0005"),  # Template Injection / File Permissions
    # T1480-T1499 → Impact
    (1480, 1480, "TA0005"),  # Registry Run Keys (legacy)
    (1482, 1482, "TA0007"),  # Domain Trust Discovery
    (1484, 1499, "TA0012"),  # Impact
    # T1505 → Persistence
    (1505, 1505, "TA0003"),  # Server Software Component
    # T1518-T1539 → Discovery / Collection / Credential Access
    (1518, 1518, "TA0007"),  # Software Discovery
    (1526, 1531, "TA0007"),  # Various Discovery
    (1537, 1537, "TA0009"),  # Transfer Data to Cloud
    (1539, 1539, "TA0006"),  # Steal Browser Cookies
    # T1542-T1548 → Persistence / Privilege Escalation
    (1542, 1543, "TA0003"),  # CA Misuse / Create System Process
    (1546, 1547, "TA0003"),  # Event Triggered / Autostart
    (1548, 1548, "TA0004"),  # Abuse Elevation Control
    # T1550-T1559 → Lateral Movement / Credential Access / Execution
    (1550, 1550, "TA0008"),  # Use Alternate Auth Material
    (1552, 1552, "TA0006"),  # Unsecured Credentials
    (1553, 1553, "TA0005"),  # Subvert Trust Controls
    (1555, 1555, "TA0003"),  # Password Stores
    (1556, 1556, "TA0006"),  # Modify Authentication Process
    (1557, 1557, "TA0006"),  # Adversary-in-the-Middle
    (1558, 1558, "TA0006"),  # Kerberos Tickets
    (1559, 1559, "TA0002"),  # Inter-Process Communication
    # T1560-T1569 → Collection / Defense Evasion / Lateral Movement
    (1560, 1560, "TA0009"),  # Archive Collected Data
    (1562, 1562, "TA0005"),  # Impair Defenses
    (1563, 1564, "TA0005"),  # Various Defense Evasion
    (1565, 1565, "TA0005"),  # Data Manipulation
    (1566, 1566, "TA0001"),  # Phishing
    (1567, 1567, "TA0010"),  # Exfiltration Over Web Service
    (1568, 1568, "TA0010"),  # Exfiltration Over Alternative Protocol
    (1569, 1569, "TA0008"),  # System Services
    # T1570-T1580 → Lateral Movement / C2
    (1570, 1570, "TA0008"),  # Lateral Tool Transfer
    (1571, 1571, "TA0011"),  # Non-Standard Port
    (1572, 1573, "TA0011"),  # Protocol Tunneling / Dynamic Resolution
    (1574, 1574, "TA0003"),  # Hijack Execution Flow
    (1578, 1578, "TA0005"),  # Modify Cloud Compute
    (1580, 1580, "TA0011"),  # Traffic Signaling
    # T1583-T1595 → Resource Development / Reconnaissance
    (1583, 1583, "TA0042"),  # Compromise Infrastructure
    (1584, 1584, "TA0042"),  # Compromise Infrastructure
    (1585, 1585, "TA0042"),  # Compromise Accounts
    (1586, 1588, "TA0042"),  # Develop Capabilities / Obtain Capabilities
    (1589, 1595, "TA0043"),  # Reconnaissance
    # T1596-T1608 → Reconnaissance / Resource Development
    (1596, 1597, "TA0043"),  # Search Open Technical/Websites
    (1600, 1602, "TA0043"),  # Various Reconnaissance
    (1606, 1606, "TA0009"),  # Forge Web Credentials
    (1608, 1608, "TA0042"),  # Stage Capabilities
    # T1609-T1611 → Execution
    (1609, 1611, "TA0002"),  # Container Admin / Deploy / Escape
    # T1612-T1615 → Defense Evasion / Discovery
    (1612, 1612, "TA0005"),  # Build Image on Host
    (1613, 1615, "TA0007"),  # Various Discovery
    # T1619-T1622 → Collection / Defense Evasion / Discovery
    (1619, 1619, "TA0009"),  # Data from Cloud
    (1620, 1620, "TA0005"),  # Reflective Code Loading
    (1622, 1622, "TA0007"),  # Debugger Evasion
    # T1647-T1675 → Defense Evasion / Discovery / Resource Development
    (1647, 1647, "TA0005"),  # Signature Verification Avoidance
    (1648, 1654, "TA0007"),  # Various Discovery
    (1655, 1657, "TA0042"),  # Resource Development
    (1659, 1659, "TA0002"),  # Cloud Administration Command
    (1660, 1675, "TA0042"),  # Resource Development
    # T1685-T1690 → Discovery / Resource Development
    (1685, 1685, "TA0007"),  # Cloud API Discovery
    (1686, 1690, "TA0042"),  # Resource Development
]


def _fallback_tactic(tech_id: str) -> str | None:
    """Try to determine tactic from technique number ranges.

    Parses the numeric portion of a technique ID like "T1059.001" → 1059,
    then checks against fallback ranges. Falls back to "TA0007" (Discovery)
    as the safest default if no range matches.
    """
    # Extract the numeric base (e.g. "T1059.001" → 1059)
    m = re.match(r"T(\d+)", tech_id)
    if not m:
        # ATLAS / AML techniques default to Prompt Injection
        if tech_id.startswith("AML."):
            return "TA0040"
        return None

    num = int(m.group(1))

    # Check fallback ranges
    for low, high, tactic_id in _FALLBACK_RANGES:
        if low <= num <= high:
            return tactic_id

    # Ultimate fallback: Discovery is safest for unknown techniques
    return "TA0007"


def get_tactic_for_technique(tech_id: str) -> str | None:
    """Return tactic_id (e.g. 'TA0002') for a given technique ID (e.g. 'T1059.001').

    First checks explicit TECHNIQUE_TO_TACTIC mapping, then falls back to
    numeric range matching. Returns None only for completely unparseable IDs.
    """
    # Direct lookup
    if tech_id in TECHNIQUE_TO_TACTIC:
        return TECHNIQUE_TO_TACTIC[tech_id]

    # For sub-techniques like "T1059.001", try the parent technique "T1059"
    if "." in tech_id:
        parent = tech_id.split(".")[0]
        if parent in TECHNIQUE_TO_TACTIC:
            return TECHNIQUE_TO_TACTIC[parent]

    # Fallback to range-based mapping
    return _fallback_tactic(tech_id)


def get_tactic_name(tactic_id: str) -> str | None:
    """Return display name (e.g. 'Execution') for a tactic ID (e.g. 'TA0002')."""
    return TACTIC_INFO.get(tactic_id)


def resolve_tactic_name_to_id(tactic_name: str) -> str | None:
    """Resolve a tactic name (e.g. 'execution' or 'privilege-escalation') to a tactic ID.

    Handles the Caldera YAML format where tactic names are lowercase with hyphens.
    """
    normalized = tactic_name.lower().strip().replace(" ", "-").replace("_", "-")
    return TACTIC_NAME_TO_ID.get(normalized)
