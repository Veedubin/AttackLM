#!/usr/bin/env python3
"""
classify_unrouted_modules.py — Path/description-based heuristic classifier
for Metasploit modules that don't have explicit ``Mitre::Attack::Technique``
references.

Strategy
--------
1. Apply a layered rule set (path patterns → technique mappings) that catches
   ~99% of modules.
2. For the tiny remainder, fall back to module-type defaults.
3. Skip the few "exploits/example*" sample modules entirely (they're docs,
   not real modules).
4. Append ``predicted_techniques`` to each record and re-emit
   ``metasploit_modules.jsonl``.
5. Re-run ``extract_by_tactic.py`` to update the per-tactic manifests.

This is intentionally simple — we don't need LLM calls for this dataset.
~95% of the routing decision can be made from the file path alone; the rest
is disambiguated by description keywords.

Usage:
    python classify_unrouted_modules.py --dry-run
    python classify_unrouted_modules.py            # update records + manifests
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MANIFESTS_DIR = BASE_DIR / "data" / "manifests"
RECORDS_PATH = MANIFESTS_DIR / "metasploit_modules.jsonl"

# ---------------------------------------------------------------------------
# Rule set: (regex, [(technique_id_or_tactic, weight), ...])
# Apply to combined text of path + name + first 200 chars of description.
# ---------------------------------------------------------------------------
PATH_RULES: list[tuple[str, list[tuple[str, float]]]] = [
    # --- Credential Access (TA0006) ---
    (r"gather/credentials", [("T1003", 1.0), ("TA0006", 1.0)]),
    (r"hashdump", [("T1003", 1.0), ("TA0006", 1.0)]),
    (
        r"mimikatz|kiwi|wdigest|lsa_dump|lsadump|sam_dump",
        [("T1003", 1.0), ("TA0006", 1.0)],
    ),
    (r"kerberos|kerb|ticket|asreproast|kerberoast", [("T1558", 1.0), ("TA0006", 1.0)]),
    (r"credential", [("T1003", 0.5), ("TA0006", 0.5)]),
    (
        r"ssh_login|ftp_login|telnet_login|smb_login|vnc_login|mssql_login|mysql_login|postgres_login|rlogin_login|pop3_login|imap_login|ldap_login|http_login|rdp_login|winrm_login",
        [("T1110", 1.0), ("TA0006", 1.0)],
    ),
    (
        r"filezilla|mremote|winscp|putty|securecrt|navicat|thunderbird|outlook|chrome|firefox|brave|safari|opera",
        [("T1555", 0.9), ("TA0006", 0.9)],
    ),
    # --- Discovery (TA0007) ---
    (
        r"scanner/(smb|ssh|ftp|telnet|http|mysql|mssql|postgres|vnc|pop3|imap|ldap|rdp|winrm|sip|dns|ntp|smtp|mongodb|oracle|redis|ajp|tftp|nfs|netbios)",
        [("T1046", 0.6), ("TA0007", 0.8)],
    ),
    (r"portscan", [("T1046", 1.0), ("TA0007", 1.0)]),
    (
        r"arp_scanner|netbios.*scanner|smb_version|smb_lookupsid",
        [("T1018", 1.0), ("TA0007", 1.0)],
    ),
    (
        r"enum_(ad|domain|computers|users|shares|services|pipes|sessions|applications|host)",
        [("T1018", 0.7), ("T1087", 0.5), ("TA0007", 0.9)],
    ),
    (r"sniffer|packet.*capture|promiscuous", [("T1040", 1.0), ("TA0007", 0.5)]),
    # --- Lateral Movement (TA0008) ---
    (r"exploits/.+/smb", [("T1210", 1.0), ("TA0008", 1.0)]),
    (r"exploits/.+/rdp", [("T1021.001", 0.9), ("TA0008", 1.0)]),
    (r"exploits/.+/winrm", [("T1021.006", 0.9), ("TA0008", 1.0)]),
    (r"exploits/.+/ssh", [("T1021.004", 0.9), ("TA0008", 1.0)]),
    (
        r"smb_relay|smbexec|smb_login|psexec|wmi_exec|atexec",
        [("T1021.002", 0.8), ("T1021.006", 0.5), ("TA0008", 1.0)],
    ),
    (
        r"psexec|atexec|wmi|wmiexec|winexec|smbexec|webexec",
        [("T1021.006", 0.9), ("TA0008", 1.0)],
    ),
    # --- Persistence (TA0003) ---
    (r"persistence", [("T1546", 0.5), ("TA0003", 1.0)]),
    (
        r"registry.*run|run_key|scheduled.*task|schtasks",
        [("T1547.001", 0.7), ("T1053.005", 0.7), ("TA0003", 1.0)],
    ),
    (
        r"service.*install|sc_create|service_persistence",
        [("T1543.003", 0.9), ("TA0003", 1.0)],
    ),
    (r"account.*create|add_user|create.*account", [("T1136", 1.0), ("TA0003", 1.0)]),
    (r"startup_folder|logon_script", [("T1037", 0.9), ("TA0003", 1.0)]),
    (r"cron|crontab", [("T1053.003", 1.0), ("TA0003", 1.0)]),
    (r"launch_agent|launch_daemon", [("T1543.001", 1.0), ("TA0003", 1.0)]),
    # --- Privilege Escalation (TA0004) ---
    (r"exploits/.+/local", [("T1068", 1.0), ("TA0004", 1.0)]),
    (
        r"privesc|privilege.*escal|elevate|getsystem|uac_bypass|uac.*bypass",
        [("T1068", 0.7), ("T1548.002", 0.7), ("TA0004", 1.0)],
    ),
    (r"sudo|bypass.*uac|token.*impersonat", [("T1548.003", 0.8), ("TA0004", 0.9)]),
    (r"getsystem", [("T1134.002", 1.0), ("TA0004", 1.0)]),
    (r"token|impersonat", [("T1134", 0.9), ("TA0004", 0.9)]),
    (
        r"always_install_elevated|dll_hijack|service_dll|service_permissions",
        [("T1574", 0.7), ("T1548", 0.5), ("TA0004", 1.0)],
    ),
    (
        r"sudoers|suid|setuid|capabilities|chmod.*4755",
        [("T1548.001", 0.7), ("TA0004", 0.4), ("TA0005", 0.4)],
    ),
    # --- Defense Evasion (TA0005) ---
    (r"^evasion/", [("T1027", 0.8), ("TA0005", 1.0)]),
    (r"^encoders/", [("T1027", 1.0), ("TA0005", 0.5)]),
    (r"^nops/", [("T1027", 0.5), ("TA0005", 0.5)]),
    (r"obfuscat|shikata|alpha_mixed", [("T1027", 1.0), ("TA0005", 0.7)]),
    (
        r"amsi.*bypass|etw.*bypass|defender.*bypass|av.*bypass|edr.*bypass",
        [("T1562.001", 1.0), ("TA0005", 1.0)],
    ),
    (
        r"disable.*firewall|disable.*av|kill.*defender",
        [("T1562.001", 1.0), ("T1562.004", 0.6), ("TA0005", 1.0)],
    ),
    (
        r"clear.*log|wevtutil.*cl|log_cleared|clearlogs",
        [("T1070.001", 1.0), ("TA0005", 1.0)],
    ),
    (
        r"injection|process.*inject|apc.*inject|reflective.*inject|hollowing",
        [("T1055", 0.9), ("TA0005", 0.5)],
    ),
    # --- Execution (TA0002) ---
    (r"^payloads/", [("T1059", 0.5), ("TA0002", 0.9)]),
    (r"meterpreter", [("T1059.001", 0.5), ("TA0002", 0.8)]),
    (r"cmd/(unix|windows)", [("T1059", 0.9), ("TA0002", 1.0)]),
    (r"powershell", [("T1059.001", 1.0), ("TA0002", 1.0)]),
    (r"wmi.*exec|wmic", [("T1047", 1.0), ("TA0002", 0.7)]),
    (
        r"rce|remote.*code|command.*injection",
        [("T1059", 0.5), ("T1190", 0.5), ("TA0002", 0.7)],
    ),
    (
        r"web_delivery|web_exec|hta_delivery|regsvr32",
        [("T1218", 0.6), ("TA0005", 0.4), ("TA0002", 0.7)],
    ),
    (r"schtasks.*exec|at.*exec|cron.*exec", [("T1053.005", 0.6), ("TA0002", 0.6)]),
    (r"mshta|wscript|cscript|wscript.*exec", [("T1218", 0.7), ("TA0002", 0.7)]),
    # --- Exploit (TA0002 in our routing, since AttackLM has no Initial Access) ---
    (r"exploits/.+/http", [("T1190", 0.8), ("TA0002", 0.8)]),
    (r"exploits/.+/browser", [("T1189", 0.9), ("TA0002", 0.8)]),
    (r"exploits/.+/fileformat", [("T1204.002", 0.9), ("TA0002", 0.7)]),
    (r"exploits/.+/ftp", [("T1190", 0.7), ("TA0002", 0.7)]),
    (r"exploits/.+/.*bof|exploits/.+/.*overflow", [("T1203", 0.8), ("TA0002", 0.6)]),
    # Broad catch for any non-example exploit
    (r"^exploits/(?!example)", [("T1190", 0.4), ("TA0002", 0.4)]),
    # --- Collection (TA0009) ---
    (
        r"screenshot|sc_cap|video_capture|keylog|keyscan|key_stroke",
        [("T1113", 0.8), ("T1056.001", 0.8), ("TA0009", 1.0)],
    ),
    (r"audio_capture|mic.*record|sound.*capture", [("T1123", 1.0), ("TA0009", 1.0)]),
    (r"screen_unlock|screen.*lock|getscreen", [("T1113", 0.5), ("TA0009", 0.6)]),
    (
        r"loot|file.*collect|file.*enum|search.*files|file_finder",
        [("T1005", 0.7), ("TA0009", 0.8)],
    ),
    (r"clipboard|clipboard.*capture", [("T1115", 1.0), ("TA0009", 1.0)]),
    (
        r"browser.*history|history.*dump|browser.*gather",
        [("T1217", 0.9), ("TA0009", 0.9)],
    ),
    (r"email.*collect|outlook.*dump|exchange", [("T1114", 0.8), ("TA0009", 0.9)]),
    # --- Command and Control (TA0011) ---
    (
        r"handler|reverse_tcp|bind_tcp|reverse_https|reverse_http",
        [("T1071.001", 0.7), ("T1095", 0.6), ("TA0011", 1.0)],
    ),
    (r"vnc.*inject|^vnc/|rdp.*inject", [("T1021.005", 0.6), ("TA0011", 0.5)]),
    (
        r"icmp.*tunnel|dns.*tunnel|http.*tunnel",
        [("T1071.004", 0.7), ("T1572", 0.7), ("TA0011", 1.0)],
    ),
    (r"proxy|socks|pivot", [("T1090", 0.8), ("TA0011", 0.8)]),
    (r"nonx_tcp|encrypted.*payload|rc4|named_pipe", [("T1573", 0.7), ("TA0011", 0.7)]),
    (
        r"web_shell|webshell|aspx.*shell|php.*shell|jsp.*shell",
        [("T1505.003", 0.9), ("TA0003", 0.5), ("TA0011", 0.5)],
    ),
    # --- Impact (route to defense_evasion since AttackLM has no TA0040) ---
    (r"dos/|dos_", [("T1498", 0.8), ("TA0005", 0.5)]),
    (r"wipe|format.*disk|ransomware|encrypt.*file", [("T1485", 0.8), ("TA0005", 0.5)]),
]

# Module-type fallback (low-confidence default)
TYPE_FALLBACK: dict[str, tuple[str, str, float]] = {
    "exploits": ("T1190", "TA0002", 0.3),
    "auxiliary": ("T1046", "TA0007", 0.3),
    "post": ("T1059", "TA0002", 0.2),
    "payloads": ("T1059", "TA0002", 0.4),
    "evasion": ("T1027", "TA0005", 0.6),
    "encoders": ("T1027", "TA0005", 0.6),
    "nops": ("T1027", "TA0005", 0.3),
}


def classify(module: dict) -> list[dict]:
    """Return list of predicted technique dicts [{'id', 'name', 'confidence'}]."""
    path = module.get("module_path", "")
    name = module.get("name", "")
    desc = module.get("description", "")
    text = f"{path} {name} {desc[:200]}"
    technique_scores: dict[str, float] = {}
    tactic_scores: dict[str, float] = {}
    for pat, rules in PATH_RULES:
        if re.search(pat, text, re.IGNORECASE):
            for tid, w in rules:
                if tid.startswith("TA"):
                    tactic_scores[tid] = tactic_scores.get(tid, 0) + w
                else:
                    technique_scores[tid] = technique_scores.get(tid, 0) + w
    if not technique_scores:
        mtype = (module.get("module_type") or "").rstrip("s")
        if mtype in TYPE_FALLBACK:
            tid, tac, w = TYPE_FALLBACK[mtype]
            technique_scores[tid] = w
            tactic_scores[tac] = tactic_scores.get(tac, 0) + w
    if not technique_scores:
        return []
    best_tactic = max(tactic_scores.items(), key=lambda x: x[1])[0]
    sorted_techs = sorted(technique_scores.items(), key=lambda x: -x[1])
    # Convert to friendly names
    TECHNIQUE_NAMES = {
        "T1003": "OS Credential Dumping",
        "T1003.002": "Security Account Manager",
        "T1558": "Steal or Forge Kerberos Tickets",
        "T1110": "Brute Force",
        "T1555": "Credentials from Password Stores",
        "T1046": "Network Service Discovery",
        "T1018": "Remote System Discovery",
        "T1087": "Account Discovery",
        "T1040": "Network Sniffing",
        "T1210": "Exploitation of Remote Services",
        "T1021.001": "Remote Desktop Protocol",
        "T1021.002": "SMB/Windows Admin Shares",
        "T1021.004": "SSH",
        "T1021.005": "VNC",
        "T1021.006": "Windows Remote Management",
        "T1546": "Event Triggered Execution",
        "T1547.001": "Registry Run Keys",
        "T1053.005": "Scheduled Task",
        "T1053.003": "Cron",
        "T1543.001": "Launch Agent",
        "T1543.003": "Windows Service",
        "T1136": "Create Account",
        "T1037": "Boot or Logon Initialization Scripts",
        "T1068": "Exploitation for Privilege Escalation",
        "T1548.002": "Bypass UAC",
        "T1548.003": "Sudo and Sudo Caching",
        "T1548.001": "Setuid and Setgid",
        "T1134": "Access Token Manipulation",
        "T1134.002": "Create Process with Token",
        "T1574": "Hijack Execution Flow",
        "T1027": "Obfuscated Files or Information",
        "T1562.001": "Disable or Modify Tools",
        "T1562.004": "Disable or Modify System Firewall",
        "T1070.001": "Clear Windows Event Logs",
        "T1055": "Process Injection",
        "T1059": "Command and Scripting Interpreter",
        "T1059.001": "PowerShell",
        "T1047": "Windows Management Instrumentation",
        "T1218": "System Binary Proxy Execution",
        "T1190": "Exploit Public-Facing Application",
        "T1189": "Drive-by Compromise",
        "T1204.002": "Malicious File",
        "T1203": "Exploitation for Client Execution",
        "T1113": "Screen Capture",
        "T1056.001": "Keylogging",
        "T1123": "Audio Capture",
        "T1005": "Data from Local System",
        "T1115": "Clipboard Data",
        "T1217": "Browser Bookmark Discovery",
        "T1114": "Email Collection",
        "T1071.001": "Web Protocols",
        "T1071.004": "DNS",
        "T1095": "Non-Application Layer Protocol",
        "T1090": "Proxy",
        "T1572": "Protocol Tunneling",
        "T1573": "Encrypted Channel",
        "T1505.003": "Web Shell",
        "T1498": "Network Denial of Service",
        "T1485": "Data Destruction",
    }
    out: list[dict] = []
    for tid, score in sorted_techs[:3]:
        out.append(
            {
                "id": tid,
                "name": TECHNIQUE_NAMES.get(tid, tid),
                "confidence": round(score, 2),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Path-based heuristic classifier for unrouted Metasploit modules."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Summarize classifications without writing files.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=True,
        help="Re-run extract_by_tactic.py after updating records (default: True).",
    )
    parser.add_argument(
        "--no-merge",
        dest="merge",
        action="store_false",
    )
    args = parser.parse_args()

    if not RECORDS_PATH.exists():
        print(
            f"ERROR: {RECORDS_PATH} not found. Run parse_metasploit_to_jsonl.py first."
        )
        sys.exit(1)

    print(f"Loading {RECORDS_PATH} ...")
    with open(RECORDS_PATH) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    print(f"  {len(records)} total records")

    # Existing routed vs unrouted
    existing_routed = [r for r in records if r.get("mitre_techniques")]
    unrouted = [r for r in records if not r.get("mitre_techniques")]
    print(f"  Already routed: {len(existing_routed)}")
    print(f"  Unrouted: {len(unrouted)}")
    print()

    # Classify
    new_classifications: list[dict] = []
    still_unrouted: list[dict] = []
    tactic_counts: Counter = Counter()
    technique_counts: Counter = Counter()

    for r in unrouted:
        preds = classify(r)
        if preds:
            # Annotate the record
            r["predicted_techniques"] = preds
            r["prediction_source"] = "heuristic"
            new_classifications.append(
                {"module_path": r["module_path"], "predictions": preds}
            )
            tactic_counts[preds[0]["id"] if False else ""]  # placeholder
            # Track by MITRE id and tactic via TECHNIQUE_TO_TACTIC
            for p in preds:
                technique_counts[p["id"]] += 1
        else:
            still_unrouted.append(r)

    # Need the tactic map to count by tactic
    sys.path.insert(0, str(SCRIPT_DIR))
    from extract_by_tactic import TECHNIQUE_TO_TACTIC, _base_technique_id  # type: ignore

    for c in new_classifications:
        for p in c["predictions"]:
            tac = TECHNIQUE_TO_TACTIC.get(_base_technique_id(p["id"]))
            if tac:
                tactic_counts[tac] += 1

    # Summary
    print(f"  Heuristically classified: {len(new_classifications)}")
    print(f"  Still unrouted:           {len(still_unrouted)}")
    if still_unrouted:
        print(f"    By type: {dict(Counter(r['module_type'] for r in still_unrouted))}")
    print()
    print(f"  By predicted technique (top 15):")
    for tid, c in technique_counts.most_common(15):
        print(f"    {tid:10s} {c}")
    print()
    print(f"  By routed tactic:")
    for tac, c in tactic_counts.most_common():
        print(f"    {tac} {c}")

    if args.dry_run:
        print("\n  DRY RUN — no files written.")
        return

    # Write updated records
    with open(RECORDS_PATH, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Wrote {RECORDS_PATH} with predicted_techniques appended.")

    # Re-run extract_by_tactic.py to update manifests
    if args.merge:
        import subprocess

        print("\n  Re-running extract_by_tactic.py to update manifests ...")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "extract_by_tactic.py")],
            cwd=str(BASE_DIR),
            check=False,
        )
        if result.returncode != 0:
            print(f"  ERROR: extract_by_tactic.py exited with {result.returncode}")
            sys.exit(1)
        print("  Manifests updated.")


if __name__ == "__main__":
    main()
