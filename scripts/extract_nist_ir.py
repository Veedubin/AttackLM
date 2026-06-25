#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script generates training pairs from: NIST SP 800-61 Revision 3
# Document: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf
# License:    Public Domain (United States Government work)
# ----------------------------------
"""Deterministic generation of NIST SP 800-61r3 incident response training pairs.

Generates ~200 OpenAI-style message triples covering the four IR phases
(Preparation, Detection & Analysis, Containment/Eradication/Recovery,
Post-Incident Activity) across 8 incident types and 5 asset types.

Output: ``data/datasets/buckets/sources/nist-ir/defensive/incident_response/data.jsonl``

Usage:
    python scripts/extract_nist_ir.py
    python scripts/extract_nist_ir.py --dry-run --max-pairs 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "nist-ir"
    / "defensive"
    / "incident_response"
)
OUTPUT_PATH = OUTPUT_DIR / "data.jsonl"

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are an Incident Response specialist following the NIST SP 800-61 "
    "framework. Provide phase-specific IR procedures including containment, "
    "eradication, recovery, evidence collection, and stakeholder communication."
)

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "nist-sp800-61r3",
    "source_uri": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf",
    "license": "Public-Domain",
    "license_uri": "https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications",
    "rights_contact": "NIST",
    "attribution_text": (
        "Public Domain — United States Government work. "
        "NIST Special Publication 800-61 Revision 3, Computer Security Incident Handling Guide."
    ),
}

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
INCIDENT_TYPES = [
    "Ransomware",
    "Data Breach",
    "DDoS Attack",
    "Insider Threat",
    "Web Application Attack",
    "Phishing Campaign",
    "Malware Outbreak",
    "Credential Theft",
]

ASSET_TYPES = [
    "Domain Controller",
    "File Server",
    "Web Server",
    "Database Server",
    "Workstation",
]

IR_PHASES = [
    "Preparation",
    "Detection and Analysis",
    "Containment, Eradication, and Recovery",
    "Post-Incident Activity",
]

# ---------------------------------------------------------------------------
# Phase-specific response templates
# ---------------------------------------------------------------------------
PHASE_RESPONSES: dict[str, dict[str, str]] = {
    "Ransomware": {
        "Preparation": (
            "## Preparation Phase — Ransomware Incident\n\n"
            "### Pre-Incident Readiness\n"
            "1. **Backup Strategy**: Maintain offline, immutable backups of all critical "
            "systems. Test restoration procedures quarterly. Ensure backups are stored "
            "on isolated networks or air-gapped media.\n"
            "2. **Incident Response Team**: Designate a ransomware-specific IR lead, "
            "legal counsel, and communications coordinator. Establish contact with "
            "law enforcement (FBI IC3, CISA) and cyber insurance provider.\n"
            "3. **Network Segmentation**: Implement network segmentation to limit "
            "lateral movement. Restrict SMB/RDP access between VLANs. Deploy "
            "application allowlisting on critical servers.\n"
            "4. **Detection Capabilities**: Deploy EDR with ransomware-specific "
            "behavioral detections (mass file renames, shadow copy deletion, "
            "encryption patterns). Configure SIEM alerts for unusual file system "
            "activity.\n"
            "5. **Communication Templates**: Prepare internal notification templates, "
            "customer notification drafts, and regulatory reporting forms (GDPR, "
            "HIPAA, state breach notification laws).\n"
            "6. **Tabletop Exercises**: Conduct quarterly ransomware tabletop "
            "exercises covering decision points: pay vs. don't pay, contain vs. "
            "observe, when to invoke disaster recovery."
        ),
        "Detection and Analysis": (
            "## Detection and Analysis Phase — Ransomware Incident\n\n"
            "### Initial Detection\n"
            "1. **Alert Triage**: Investigate EDR alerts for mass file modifications, "
            "shadow copy deletion (vssadmin.exe), and unusual process behavior.\n"
            "2. **User Reports**: Prioritize user reports of inaccessible files, "
            "ransom notes (.txt/.html on desktop), or system performance degradation.\n"
            "3. **Network Monitoring**: Check for unusual SMB traffic spikes, "
            "outbound connections to known ransomware C2 infrastructure.\n\n"
            "### Scope Assessment\n"
            "1. **Identify Patient Zero**: Trace the initial access vector — phishing "
            "email, RDP brute force, exploit kit, or software vulnerability.\n"
            "2. **Map Lateral Movement**: Analyze Windows Event Logs (4624, 4625, "
            "4672, 4768) to determine which accounts and systems were accessed.\n"
            "3. **Determine Encryption Scope**: Identify which file shares, databases, "
            "and systems were encrypted. Check backup integrity.\n"
            "4. **Ransomware Family Identification**: Submit encrypted file samples "
            "and ransom note to ID Ransomware (https://id-ransomware.malwarehunterteam.com) "
            "or No More Ransom Project.\n\n"
            "### Evidence Collection\n"
            "1. Acquire forensic images of affected systems (memory + disk).\n"
            "2. Collect network flow data, firewall logs, and VPN access logs.\n"
            "3. Preserve ransom note, encrypted file samples, and any communication "
            "from threat actors.\n"
            "4. Document timeline of events with UTC timestamps.\n\n"
            "### Stakeholder Communication\n"
            "1. Notify IR lead and CISO immediately upon confirmation.\n"
            "2. Engage legal counsel to assess notification obligations.\n"
            "3. Contact cyber insurance provider per policy requirements.\n"
            "4. Prepare executive summary for leadership."
        ),
        "Containment, Eradication, and Recovery": (
            "## Containment, Eradication, and Recovery Phase — Ransomware Incident\n\n"
            "### Containment Strategy\n"
            "1. **Immediate Actions**: Isolate affected systems from the network "
            "(disable switch ports, revoke VPN access, disable AD accounts).\n"
            "2. **Prevent Spread**: Block outbound C2 traffic at the firewall. "
            "Disable SMBv1 and restrict RDP access. Reset all domain admin passwords.\n"
            "3. **Preserve Evidence**: Do NOT power off affected systems — acquire "
            "memory dumps and disk images first. The encryption key may be recoverable "
            "from memory.\n"
            "4. **Decision Point — Pay Ransom?**: Consult legal counsel and law "
            "enforcement. Consider: availability of backups, sensitivity of encrypted "
            "data, threat actor reputation, regulatory implications.\n\n"
            "### Eradication\n"
            "1. Rebuild affected systems from known-good images or fresh OS installs.\n"
            "2. Remove persistence mechanisms: scheduled tasks, registry Run keys, "
            "WMI event subscriptions, services.\n"
            "3. Rotate all credentials that may have been compromised (domain admin, "
            "service accounts, user passwords).\n"
            "4. Patch the initial access vulnerability.\n\n"
            "### Recovery\n"
            "1. Restore data from clean, verified backups.\n"
            "2. Reconnect systems to the network in phases, monitoring for signs "
            "of reinfection.\n"
            "3. Verify integrity of restored data (hash comparison, application testing).\n"
            "4. Resume normal operations only after 24-48 hours of clean monitoring.\n"
            "5. Conduct post-recovery vulnerability scan of all restored systems."
        ),
        "Post-Incident Activity": (
            "## Post-Incident Activity Phase — Ransomware Incident\n\n"
            "### Lessons Learned Meeting\n"
            "1. Schedule within 2 weeks of incident closure.\n"
            "2. Include: IR team, IT operations, security engineering, legal, "
            "communications, executive sponsor.\n"
            "3. Review: timeline of events, detection speed, containment effectiveness, "
            "communication quality, decision-making process.\n\n"
            "### Root Cause Analysis\n"
            "1. Document the initial access vector and why existing controls failed.\n"
            "2. Identify gaps in detection: why wasn't the attack caught earlier?\n"
            "3. Assess backup strategy: were backups protected? How fast was recovery?\n\n"
            "### Evidence Retention\n"
            "1. Preserve forensic evidence per legal hold requirements.\n"
            "2. Maintain chain of custody documentation.\n"
            "3. Retain incident records per regulatory requirements (typically 3-7 years).\n\n"
            "### Improvement Plan\n"
            "1. Update IR playbook with lessons learned.\n"
            "2. Implement new detection rules based on observed TTPs.\n"
            "3. Schedule additional tabletop exercises for identified gaps.\n"
            "4. Update backup strategy if recovery was slower than RTO.\n"
            "5. Report findings to board/executive leadership."
        ),
    },
    "Data Breach": {
        "Preparation": (
            "## Preparation Phase — Data Breach Incident\n\n"
            "### Pre-Incident Readiness\n"
            "1. **Data Classification**: Maintain current data inventory with "
            "classification levels (PII, PHI, PCI, IP). Know where sensitive data "
            "resides.\n"
            "2. **Breach Response Team**: Designate privacy officer, legal counsel "
            "(breach notification specialist), forensics firm (on retainer), and "
            "communications lead.\n"
            "3. **Regulatory Mapping**: Document applicable breach notification "
            "requirements: GDPR (72 hours), HIPAA (60 days), state laws (varies), "
            "PCI DSS, SEC (4 business days for material incidents).\n"
            "4. **DLP Deployment**: Deploy Data Loss Prevention (DLP) tools on "
            "email, web, and endpoint. Configure alerts for large data exfiltration.\n"
            "5. **Access Review**: Conduct quarterly access reviews for sensitive "
            "data repositories. Implement just-in-time access for privileged accounts.\n"
            "6. **Tabletop Exercises**: Conduct breach simulation exercises covering "
            "notification decision trees, media response, and regulatory reporting."
        ),
        "Detection and Analysis": (
            "## Detection and Analysis Phase — Data Breach Incident\n\n"
            "### Initial Detection\n"
            "1. **DLP Alerts**: Investigate DLP alerts for large data transfers, "
            "unusual destinations, or sensitive data patterns.\n"
            "2. **Database Monitoring**: Check database audit logs for unusual "
            "SELECT queries, mass exports, or privileged account usage.\n"
            "3. **Cloud Monitoring**: Review cloud access logs (AWS CloudTrail, "
            "Azure Monitor) for unusual S3 bucket access, IAM role changes, or "
            "data transfer operations.\n"
            "4. **Dark Web Monitoring**: Check for organization's data appearing "
            "on dark web forums, paste sites, or breach notification services.\n\n"
            "### Scope Assessment\n"
            "1. Identify which data was accessed/exfiltrated (tables, files, buckets).\n"
            "2. Determine number of affected individuals/records.\n"
            "3. Classify data sensitivity (PII, PHI, PCI, trade secrets).\n"
            "4. Identify the exfiltration method (email, cloud storage, FTP, API).\n"
            "5. Determine the time window of exposure.\n\n"
            "### Evidence Collection\n"
            "1. Preserve database audit logs, application logs, and access logs.\n"
            "2. Collect network flow data showing data exfiltration.\n"
            "3. Document the data types and volumes involved.\n"
            "4. Maintain chain of custody for all forensic evidence.\n\n"
            "### Stakeholder Communication\n"
            "1. Notify privacy officer and legal counsel immediately.\n"
            "2. Engage forensics firm if not already on retainer.\n"
            "3. Begin regulatory notification clock tracking.\n"
            "4. Prepare internal and external communication drafts."
        ),
        "Containment, Eradication, and Recovery": (
            "## Containment, Eradication, and Recovery Phase — Data Breach Incident\n\n"
            "### Containment\n"
            "1. Revoke compromised credentials and API keys immediately.\n"
            "2. Block exfiltration channels (firewall rules, proxy blocks).\n"
            "3. Disable compromised accounts and service principals.\n"
            "4. Rotate all secrets that may have been exposed.\n\n"
            "### Eradication\n"
            "1. Remove attacker persistence (backdoor accounts, API keys, OAuth tokens).\n"
            "2. Patch the vulnerability that enabled access.\n"
            "3. Review and harden access controls on affected data stores.\n\n"
            "### Recovery\n"
            "1. Restore access controls to pre-incident state.\n"
            "2. Verify no unauthorized access persists (review recent audit logs).\n"
            "3. Implement additional monitoring on affected data stores.\n"
            "4. Conduct penetration test of the access path used by attacker.\n\n"
            "### Notification\n"
            "1. Determine notification obligations with legal counsel.\n"
            "2. Prepare notification letters per regulatory requirements.\n"
            "3. Notify affected individuals, regulators, and partners as required.\n"
            "4. Establish call center/support for affected individuals if needed."
        ),
        "Post-Incident Activity": (
            "## Post-Incident Activity Phase — Data Breach Incident\n\n"
            "### Lessons Learned\n"
            "1. Review detection speed: how long was data exposed before detection?\n"
            "2. Assess DLP effectiveness: did existing controls detect the exfiltration?\n"
            "3. Evaluate notification process: was it timely and compliant?\n\n"
            "### Root Cause Analysis\n"
            "1. Document the access vector and data exfiltration method.\n"
            "2. Identify why access controls failed.\n"
            "3. Assess data classification accuracy.\n\n"
            "### Improvement Plan\n"
            "1. Implement additional DLP rules based on observed exfiltration patterns.\n"
            "2. Enhance access controls (MFA, JIT, PIM) on sensitive data stores.\n"
            "3. Update data classification and inventory.\n"
            "4. Conduct additional tabletop exercises for breach scenarios.\n"
            "5. Review and update cyber insurance coverage."
        ),
    },
}

# Default responses for incident types without full templates
_DEFAULT_PHASE = {
    "Preparation": (
        "## Preparation Phase\n\n"
        "### Pre-Incident Readiness\n"
        "1. Establish incident response team with defined roles and responsibilities.\n"
        "2. Deploy detection capabilities: SIEM, EDR, network monitoring, DLP.\n"
        "3. Maintain current asset inventory and network diagrams.\n"
        "4. Establish communication templates and escalation procedures.\n"
        "5. Conduct regular tabletop exercises and training.\n"
        "6. Maintain relationships with external partners: forensics firm, "
        "legal counsel, law enforcement, cyber insurance provider."
    ),
    "Detection and Analysis": (
        "## Detection and Analysis Phase\n\n"
        "### Initial Detection\n"
        "1. Investigate alerts from SIEM, EDR, and monitoring systems.\n"
        "2. Correlate indicators across multiple data sources.\n"
        "3. Determine if the event constitutes a security incident.\n\n"
        "### Scope Assessment\n"
        "1. Identify affected systems, accounts, and data.\n"
        "2. Determine the attack vector and timeline.\n"
        "3. Assess business impact and data sensitivity.\n\n"
        "### Evidence Collection\n"
        "1. Preserve logs, forensic images, and network captures.\n"
        "2. Document findings with timestamps and chain of custody.\n"
        "3. Maintain evidence integrity for potential legal action.\n\n"
        "### Stakeholder Communication\n"
        "1. Notify IR lead and CISO.\n"
        "2. Engage legal counsel for privilege and notification assessment.\n"
        "3. Prepare status updates for leadership."
    ),
    "Containment, Eradication, and Recovery": (
        "## Containment, Eradication, and Recovery Phase\n\n"
        "### Containment\n"
        "1. Isolate affected systems to prevent further damage.\n"
        "2. Block attacker infrastructure at network perimeter.\n"
        "3. Preserve evidence before taking destructive actions.\n\n"
        "### Eradication\n"
        "1. Remove malware, backdoors, and persistence mechanisms.\n"
        "2. Patch vulnerabilities exploited in the attack.\n"
        "3. Reset compromised credentials.\n\n"
        "### Recovery\n"
        "1. Restore systems from clean backups.\n"
        "2. Verify system integrity before reconnecting to network.\n"
        "3. Monitor for signs of reinfection for 24-48 hours.\n"
        "4. Return to normal operations after verification."
    ),
    "Post-Incident Activity": (
        "## Post-Incident Activity Phase\n\n"
        "### Lessons Learned\n"
        "1. Conduct post-incident review within 2 weeks.\n"
        "2. Document what worked well and what didn't.\n"
        "3. Identify detection and response gaps.\n\n"
        "### Evidence Retention\n"
        "1. Preserve evidence per legal and regulatory requirements.\n"
        "2. Maintain chain of custody documentation.\n\n"
        "### Improvement Plan\n"
        "1. Update IR playbook with lessons learned.\n"
        "2. Implement new detection rules.\n"
        "3. Schedule follow-up training and exercises.\n"
        "4. Report findings to executive leadership."
    ),
}


# ---------------------------------------------------------------------------
# Generate pairs
# ---------------------------------------------------------------------------
def generate_pairs(max_pairs: int = 0) -> list[dict[str, Any]]:
    """Generate incident response training pairs."""
    random.seed(42)
    pairs = []

    # Phase-based pairs: incident type × asset type × phase
    for incident in INCIDENT_TYPES:
        for asset in ASSET_TYPES:
            for phase in IR_PHASES:
                # Get response text
                if incident in PHASE_RESPONSES and phase in PHASE_RESPONSES[incident]:
                    response = PHASE_RESPONSES[incident][phase]
                else:
                    response = _DEFAULT_PHASE[phase]

                user = (
                    f"A {incident} has been detected on a {asset}. Walk through "
                    f"the NIST SP 800-61 {phase} phase. Include containment steps, "
                    f"evidence collection procedures, and stakeholder communication "
                    f"requirements."
                )

                pair = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": response},
                    ],
                    "mitre_ids": [],
                    **ATTRIBUTION,
                }
                pairs.append(pair)

    # Decision tree pairs
    decision_scenarios = [
        (
            "Ransomware on Domain Controller",
            "contain the DC immediately or observe to gather threat intelligence",
        ),
        (
            "Data Breach involving PII",
            "notify affected individuals immediately or complete forensic investigation first",
        ),
        (
            "DDoS against Web Server",
            "implement rate limiting or engage upstream DDoS mitigation provider",
        ),
        (
            "Insider Threat with Admin Access",
            "revoke access immediately or monitor to gather evidence",
        ),
        (
            "Phishing Campaign targeting Executives",
            "reset all executive passwords or investigate which accounts were compromised first",
        ),
    ]

    for scenario, decision in decision_scenarios:
        user = (
            f"You are responding to a {scenario}. Based on NIST SP 800-61, what "
            f"is the appropriate escalation path? What factors determine whether "
            f"to {decision}?"
        )

        assistant = (
            f"## Decision Analysis: {scenario}\n\n"
            f"### Factors to Consider\n"
            f"1. **Business Impact**: What is the operational impact of each option? "
            f"Can the organization tolerate the downtime or data exposure?\n"
            f"2. **Evidence Preservation**: Will immediate action destroy forensic "
            f"evidence needed for investigation or prosecution?\n"
            f"3. **Regulatory Requirements**: Do breach notification laws require "
            f"immediate action? Is there a mandated timeline?\n"
            f"4. **Threat Actor Behavior**: Is the adversary likely to escalate "
            f"if they detect response actions?\n"
            f"5. **Recovery Capability**: Do you have verified backups? How long "
            f"would recovery take?\n\n"
            f"### Recommended Escalation Path\n"
            f"1. **Immediate**: Notify IR lead and CISO. Engage legal counsel.\n"
            f"2. **Short-term** (0-2 hours): Contain the most critical impact "
            f"while preserving evidence where possible.\n"
            f"3. **Medium-term** (2-24 hours): Complete forensic investigation, "
            f"eradicate threat, begin recovery.\n"
            f"4. **Long-term** (24+ hours): Full recovery, notification, "
            f"post-incident review.\n\n"
            f"### Decision Framework\n"
            f"Use the NIST SP 800-61 decision tree: if containment will destroy "
            f"critical evidence AND the threat is not actively causing damage, "
            f"monitor and collect evidence first. If the threat is actively "
            f"causing damage (data exfiltration, encryption, lateral movement), "
            f"contain immediately regardless of evidence impact."
        )

        pair = {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "mitre_ids": [],
            **ATTRIBUTION,
        }
        pairs.append(pair)

    # Cross-phase pairs
    phase_pairs = [
        ("Preparation", "Detection and Analysis"),
        ("Detection and Analysis", "Containment, Eradication, and Recovery"),
        ("Containment, Eradication, and Recovery", "Post-Incident Activity"),
    ]

    for phase_a, phase_b in phase_pairs:
        user = (
            f"Compare the NIST SP 800-61 {phase_a} and {phase_b} phases. What "
            f"handoff artifacts should be produced? What decisions made in "
            f"{phase_a} constrain options in {phase_b}?"
        )

        assistant = (
            f"## Phase Comparison: {phase_a} → {phase_b}\n\n"
            f"### Handoff Artifacts from {phase_a}\n"
            f"1. **Incident Timeline**: Complete chronological record of all "
            f"events, actions, and decisions with UTC timestamps.\n"
            f"2. **Evidence Inventory**: Catalog of all collected evidence with "
            f"chain of custody documentation.\n"
            f"3. **Scope Assessment**: Documented list of affected systems, "
            f"accounts, and data.\n"
            f"4. **Containment Status**: Current state of containment measures "
            f"and any systems still isolated.\n"
            f"5. **Stakeholder Notifications**: Record of who was notified, when, "
            f"and what information was shared.\n\n"
            f"### Constraining Decisions from {phase_a}\n"
            f"1. **Containment Choices**: Systems that were isolated in {phase_a} "
            f"may have lost forensic evidence needed for {phase_b} analysis.\n"
            f"2. **Evidence Collection Scope**: If evidence collection was "
            f"incomplete in {phase_a}, {phase_b} may lack critical data for "
            f"root cause analysis.\n"
            f"3. **Communication Commitments**: Promises made to stakeholders "
            f"in {phase_a} (e.g., 'we will notify within 24 hours') create "
            f"deadlines that constrain {phase_b} activities.\n"
            f"4. **Resource Allocation**: Resources committed in {phase_a} "
            f"may not be available for {phase_b} tasks.\n\n"
            f"### Best Practices\n"
            f"1. Document all decisions with rationale in {phase_a} so {phase_b} "
            f"teams understand the context.\n"
            f"2. Hold a formal handoff meeting between phase leads.\n"
            f"3. Use a shared incident management platform (TheHive, Jira, "
            f"ServiceNow) for continuity.\n"
            f"4. Assign a liaison who spans both phases to ensure continuity."
        )

        pair = {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "mitre_ids": [],
            **ATTRIBUTION,
        }
        pairs.append(pair)

    # Shuffle and limit
    random.shuffle(pairs)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate NIST SP 800-61r3 incident response training pairs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print pairs without writing"
    )
    parser.add_argument(
        "--max-pairs", type=int, default=0, help="Max pairs to generate (0=all)"
    )
    args = parser.parse_args()

    pairs = generate_pairs(args.max_pairs)
    print(f"Generated {len(pairs)} incident response pairs", file=sys.stderr)

    if args.dry_run:
        for pair in pairs[:5]:
            print(json.dumps(pair, indent=2))
        print(f"\n... and {len(pairs) - 5} more pairs", file=sys.stderr)
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Wrote {len(pairs)} pairs to {OUTPUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
