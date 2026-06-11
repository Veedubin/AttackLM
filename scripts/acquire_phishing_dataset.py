#!/usr/bin/env python3
"""acquire_phishing_dataset.py — Generate synthetic phishing/social engineering
training pairs for AttackLM bucket format.

Covers: spear phishing, BEC, credential harvesting, vishing, pretexting, deepfake SE.

Output: data/datasets/buckets/social_engineering/phishing/data_synth.jsonl
        data/datasets/buckets/social_engineering/phishing/metadata.json

Usage:
    python scripts/acquire_phishing_dataset.py
    python scripts/acquire_phishing_dataset.py --count 10
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BUCKET_DIR = (
    BASE_DIR / "data" / "datasets" / "buckets" / "social_engineering" / "phishing"
)
DATA_FILE = BUCKET_DIR / "data_synth.jsonl"
META_FILE = BUCKET_DIR / "metadata.json"

SEED = 42
SOURCE_NAME = "phishing_synthetic"
LICENSE = "MIT"
SYSTEM_PROMPT = (
    "You are an authorized Red Team SOCIAL ENGINEERING specialist. "
    "You provide precise adversary emulation techniques for security validation."
)

ENVIRONMENT_TYPES = [
    "corporate environment",
    "remote work setting",
    "financial institution",
    "healthcare organization",
    "government agency",
    "educational institution",
    "technology company",
    "manufacturing firm",
    "retail organization",
    "small business",
]

SEVERITY_LEVELS = ["critical", "high", "medium", "low"]

TESTING_FRAMEWORKS = [
    "MITRE ATT&CK",
    "OWASP Social Engineering Guide",
    "NIST SP 800-115",
    "PTES",
    "OSSTMM",
    "SEVT (Social Engineering Vulnerability Testing)",
]

SCENARIOS: list[dict] = [
    # === SPEAR PHISHING ===
    {
        "category": "spear_phishing",
        "name": "Executive Spear-Phishing (Whaling)",
        "mitre_ids": ["T1566.001"],
        "desc": "Targeted phishing attack against C-suite executives using personalized lures.",
        "steps": "1. Reconnaissance: Gather executive info from LinkedIn, SEC filings, press releases\n2. Email crafting: Use executive's name, recent events, and industry terminology\n3. Sender spoofing: Register lookalike domain (e.g., c0mpany.com)\n4. Payload: Malware-laden PDF 'Q3 Board Presentation.pdf' or credential harvesting link\n5. Delivery: Send during business hours with urgent subject line\n6. Follow-up: Call posing as IT to verify email was received",
        "email_templates": [
            "Subject: Urgent: Q3 Financial Review Required\n\nDear {name},\n\nAttached is the updated Q3 financial report that requires your review before tomorrow's board meeting. Please review and confirm receipt.\n\nBest regards,\n{spoofed_sender}\nCFO Office",
            "Subject: Confidential: M&A Discussion\n\n{name},\n\nPer our discussion, I've prepared the confidential materials regarding the acquisition target. Please access the secure document portal at the link below.\n\n{phishing_link}\n\nThis link expires in 24 hours.\n\nRegards,\n{spoofed_sender}",
        ],
        "detection": [
            "Email authentication failures (SPF/DKIM/DMARC)",
            "Lookalike domain registration",
            "Urgency + authority social proof patterns",
            "Attachment analysis (malware signatures)",
        ],
        "mitigations": [
            "DMARC enforcement (p=reject)",
            "Executive security awareness training",
            "Email authentication (SPF, DKIM)",
            "Lookalike domain monitoring",
            "Attachment sandboxing",
        ],
    },
    {
        "category": "spear_phishing",
        "name": "Employee Credential Harvesting",
        "mitre_ids": ["T1566.001"],
        "desc": "Phishing campaign targeting employees with fake login portals to steal credentials.",
        "steps": "1. Set up credential harvesting infrastructure: Evilginx2 or GoPhish\n2. Clone target login page: `httrack <target_login_url> && python3 server.py`\n3. Configure Evilginx2: `evilginx2 -c phishing.yaml` with SSL proxy\n4. Craft phishing email with pretext (IT password reset, bonus notification)\n5. Deploy URL shortener: `bit.ly/<custom>` or custom domain\n6. Monitor credential capture and session cookies\n7. Use captured cookies for session hijacking",
        "email_templates": [
            "Subject: Action Required: Password Reset\n\nDear Employee,\n\nWe've detected unusual activity on your account. Please reset your password immediately using the link below:\n\n{phishing_link}\n\nIf you don't reset within 24 hours, your account will be locked.\n\nIT Security Team",
            "Subject: Your Annual Bonus Statement\n\nHi {name},\n\nYour annual bonus statement is now available. Please log in to the HR portal to view your compensation details.\n\n{phishing_link}\n\nHR Department",
        ],
        "detection": [
            "Login page anomalies (URL, certificate, content)",
            "Phishing URL in email body",
            "Session cookie capture in proxy logs",
            "Multiple failed authentication attempts",
        ],
        "mitigations": [
            "MFA enforcement",
            "FIDO2/WebAuthn hardware tokens",
            "Conditional Access policies",
            "Phishing-resistant authentication",
            "Email authentication enforcement",
        ],
    },
    {
        "category": "spear_phishing",
        "name": "Cloud Service Phishing",
        "mitre_ids": ["T1566.001"],
        "desc": "Phishing targeting cloud service credentials (O365, Google Workspace, AWS Console).",
        "steps": "1. Identify target cloud services: check MX records for O365, SPF for Google\n2. Set up Evilginx2 with cloud service proxy: configure for login.microsoftonline.com\n3. Craft email mimicking cloud service notification: 'Your storage is full', 'New shared document'\n4. Deploy with sender domain mimicking the cloud provider\n5. Capture session cookies including refresh tokens\n6. Use stolen session to access email, OneDrive, Teams, etc.",
        "email_templates": [
            "Subject: Shared Document: Q4 Planning.xlsx\n\n{name} has shared a document with you.\n\nClick here to view: {phishing_link}\n\nMicrosoft Office 365",
            "Subject: Your Google Workspace Storage is Almost Full\n\nYour Google Workspace storage is 95% full. Please review and manage your storage.\n\n{phishing_link}\n\nGoogle Workspace Team",
        ],
        "detection": [
            "OAuth token anomalies",
            "Impossible travel sign-ins",
            "Session token replay detection",
            "Cloud provider phishing reports",
        ],
        "mitigations": [
            "Conditional Access with device compliance",
            "MFA with number matching",
            "Token protection policies",
            "Cloud app security monitoring",
        ],
    },
    {
        "category": "spear_phishing",
        "name": "Social Media Spear-Phishing",
        "mitre_ids": ["T1598.002"],
        "desc": "Targeting employees through social media platforms with personalized lures.",
        "steps": "1. Create fake social media profile: professional-looking, industry-relevant\n2. Build rapport: engage with target's posts, share relevant content\n3. Move to DM: share 'interesting article' with embedded malicious link\n4. Weaponize: link leads to credential harvesting page or malware download\n5. Follow up: maintain conversation to avoid suspicion\n6. Escalate: use gathered information for secondary attacks",
        "email_templates": [],
        "detection": [
            "New social media connections followed by security incidents",
            "DMs with shortened URLs",
            "Fake professional profiles",
            "Social media profile analysis (low follower count, recent creation)",
        ],
        "mitigations": [
            "Social media awareness training",
            "Link scanning in DMs",
            "Profile verification procedures",
            "Social media monitoring for executive impersonation",
        ],
    },
    {
        "category": "spear_phishing",
        "name": "Attachment-Based Phishing",
        "mitre_ids": ["T1566.001"],
        "desc": "Delivering malware through email attachments (malicious documents, archives, executables).",
        "steps": "1. Create malicious document: embed VBA macro in .docm or .xlsm\n2. Macro payload: PowerShell downloader that fetches and executes stage 2\n3. Alternatively: LNK file with embedded command, ISO with sideloaded DLL\n4. Social engineering: 'Updated invoice', 'shipping notification', 'HR policy update'\n5. Evade AV: use obfuscation, encrypted archives, or LOLBIN execution\n6. Deliver via phishing email with convincing pretext",
        "email_templates": [
            "Subject: Invoice #{invoice_num} - Payment Required\n\nDear {name},\n\nPlease find attached the updated invoice for services rendered. Payment is due within 30 days.\n\nAttachment: Invoice_{invoice_num}.docm\n\nAccounts Receivable",
            "Subject: Updated HR Policy - Action Required\n\nDear {name},\n\nPlease review the attached updated HR policy document. Acknowledgment is required by end of week.\n\nAttachment: HR_Policy_Update_2025.docm\n\nHuman Resources",
        ],
        "detection": [
            "Macro-enabled attachments (.docm, .xlsm)",
            "LNK and ISO attachments",
            "Password-protected archives",
            "Attachment hash reputation",
        ],
        "mitigations": [
            "Block macro-enabled attachments at email gateway",
            "Sandbox all attachments",
            "Disable macros in Office products",
            "Application whitelisting",
        ],
    },
    {
        "category": "spear_phishing",
        "name": "CEO Impersonation with Deepfake Voice Verification",
        "mitre_ids": ["T1566.001"],
        "desc": "Spear phishing targeting executives with a follow-up deepfake voice call to establish legitimacy.",
        "steps": "1. Recon: identify CEO communication patterns, recent travels, meeting schedules\n2. Register lookalike domain and craft whaling email requesting urgent action\n3. Obtain CEO voice samples from earnings calls, keynotes, interviews (15+ minutes)\n4. Train voice clone using ElevenLabs or Resemble AI with executive speaking patterns\n5. Send phishing email with urgent financial request\n6. Follow up with deepfake voice call to 'confirm' the email request\n7. Voicemail + email combination significantly increases trust and compliance",
        "email_templates": [
            "Subject: Time-Sensitive Wire Transfer - Confirmation to Follow\n\n{name},\n\nI need an urgent wire transfer processed for the acquisition. I will call you shortly to confirm the details — please do not discuss with anyone else until after the deal closes.\n\nAccount details to follow on call.\n\n{spoofed_sender}",
        ],
        "detection": [
            "Email followed by suspicious phone call",
            "Voice quality artifacts in follow-up call",
            "Lookalike domain in initial email",
            "Urgency + secrecy pattern across communication channels",
        ],
        "mitigations": [
            "Out-of-band verification protocol for financial requests",
            "Codeword system for executive communications",
            "Deepfake voice awareness training",
            "Multi-channel verification before large transfers",
        ],
    },
    {
        "category": "spear_phishing",
        "name": "Spear Phishing via SMS (Smishing)",
        "mitre_ids": ["T1566.001"],
        "desc": "Targeted SMS phishing using personalized text messages to steal credentials or install malware.",
        "steps": "1. Gather target phone numbers from data brokers, LinkedIn, company directories\n2. Register SMS sender ID or use bulk SMS service (Twilio, etc.)\n3. Craft short, urgent message: 'Unusual login detected. Verify: {link}'\n4. Shorten URL to reduce suspicion on mobile screens\n5. Deploy during business hours when targets check phones frequently\n6. Mobile phishing page optimized for small screens — harder to verify URLs\n7. Capture credentials, MFA codes, or deliver mobile malware",
        "email_templates": [],
        "detection": [
            "SMS with shortened URLs from unknown numbers",
            "Credential entry on mobile-optimized phishing pages",
            "Unexpected SMS from 'IT' or 'security'",
            "Spikes in MFA prompt fatigue after SMS campaigns",
        ],
        "mitigations": [
            "SMS phishing awareness training",
            "MFA with phishing-resistant tokens (FIDO2)",
            "Mobile device management policies",
            "SMS filtering and URL scanning",
        ],
    },
    {
        "category": "spear_phishing",
        "name": "Spear Phishing with Malicious OneDrive Link",
        "mitre_ids": ["T1566.001"],
        "desc": "Phishing using legitimate OneDrive/SharePoint sharing infrastructure to bypass email security.",
        "steps": "1. Create Microsoft 365 trial tenant or compromise existing account\n2. Upload malicious file (HTML with credential harvesting, macro-enabled doc) to OneDrive\n3. Use legitimate Microsoft sharing link — passes SPF/DKIM/DMARC checks\n4. Craft email appearing to be from colleague: 'Shared document for review'\n5. Target clicks link — lands on actual Microsoft domain, sees file\n6. File executes payload or redirects to credential harvesting page\n7. Legitimate infrastructure bypasses most email gateway checks",
        "email_templates": [
            "Subject: {spoofed_sender} shared a document with you\n\n{name} shared a file with you via OneDrive.\n\nOpen file: {phishing_link}\n\nMicrosoft OneDrive",
        ],
        "detection": [
            "OneDrive/SharePoint sharing links to unknown external tenants",
            "Shared files from recently created accounts",
            "Anomalous sharing patterns (mass shares, external recipients)",
            "File type mismatches in shared documents",
        ],
        "mitigations": [
            "Conditional Access policies for shared links",
            "External sharing restrictions in M365",
            "Safe Links and Safe Attachments policies",
            "Cross-tenant sharing controls",
        ],
    },
    {
        "category": "spear_phishing",
        "name": "Spear Phishing with LinkedIn-Derived Lures",
        "mitre_ids": ["T1566.001"],
        "desc": "Crafting phishing emails using detailed LinkedIn reconnaissance to create highly personalized lures.",
        "steps": "1. OSINT: scrape target's LinkedIn for connections, skills, certifications, group memberships\n2. Identify professional context: recent job change, certification, project mention\n3. Craft lure referencing specific LinkedIn activity: 'Based on your interest in {topic}'\n4. Send from lookalike domain of a connection or professional organization\n5. Link leads to credential harvesting page themed around the referenced topic\n6. Follow up via LinkedIn message to reinforce legitimacy",
        "email_templates": [
            "Subject: Invitation: {conference} Speaker Network\n\nHi {name},\n\nBased on your expertise in {skill}, I'd like to invite you to join our professional network. Several of your LinkedIn connections are already members.\n\nComplete your profile here: {phishing_link}\n\nBest regards,\nConference Organizing Committee",
        ],
        "detection": [
            "Email referencing specific LinkedIn activity",
            "Lookalike domains of professional organizations",
            "Phishing pages themed around professional events",
            "Correlation between LinkedIn activity and phishing emails",
        ],
        "mitigations": [
            "LinkedIn privacy settings (restrict visible info)",
            "Professional network verification procedures",
            "Email gateway with social context awareness",
            "Security awareness for personalized lures",
        ],
    },
    {
        "category": "spear_phishing",
        "name": "Spear Phishing Targeting Remote Workers (VPN Credential Theft)",
        "mitre_ids": ["T1566.001"],
        "desc": "Phishing targeting remote workers with fake VPN login pages to steal VPN credentials.",
        "steps": "1. Identify VPN portal: check target's career pages, job postings for VPN info\n2. Clone VPN login page: Fortinet, Cisco AnyConnect, Palo Alto GlobalProtect\n3. Set up Evilginx2 with VPN phishlet for session cookie capture\n4. Craft email: 'VPN upgrade required', 'New VPN client version available'\n5. Time delivery for Monday morning (common VPN reconnection time)\n6. Capture VPN credentials and session tokens\n7. Use stolen session to access internal network",
        "email_templates": [
            "Subject: Mandatory VPN Client Update - Action Required\n\nDear {name},\n\nOur IT team has released a critical security update for the VPN client. Please update immediately to maintain remote access.\n\nDownload the update and authenticate here: {phishing_link}\n\nVPN access will be suspended for non-compliant clients starting Friday.\n\nIT Infrastructure Team",
        ],
        "detection": [
            "VPN credential use from anomalous locations",
            "Phishing URLs mimicking VPN portals",
            "Multiple VPN authentication failures",
            "Session tokens from unknown devices",
        ],
        "mitigations": [
            "Certificate-based VPN authentication",
            "Device posture checks before VPN connection",
            "Phishing-resistant MFA for VPN",
            "Endpoint integrity verification",
        ],
    },
    {
        "category": "spear_phishing",
        "name": "Spear Phishing with QR Code in Email",
        "mitre_ids": ["T1566.001"],
        "desc": "Embedding QR codes in phishing emails to redirect mobile users to credential harvesting pages.",
        "steps": "1. Create credential harvesting page optimized for mobile browsers\n2. Generate QR code encoding phishing URL: use qrterminal or pyqrcode\n3. Embed QR code image in professionally formatted email\n4. Craft pretext: 'Scan for two-factor enrollment', 'Mobile-optimized portal'\n5. QR code bypasses email URL scanners (URL is embedded in image)\n6. Victim scans with mobile camera — URL difficult to verify on small screen\n7. Capture credentials on mobile-optimized phishing page",
        "email_templates": [
            "Subject: Mobile-Optimized Security Enrollment\n\nDear {name},\n\nPlease enroll your mobile device for enhanced security verification. Scan the QR code below with your phone camera to begin.\n\n[QR CODE IMAGE]\n\nThis enrollment is mandatory by end of week.\n\nIT Security Operations",
        ],
        "detection": [
            "QR codes embedded in emails",
            "Mobile-optimized phishing pages",
            "URLs embedded in images bypassing text scanners",
            "MFA enrollment anomalies after QR scans",
        ],
        "mitigations": [
            "QR code scanning policies (verify URL before opening)",
            "Email gateway image analysis for embedded QR codes",
            "Phishing-resistant MFA (FIDO2)",
            "Mobile browser security extensions",
        ],
    },
    # === BEC (Business Email Compromise) ===
    {
        "category": "bec",
        "name": "CEO Fraud / Business Email Compromise",
        "mitre_ids": ["T1566.002"],
        "desc": "Impersonating a CEO or executive to authorize fraudulent wire transfers.",
        "steps": "1. Research target: Identify CEO, CFO, and finance team from LinkedIn, website\n2. Register lookalike domain: ce0-name.com or c0mpany.com\n3. Set up email with display name matching CEO: 'CEO Name <ceo@l00kalike.com>'\n4. Craft urgent wire transfer request to CFO or accounts payable\n5. Create email thread to build legitimacy\n6. Provide 'new' bank account details for payment redirect\n7. Follow up with phone call if needed",
        "email_templates": [
            "Subject: Urgent Wire Transfer\n\nHi {cfo_name},\n\nI need you to process a wire transfer for $250,000 to a new vendor. This is time-sensitive and I'm in meetings all day.\n\nWire details:\nAccount: {fraud_account}\nRouting: {fraud_routing}\n\nPlease handle this immediately.\n\n{ceo_name}",
            "Subject: Confidential Acquisition Payment\n\n{cfo_name},\n\nAs discussed, please wire the acquisition payment to our new escrow account. This is strictly confidential.\n\nAccount details attached.\n\n{ceo_name}",
        ],
        "detection": [
            "Urgent payment requests from executives",
            "New bank account details",
            "Lookalike domain email analysis",
            "Unusual transfer amounts or destinations",
        ],
        "mitigations": [
            "Dual authorization for wire transfers",
            "Out-of-band verification for payment changes",
            "Executive email monitoring",
            "Domain similarity alerts",
            "DMARC enforcement",
        ],
    },
    {
        "category": "bec",
        "name": "Vendor Email Compromise",
        "mitre_ids": ["T1566.002"],
        "desc": "Compromising vendor email accounts to redirect legitimate payments.",
        "steps": "1. Compromise vendor email: credential stuffing, phishing, or password spray\n2. Monitor communications for invoicing patterns\n3. Create email rule to hide forwarded messages\n4. Send invoice with updated bank details to customer\n5. Redirect payment to attacker-controlled account\n6. Maintain communication to delay detection",
        "email_templates": [
            "Subject: Updated Payment Information - Invoice #{invoice_num}\n\nDear Customer,\n\nWe have updated our banking information. Please update your records and use the following account for all future payments:\n\nNew Bank: {fraud_bank}\nNew Account: {fraud_account}\nNew Routing: {fraud_routing}\n\nThank you for your continued partnership.\n\nVendor Finance Team",
        ],
        "detection": [
            "Vendor bank detail changes",
            "Email rule anomalies",
            "Login from unusual locations",
            "Delayed email forwarding rules",
        ],
        "mitigations": [
            "Verify bank detail changes via phone",
            "Multi-channel payment verification",
            "Email anomaly detection",
            "Vendor email monitoring",
        ],
    },
    {
        "category": "bec",
        "name": "Vendor Email Compromise with Document Sharing",
        "mitre_ids": ["T1566.002"],
        "desc": "Compromising vendor accounts and using legitimate document-sharing platforms to distribute malicious links.",
        "steps": "1. Compromise vendor email via credential stuffing or phishing\n2. Monitor email for document-sharing patterns (SharePoint, Dropbox, Google Drive)\n3. Create malicious document on legitimate sharing platform\n4. Send email from compromised account with sharing link to malicious document\n5. Legitimate domain and sender bypass email gateway checks\n6. Document contains embedded phishing link or macro payload\n7. Maintain access by hiding forwarded emails with inbox rules",
        "email_templates": [
            "Subject: Updated Contract - Review Required\n\nHi {name},\n\nPer our conversation, I've uploaded the revised contract to our shared workspace. Please review and sign.\n\nAccess document: {phishing_link}\n\nLet me know if you have any questions.\n\n{spoofed_sender}\nVendor Account Management",
        ],
        "detection": [
            "Document sharing links from compromised vendors",
            "Unusual file types in shared documents",
            "New inbox rules on vendor accounts",
            "Anomalous login locations for vendor accounts",
        ],
        "mitigations": [
            "Verify shared documents via separate channel",
            "Document sharing link inspection protocols",
            "Inbox rule auditing on business accounts",
            "Conditional Access for document platforms",
        ],
    },
    {
        "category": "bec",
        "name": "BEC Targeting HR/Payroll (W-2 Theft)",
        "mitre_ids": ["T1566.002"],
        "desc": "Business email compromise targeting HR and payroll departments to steal employee W-2 tax forms.",
        "steps": "1. Identify HR and payroll staff from LinkedIn, company website\n2. Compromise or spoof executive email: 'CEO requesting employee W-2 data for audit'\n3. Send email to HR/payroll during tax season (Jan-Apr) for legitimacy\n4. Request all employees' W-2 forms: names, SSNs, salaries, withholdings\n5. Use stolen W-2 data for identity theft and tax fraud\n6. File fraudulent tax returns before employees",
        "email_templates": [
            "Subject: Employee W-2 Data Request - IRS Audit\n\nDear HR Team,\n\nWe received an IRS audit notice and need to compile all employee W-2 data immediately. Please send the complete W-2 records for all employees to this address.\n\nThis is time-sensitive and must be completed by end of day.\n\n{ceo_name}\nCEO",
        ],
        "detection": [
            "Requests for bulk employee tax data",
            "W-2 data requests outside normal payroll processing",
            "Executive email requesting HR data directly",
            "Tax season spike in W-2 related requests",
        ],
        "mitigations": [
            "W-2 data requests require dual approval",
            "Out-of-band verification for tax data requests",
            "HR awareness training on BEC targeting",
            "SSN encryption and access controls",
        ],
    },
    {
        "category": "bec",
        "name": "BEC with Attorney Impersonation",
        "mitre_ids": ["T1566.002"],
        "desc": "Impersonating attorneys or legal counsel to authorize payments or obtain confidential information.",
        "steps": "1. Research target company's law firm from press releases, court filings\n2. Register lookalike domain of the law firm\n3. Send email from 'attorney' requesting payment for legal settlement\n4. Create urgency: 'Court deadline', 'Settlement expires', 'Confidential matter'\n5. Request wire transfer to 'escrow account' or 'trust account'\n6. Use legal jargon and confidentiality to prevent victim from verifying",
        "email_templates": [
            "Subject: CONFIDENTIAL - Settlement Payment Due\n\nDear {name},\n\nI am writing on behalf of {law_firm} regarding a confidential settlement matter. Per our discussion with {ceo_name}, payment must be wired to the escrow account by end of business today.\n\nDue to attorney-client privilege, this matter should not be discussed with others in your organization.\n\nEscrow Account: {fraud_account}\nRouting: {fraud_routing}\n\n{attorney_name}\n{law_firm}",
        ],
        "detection": [
            "Legal settlement payment requests via email",
            "Attorney requests for wire transfers",
            "Confidentiality clauses preventing verification",
            "Lookalike domains of law firms",
        ],
        "mitigations": [
            "Verify legal payment requests with known attorney",
            "Never wire based on email alone — phone verification required",
            "Legal counsel verification protocol",
            "Training on attorney impersonation BEC",
        ],
    },
    {
        "category": "bec",
        "name": "BEC Gift Card Scam",
        "mitre_ids": ["T1566.002"],
        "desc": "Impersonating executives to request gift card purchases, typically targeting lower-level employees.",
        "steps": "1. Spoof or compromise executive email account\n2. Target employees who may not question executive requests (new hires, junior staff)\n3. Send email: 'I need you to buy gift cards for employee rewards — I'm in a meeting'\n4. Request specific gift cards: Amazon, Apple, Google Play, Visa\n5. Ask employee to scratch off and send photos of card codes\n6. Cash out gift cards immediately or resell on secondary market\n7. Total losses typically $500-$5,000 per incident",
        "email_templates": [
            "Subject: Quick Favor Needed\n\nHi {name},\n\nI need a quick favor. I'm in meetings all day and need to get some gift cards for employee recognition. Can you purchase the following:\n\n- 5 x $100 Amazon gift cards\n- 3 x $100 Apple gift cards\n\nPlease scratch off and send me the codes as soon as possible. I'll reimburse you tomorrow.\n\nThanks,\n{ceo_name}",
        ],
        "detection": [
            "Gift card purchase requests via email",
            "Executive requesting gift cards for 'employee rewards'",
            "Urgent requests with 'I'm in meetings' excuse",
            "Requests to send card codes via email or chat",
        ],
        "mitigations": [
            "Gift card purchase policy requiring manager approval",
            "Training on gift card scam patterns",
            "Report gift card requests to security team",
            "Anti-phishing email rules for gift card keywords",
        ],
    },
    {
        "category": "bec",
        "name": "BEC with Domain Impersonation",
        "mitre_ids": ["T1566.002"],
        "desc": "Business email compromise using typosquatted or homoglyph domains to impersonate legitimate business contacts.",
        "steps": "1. Identify target organization's domain\n2. Register typosquatted domain: swap letters (c0mpany), add characters (companyy), or use homoglyphs (cοmpany with Greek omicron)\n3. Set up email infrastructure on lookalike domain with valid SPF/DKIM/DMARC\n4. Create email with display name matching legitimate contact\n5. Send payment redirect or information request\n6. Use homoglyph domains that are nearly indistinguishable from legitimate ones\n7. Monitor for responses to maintain cover",
        "email_templates": [
            "Subject: Re: Invoice Payment Confirmation\n\nHi {name},\n\nThank you for the payment on Invoice #{invoice_num}. However, we've had a banking change. Please use the following updated details for all future payments:\n\nBank: {fraud_bank}\nAccount: {fraud_account}\nRouting: {fraud_routing}\n\nPlease confirm receipt.\n\nVendor Finance Team",
        ],
        "detection": [
            "Typosquatted domain registration (domain monitoring)",
            "Homoglyph character detection in email addresses",
            "Sender domain changes in existing email threads",
            "Payment detail change requests",
        ],
        "mitigations": [
            "Domain monitoring for typosquatted registrations",
            "DMARC enforcement (p=reject)",
            "Visual email address inspection training",
            "Payment detail change verification via phone",
        ],
    },
    # === CREDENTIAL HARVESTING ===
    {
        "category": "credential_harvesting",
        "name": "OAuth Phishing (Illicit Consent Grant)",
        "mitre_ids": ["T1566.001"],
        "desc": "Tricking users into granting OAuth permissions to malicious applications.",
        "steps": "1. Register malicious Azure AD application: create multi-tenant app\n2. Configure requested permissions: Mail.Read, Files.Read, User.Read\n3. Craft phishing URL: `https://login.microsoftonline.com/organizations/v2.0/authorize?client_id={malicious_app_id}&response_type=code&redirect_uri={attacker_url}&scope=Mail.Read+Files.Read`\n4. Send phishing email pretending to be from IT: 'Approve required app access'\n5. User clicks, grants consent, attacker receives refresh token\n6. Use refresh token to access victim's email and files via Microsoft Graph API",
        "email_templates": [
            "Subject: Action Required: Approve Application Access\n\nDear User,\n\nYour organization requires you to approve access for the 'Microsoft Teams Update' application. Please click below to approve:\n\n{phishing_link}\n\nThis is required for continued Teams functionality.\n\nIT Department",
        ],
        "detection": [
            "OAuth consent grants to unknown applications",
            "Application permissions audit",
            "Multi-tenant app registrations",
            "Graph API access from unknown clients",
        ],
        "mitigations": [
            "Admin consent workflow for all apps",
            "Application registration restrictions",
            "Conditional Access for OAuth flows",
            "Regular OAuth consent audit",
        ],
    },
    {
        "category": "credential_harvesting",
        "name": "MFA Bypass Phishing (Evilginx2)",
        "mitre_ids": ["T1566.001"],
        "desc": "Phishing that captures MFA tokens by proxying the entire authentication flow.",
        "steps": "1. Install Evilginx2: `docker run -p 443:443 -p 80:80 evilginx2`\n2. Configure phishlet for target: `config domain phishing.com`\n3. Set up SSL: `config cert {domain}` (auto-generate)\n4. Create campaign: `campaign create {name} {phishlet}`\n5. Generate lure URL: `lure create {campaign} {phishlet}`\n6. Send phishing email with lure URL\n7. Victim authenticates through proxy — credentials AND session cookies captured\n8. Use stolen session cookies for persistent access (bypasses MFA)",
        "email_templates": [
            "Subject: Important: Secure Your Account\n\nWe've detected a potential security issue with your account. Please verify your identity immediately:\n\n{phishing_link}\n\nIf you don't verify within 24 hours, your account will be suspended.\n\nSecurity Team",
        ],
        "detection": [
            "Session cookie anomalies in authentication logs",
            "Impossible travel sign-in patterns",
            "New device/location for existing sessions",
            "TLS certificate pinning failures",
        ],
        "mitigations": [
            "FIDO2/WebAuthn hardware tokens (phishing-resistant)",
            "Conditional Access with device trust",
            "Session token binding to device",
            "Phishing detection in email gateway",
        ],
    },
    {
        "category": "credential_harvesting",
        "name": "QR Code Phishing (Quishing)",
        "mitre_ids": ["T1566.001"],
        "desc": "Using QR codes in phishing attacks to redirect victims to credential harvesting pages.",
        "steps": "1. Create credential harvesting page: clone target login portal\n2. Generate QR code pointing to phishing URL: `qrterminal '{phishing_url}'`\n3. Embed QR code in physical mail, posters, or email\n4. Create pretext: 'Scan to verify your account' or 'Scan for exclusive offer'\n5. Victim scans QR code with mobile device\n6. Mobile browser goes to phishing page — often harder to verify URL on mobile\n7. Capture credentials and session tokens",
        "email_templates": [
            "Subject: Important Security Update\n\nPlease scan the QR code below to update your security settings immediately.\n\n[QR CODE IMAGE]\n\nThis link expires in 24 hours.\n\nIT Security Team",
        ],
        "detection": [
            "QR codes in emails pointing to suspicious URLs",
            "URLs embedded in QR codes that differ from expected domains",
            "Mobile login anomalies",
            "QR code generation services in email attachments",
        ],
        "mitigations": [
            "QR code URL inspection before scanning",
            "MFA enforcement",
            "Mobile phishing detection",
            "Email gateway QR code scanning",
        ],
    },
    {
        "category": "credential_harvesting",
        "name": "AitM Phishing with Reverse Proxy",
        "mitre_ids": ["T1566.001"],
        "desc": "Adversary-in-the-Middle phishing using reverse proxy to intercept authentication flows and bypass MFA.",
        "steps": "1. Deploy reverse proxy infrastructure (Evilginx2, Modlishka, or custom Nginx reverse proxy)\n2. Configure TLS certificates for lookalike domain\n3. Set up proxy rules to forward legitimate auth traffic while capturing credentials\n4. Proxy handles MFA challenge: victim sees real MFA prompt, enters code, proxy forwards it\n5. Capture session cookies and refresh tokens in transit\n6. Session cookies provide persistent access even after MFA\n7. Deploy via phishing email with link to proxy domain",
        "email_templates": [
            "Subject: Your Session Has Expired - Please Re-Authenticate\n\nDear {name},\n\nYour session has expired due to inactivity. Please re-authenticate to continue accessing your account:\n\n{phishing_link}\n\nThis link will expire in 15 minutes.\n\nIT Security Operations",
        ],
        "detection": [
            "Reverse proxy headers in authentication logs",
            "TLS certificate mismatches on login pages",
            "Session tokens with anomalous proxy indicators",
            "Impossible travel for authentication events",
        ],
        "mitigations": [
            "FIDO2/WebAuthn (bound to origin, resists AitM)",
            "Conditional Access with device compliance",
            "Token protection (session binding)",
            "Network-level proxy detection",
        ],
    },
    {
        "category": "credential_harvesting",
        "name": "ADFS Credential Harvesting",
        "mitre_ids": ["T1111"],
        "desc": "Phishing targeting Active Directory Federation Services (ADFS) to steal federated credentials and generate SAML tokens.",
        "steps": "1. Identify ADFS endpoint: discover via Autodiscover or DNS (adfs.company.com)\n2. Clone ADFS login page: replicate the organizational branding and forms-based auth\n3. Deploy phishing site with lookalike domain (adfs.c0mpany.com)\n4. Send phishing email mimicking IT: 'Federation service update required'\n5. Capture credentials including domain username and password\n6. Optional: use captured credentials with ADFS password spray or golden SAML attack\n7. Federated access provides entry to all connected cloud services",
        "email_templates": [
            "Subject: ADFS Service Update - Re-Authentication Required\n\nDear {name},\n\nThe single sign-on federation service has been updated. Please re-authenticate to restore access to all connected services.\n\nAuthenticate now: {phishing_link}\n\nFederation Service Team",
        ],
        "detection": [
            "ADFS authentication failures from phishing URLs",
            "SAML token anomalies (unusual issuer, conditions)",
            "ADFS login page URL mismatches",
            "Federated authentication from unknown IPs",
        ],
        "mitigations": [
            "FIDO2 authentication for federated logins",
            "ADFS monitoring and anomaly detection",
            "SAML token signing key protection",
            "Conditional Access policies for federated users",
        ],
    },
    {
        "category": "credential_harvesting",
        "name": "OAuth App Consent Phishing (Expanded Illicit Consent Grant)",
        "mitre_ids": ["T1566.001"],
        "desc": "Advanced OAuth consent phishing targeting multi-tenant applications with broad permissions, expanding beyond basic Mail.Read.",
        "steps": "1. Register Azure AD multi-tenant app with enticing name: 'Teams Compliance Check', 'Security Dashboard'\n2. Request broad permissions: Mail.ReadWrite, Files.ReadWrite.All, User.Read.All, Notes.Read.All\n3. Create professional app branding and landing page\n4. Generate consent URL with redirect to attacker-controlled endpoint\n5. Send phishing email: 'IT requires you to approve this compliance application'\n6. User grants consent — attacker obtains refresh token with broad access\n7. Use Graph API to read email, files, notes, and user directory data\n8. Persist access: tokens auto-refresh as long as consent not revoked",
        "email_templates": [
            "Subject: Required: Approve Compliance Application\n\nDear Employee,\n\nAs part of our security compliance initiative, all employees must approve the 'Microsoft Compliance Dashboard' application by end of week.\n\nApprove here: {phishing_link}\n\nNon-compliance will result in access restrictions.\n\nIT Compliance Team",
        ],
        "detection": [
            "Consent grants to unrecognized multi-tenant applications",
            "Application with excessive permissions (Files.ReadWrite.All)",
            "Graph API calls from unknown client IDs",
            "Recent app registrations with suspicious permissions",
        ],
        "mitigations": [
            "Require admin consent for all OAuth applications",
            "Application registration restrictions in Azure AD",
            "Regular OAuth consent audit and cleanup",
            "Conditional Access requiring compliant devices for consent",
        ],
    },
    {
        "category": "credential_harvesting",
        "name": "Wi-Fi Captive Portal Credential Harvesting",
        "mitre_ids": ["T1110.001"],
        "desc": "Setting up rogue Wi-Fi access points with captive portals to harvest corporate credentials.",
        "steps": "1. Set up rogue AP: use WiFi-Pumpkin or hostapd-wpe with captive portal\n2. Configure SSID matching corporate or public Wi-Fi: 'Corporate-Guest', 'Free_Airport_WiFi'\n3. Create captive portal: clone corporate login page or common SSO portal\n4. Deploy near target location: office building, conference center, airport\n5. Stronger signal than legitimate AP to attract connections\n6. Capture credentials when users attempt to authenticate\n7. Deauthentication attacks to force reconnection to rogue AP",
        "email_templates": [],
        "detection": [
            "Rogue access points with corporate SSIDs",
            "Captive portal credential submissions to unknown servers",
            "AP location anomalies (unexpected BSSID)",
            "Deauthentication frames in wireless logs",
        ],
        "mitigations": [
            "WPA3 Enterprise authentication",
            "802.1X certificate-based Wi-Fi",
            "Wireless intrusion detection (WIDS)",
            "VPN for all remote connections",
            "Corporate SSID monitoring",
        ],
    },
    {
        "category": "credential_harvesting",
        "name": "SIM Swap Assisted Phishing",
        "mitre_ids": ["T1566.001"],
        "desc": "Combining SIM swap attacks with phishing to intercept MFA codes and bypass authentication.",
        "steps": "1. Gather target phone number and carrier from OSINT, data brokers\n2. Social engineer carrier: pose as target, provide minimal info to port number\n3. Alternative: bribe insider at carrier to perform SIM swap\n4. Once SIM swapped: receive all SMS-based MFA codes\n5. Send phishing email requiring MFA verification\n6. Target enters password on phishing site, attacker enters on real site\n7. MFA code sent to swapped SIM, attacker receives it\n8. Full account takeover: credentials + MFA bypassed",
        "email_templates": [
            "Subject: Security Alert: Unrecognized Sign-In Attempt\n\nDear {name},\n\nWe detected an unrecognized sign-in attempt on your account. If this wasn't you, please verify your identity immediately:\n\n{phishing_link}\n\nVerification requires your phone number. Please ensure your number on file is correct.\n\nSecurity Team",
        ],
        "detection": [
            "SIM swap notifications from carrier",
            "MFA codes used from different device/location",
            "Account recovery requests after SIM swap",
            "Multiple authentication failures followed by success",
        ],
        "mitigations": [
            "FIDO2/WebAuthn (not SMS-based MFA)",
            "Carrier SIM lock / port freeze",
            "Authenticator app MFA instead of SMS",
            "Account recovery with backup codes only",
        ],
    },
    # === VISHING ===
    {
        "category": "vishing",
        "name": "Voice Phishing (Vishing) - IT Helpdesk",
        "mitre_ids": ["T1598.001"],
        "desc": "Calling targets posing as IT helpdesk to extract credentials or MFA codes.",
        "steps": "1. Recon: Gather employee names and phone numbers from LinkedIn, company directory\n2. Prepare pretext: 'IT helpdesk calling about password reset' or 'security verification'\n3. Call during business hours for credibility\n4. Build rapport: reference real IT systems, use company terminology\n5. Request: ask for username, password, and MFA code 'for verification'\n6. Use credentials immediately while MFA code is still valid\n7. If successful, call more targets using same pretext",
        "email_templates": [],
        "detection": [
            "Multiple helpdesk-related calls in short period",
            "Credential reset requests outside normal patterns",
            "Helpdesk call recording anomalies",
            "User reports of suspicious calls",
        ],
        "mitigations": [
            "IT will NEVER ask for passwords or MFA codes by phone",
            "Verify caller ID through callback procedure",
            "Log all helpdesk interactions",
            "Security awareness training on vishing",
        ],
    },
    {
        "category": "vishing",
        "name": "Vishing - Bank Fraud Pretext",
        "mitre_ids": ["T1598.001"],
        "desc": "Calling targets posing as bank fraud department to extract financial information.",
        "steps": "1. Target selection: Use leaked data or OSINT to identify bank customers\n2. Prepare caller ID spoofing: use service to show bank's phone number\n3. Create urgency: 'We've detected fraudulent charges on your account'\n4. Verification pretext: 'I need to verify your identity'\n5. Extract: full card number, CVV, PIN, online banking credentials\n6. Optional: send SMS with confirmation code, ask target to read it back (MFA bypass)",
        "email_templates": [],
        "detection": [
            "Unusual account access patterns after phone calls",
            "Bank fraud reports from multiple customers",
            "Caller ID spoofing detection",
            "Rapid credential use after phone interaction",
        ],
        "mitigations": [
            "Banks never ask for full card numbers by phone",
            "Callback verification procedure",
            "Fraud detection systems",
            "Customer education",
        ],
    },
    {
        "category": "vishing",
        "name": "Vishing with Caller ID Spoofing for Tech Support",
        "mitre_ids": ["T1598.001"],
        "desc": "Using caller ID spoofing to impersonate tech support and gain remote access to systems.",
        "steps": "1. Acquire caller ID spoofing service or VoIP with number masking\n2. Spoof caller ID to show legitimate tech support number (Microsoft, Dell, internal IT)\n3. Call target: 'We've detected malware on your computer' or 'Your license needs renewal'\n4. Build trust: reference real software, OS version (gathered from OSINT)\n5. Guide target to install remote access tool: 'Download TeamViewer for diagnosis'\n6. Once connected: install persistent backdoor, exfiltrate data, or encrypt files\n7. Request payment for 'repair services' (adds financial fraud layer)",
        "email_templates": [],
        "detection": [
            "Unsolicited tech support calls reported by users",
            "Remote access tool installations outside policy",
            "Caller ID spoofing patterns (same spoofed number, multiple targets)",
            "Rapid software installation after phone calls",
        ],
        "mitigations": [
            "Never grant remote access to unsolicited callers",
            "Verify tech support calls via official callback number",
            "Remote access tool restrictions (whitelisting only approved tools)",
            "User training on tech support scam patterns",
        ],
    },
    {
        "category": "vishing",
        "name": "Vishing Targeting Finance Department (Wire Fraud)",
        "mitre_ids": ["T1598.001"],
        "desc": "Voice phishing specifically targeting finance staff to authorize fraudulent wire transfers.",
        "steps": "1. Identify finance team members: AP clerk, controller, CFO from LinkedIn, website\n2. Research current wire transfer processes, approval workflows, and vendor relationships\n3. Call finance staff posing as executive or vendor: 'I need an urgent wire processed'\n4. Create urgency: 'Acquisition closing tomorrow', 'Vendor will halt shipments'\n5. Provide fraudulent bank account details over phone\n6. Follow up with spoofed email confirmation for 'paper trail'\n7. Target processes wire based on call + email 'verification'",
        "email_templates": [],
        "detection": [
            "Phone-initiated wire transfers without proper documentation",
            "Urgent wire requests with new account details",
            "Finance calls from unknown numbers followed by wire transfers",
            "Wire transfers outside normal business processes",
        ],
        "mitigations": [
            "No wire transfers based on phone calls alone",
            "Dual authorization requiring in-person or verified callback",
            "Bank detail changes require separate verification channel",
            "Finance staff vishing awareness training",
        ],
    },
    {
        "category": "vishing",
        "name": "Vishing with AI-Generated Voice Deepfake",
        "mitre_ids": ["T1598.001"],
        "desc": "Using AI voice cloning technology to impersonate known individuals in vishing attacks.",
        "steps": "1. Obtain target voice samples: extract from YouTube, podcasts, voicemail greetings (5-10 min minimum)\n2. Train voice clone: use ElevenLabs, Resemble AI, or open-source TTS (Coqui, Bark)\n3. Prepare call script with executive's known phrases and speaking patterns\n4. Spoof caller ID to show executive's number or internal extension\n5. Execute call: deepfake voice makes urgent request with executive's mannerisms\n6. Real-time voice synthesis allows natural conversation with pauses and responses\n7. Record call for evidence and quality analysis",
        "email_templates": [],
        "detection": [
            "Voice artifacts: slight robotic quality, unnatural pauses",
            "Impossible communication patterns (executive known to be elsewhere)",
            "Caller ID inconsistencies",
            "Unusual requests from executives via phone only",
        ],
        "mitigations": [
            "Codeword/phrase system for financial and sensitive requests",
            "Out-of-band verification for all phone-initiated transactions",
            "Voice biometric authentication as secondary factor",
            "Training on deepfake voice detection indicators",
        ],
    },
    {
        "category": "vishing",
        "name": "Vishing for MFA Reset Codes",
        "mitre_ids": ["T1598.001"],
        "desc": "Calling targets to trick them into providing MFA reset codes or approving MFA push notifications.",
        "steps": "1. Identify target with known username/password (from breach data)\n2. Attempt login to trigger MFA push notification to target's device\n3. Simultaneously call target: 'IT Security — we need you to approve the prompt on your phone'\n4. Alternative: call claiming 'security team needs your MFA code for account verification'\n5. Pretext: 'There's been a breach, we need to verify your MFA is working'\n6. Target approves MFA push or reads code aloud\n7. Attacker gains authenticated session",
        "email_templates": [],
        "detection": [
            "MFA approval immediately following phone call",
            "Multiple MFA push notifications in rapid succession (MFA fatigue)",
            "Helpdesk calls followed by authentication events",
            "MFA approvals from anomalous locations",
        ],
        "mitigations": [
            "Never approve unexpected MFA prompts",
            "MFA number matching (requires matching number on both ends)",
            "IT will NEVER call asking for MFA codes",
            "MFA fatigue protection (limit daily pushes)",
        ],
    },
    {
        "category": "vishing",
        "name": "Vishing Targeting Helpdesk",
        "mitre_ids": ["T1598.001"],
        "desc": "Calling IT helpdesk posing as employee to reset credentials or gain account access.",
        "steps": "1. Recon: gather employee names, titles, department, manager info from OSINT\n2. Call helpdesk posing as employee: 'I'm locked out, traveling, need password reset ASAP'\n3. Provide gathered info to answer security questions: manager name, office, employee ID\n4. Create urgency: 'I'm about to present to the board, I need access NOW'\n5. Helpdesk resets password or provides temporary credentials\n6. Attacker logs in with new credentials\n7. Optional: set up email forwarding rule for persistent access",
        "email_templates": [],
        "detection": [
            "Password resets for employees who don't match caller ID",
            "Multiple resets from same phone number for different accounts",
            "Helpdesk ticket anomalies (urgent resets, travel excuses)",
            "Login from new device immediately after password reset",
        ],
        "mitigations": [
            "Callback verification before password resets",
            "Multi-factor identity verification for helpdesk",
            "Helpdesk ticket documentation requirements",
            "Rate limiting on password resets per phone number",
        ],
    },
    # === PRETEXTING ===
    {
        "category": "pretexting",
        "name": "IT Support Pretexting",
        "mitre_ids": ["T1598"],
        "desc": "Creating a false scenario (pretext) to trick targets into revealing information.",
        "steps": "1. Research target: identify IT support vendor, helpdesk numbers, recent IT changes\n2. Create pretext: 'I'm {name} from {IT_vendor}, we're doing emergency maintenance'\n3. Build credibility: reference real systems, use correct terminology\n4. Make contact: phone call, email, or in-person visit\n5. Execute: request credentials, remote access, or physical access\n6. Document and maintain cover story throughout interaction",
        "email_templates": [
            "Subject: Scheduled Maintenance - VPN Access Required\n\nDear {name},\n\nWe are performing emergency maintenance on the corporate VPN tonight from 11 PM to 2 AM. To ensure uninterrupted access, please verify your VPN credentials by replying with:\n\n1. Your username\n2. Your current password\n3. Your MFA backup codes\n\nThis is required to migrate your account to the new VPN system.\n\nIT Infrastructure Team\n{spoofed_vendor_name}",
        ],
        "detection": [
            "Unusual IT support requests via email",
            "Requests for credentials via non-standard channels",
            "Helpdesk impersonation patterns",
            "Multiple similar pretexting attempts",
        ],
        "mitigations": [
            "Verify IT support requests through official channels",
            "Never share credentials via email",
            "Establish identity verification procedures",
            "Security awareness training",
        ],
    },
    {
        "category": "pretexting",
        "name": "Authority Figure Impersonation",
        "mitre_ids": ["T1598"],
        "desc": "Impersonating a person of authority to manipulate targets into compliance.",
        "steps": "1. Identify authority figure: CEO, CFO, law enforcement, auditor\n2. Research target: find subordinates who would comply with authority figure\n3. Create urgency: 'This is time-sensitive, don't tell anyone'\n4. Leverage authority: 'I'm the CFO, I need this wire transfer processed immediately'\n5. Create isolation: 'Don't discuss this with your manager, it's confidential'\n6. Escalate pressure if target hesitates",
        "email_templates": [
            "Subject: CONFIDENTIAL - Immediate Action Required\n\n{name},\n\nThis is {authority_name}, {title}. I need you to process an urgent payment of $50,000 to the account below. This is related to a confidential acquisition and must not be discussed with anyone else.\n\nAccount: {fraud_account}\nRouting: {fraud_routing}\n\nI'm in meetings all day, so please handle this via email only.\n\n{authority_name}",
        ],
        "detection": [
            "Urgent requests bypassing normal procedures",
            "Requests for secrecy",
            "Unusual payment destinations",
            "Authority figures using non-standard communication",
        ],
        "mitigations": [
            "Dual authorization for financial transactions",
            "Out-of-band verification",
            "No-secrecy policy for financial requests",
            "Training on authority-based social engineering",
        ],
    },
    {
        "category": "pretexting",
        "name": "Tailgating / Physical Social Engineering",
        "mitre_ids": ["T1200"],
        "desc": "Gaining physical access to restricted areas by following authorized personnel.",
        "steps": "1. Recon: observe building entry patterns, badge-in locations, smoking areas\n2. Approach: wait near entrance with hands full (carrying boxes, coffee)\n3. Pretext: 'Can you hold the door? I forgot my badge' or 'New employee, first day'\n4. Alternative: dress as delivery person, IT technician, or maintenance worker\n5. Once inside: locate server rooms, unattended workstations, or secure areas\n6. Plant devices: USB drop, keylogger, or wireless access point",
        "email_templates": [],
        "detection": [
            "Unauthorized individuals in restricted areas",
            "Badge-in without matching badge-out",
            "Unrecognized individuals in secure zones",
            "Video surveillance anomalies",
        ],
        "mitigations": [
            "Mantrap entry systems",
            "Badge-in/badge-out requirements",
            "Visitor escort policies",
            "Security awareness training",
            "Physical access audits",
        ],
    },
    {
        "category": "pretexting",
        "name": "Pretexting as Job Recruiter (Resume Phishing)",
        "mitre_ids": ["T1598"],
        "desc": "Posing as a recruiter to collect personal information and deliver malware through fake job postings.",
        "steps": "1. Create fake recruiter profile on LinkedIn with legitimate-looking company\n2. Research target's career aspirations from their LinkedIn activity\n3. Craft personalized job offer email referencing target's skills and experience\n4. Attach malicious document: 'Job description.docm' or 'Application form.xlsm'\n5. Alternative: link to fake job portal for credential harvesting\n6. Request PII: SSN for 'background check', bank info for 'direct deposit setup'\n7. Maintain recruiter persona through follow-up communications",
        "email_templates": [
            "Subject: Exciting Opportunity - {position} at {company}\n\nHi {name},\n\nI came across your profile and was impressed by your experience in {skill}. We have an opening for a {position} role that I think would be a great fit.\n\nPlease review the attached job description and let me know if you're interested. If so, we can schedule a call this week.\n\nAttachment: Job_Description_{position}.docm\n\nBest regards,\n{spoofed_sender}\nSenior Recruiter, {company}",
        ],
        "detection": [
            "Job offers from unknown recruiters with attachments",
            "Emails requesting SSN or financial info for 'background checks'",
            "Macro-enabled documents in recruiter emails",
            "Mismatched recruiter email domain vs company",
        ],
        "mitigations": [
            "Verify recruiter identity through official company channels",
            "Never provide SSN in initial job applications",
            "Resume phishing awareness training",
            "Email gateway attachment scanning for macros",
        ],
    },
    {
        "category": "pretexting",
        "name": "Pretexting as Auditor/Compliance Officer",
        "mitre_ids": ["T1598"],
        "desc": "Impersonating an auditor or compliance officer to gain access to sensitive systems and data.",
        "steps": "1. Research target company's audit schedule, compliance requirements, and audit firm\n2. Create pretext: 'I'm {name} from {audit_firm}, conducting annual compliance review'\n3. Forge audit credentials, business cards, or documentation\n4. Request access: system credentials, database access, physical server room access\n5. Cite regulatory pressure: 'SOX compliance deadline', 'HIPAA audit requirements'\n6. Document findings: actually exfiltrate sensitive data under guise of audit\n7. Maintain cover story throughout multi-day engagement",
        "email_templates": [
            "Subject: Compliance Audit - Access Request\n\nDear {name},\n\nI am conducting the annual SOX compliance audit for your organization. Per regulatory requirements, I need access to the following systems:\n\n1. Financial reporting database (read access)\n2. User access management logs\n3. Network security audit logs\n\nPlease provide credentials and schedule access by end of business today.\n\n{authority_name}\n{audit_firm}\nCompliance Division",
        ],
        "detection": [
            "Unscheduled audit requests",
            "Auditor credentials that cannot be verified with firm",
            "Direct credential requests instead of escorted access",
            "Urgency citing regulatory deadlines",
        ],
        "mitigations": [
            "Verify all auditors through audit engagement letter",
            "Auditors never receive direct credentials — escorted access only",
            "Cross-reference audit schedule with management",
            "Compliance officer verification protocol",
        ],
    },
    {
        "category": "pretexting",
        "name": "Pretexting as IT Vendor Performing Security Audit",
        "mitre_ids": ["T1598"],
        "desc": "Impersonating an IT security vendor conducting a penetration test to gain system access.",
        "steps": "1. Identify target's current or past security vendors from job postings, case studies\n2. Create pretext: 'I'm from {known_vendor}, performing scheduled penetration test'\n3. Forge engagement letter or reference real pentest scope document\n4. Request network access, VPN credentials, or admin accounts 'for testing'\n5. Cite NDA and confidentiality to prevent target from verifying with management\n6. Once inside: enumerate network, exfiltrate data, establish persistence\n7. Cover: deliver fake findings report to maintain legitimacy",
        "email_templates": [
            "Subject: Scheduled Penetration Test - Access Credentials Required\n\nDear {name},\n\nPer the engagement letter signed on {date}, our team will begin the scheduled penetration test on {start_date}. To conduct thorough testing, we require:\n\n1. VPN access credentials\n2. Standard user AD account\n3. Network diagram access\n\nPlease provide these by {deadline}. All findings will be reported per our NDA.\n\n{authority_name}\nSecurity Assessment Team\n{spoofed_vendor_name}",
        ],
        "detection": [
            "Unscheduled penetration test notifications",
            "Vendor requests for direct credentials (should use provided accounts)",
            "Pentest engagement that cannot be verified with CISO",
            "Vendor emails from non-standard domains",
        ],
        "mitigations": [
            "All pentests require written scope and CISO approval",
            "Never provide production credentials to pentesters",
            "Verify pentest schedule with security team and CISO",
            "Pentest engagement letter verification protocol",
        ],
    },
    {
        "category": "pretexting",
        "name": "Pretexting as Law Enforcement",
        "mitre_ids": ["T1598"],
        "desc": "Impersonating law enforcement to intimidate targets into providing information or access.",
        "steps": "1. Research target: identify employees who may be susceptible to authority pressure\n2. Create pretext: detective, federal agent, or compliance investigator\n3. Reference real or fabricated legal authority: 'ongoing investigation', 'subpoena', 'court order'\n4. Create urgency: 'You could face charges if you don't cooperate'\n5. Request: employee records, financial data, system access, or personal information\n6. Use intimidation: imply legal consequences for non-compliance\n7. Maintain cover story with legal terminology",
        "email_templates": [
            "Subject: URGENT - Legal Compliance Required\n\nDear {name},\n\nThis is Detective {authority_name} with the Cyber Crimes Division. We are conducting an investigation involving your organization. Per our ongoing case, we require the following information to be provided by end of business today:\n\n1. Employee directory with contact details\n2. Financial transaction records for Q3\n3. System access logs for user accounts\n\nFailure to comply may result in a subpoena. Please treat this matter with the utmost urgency.\n\nDetective {authority_name}\nBadge #{badge_number}\nCyber Crimes Division",
        ],
        "detection": [
            "Law enforcement requests via email (law enforcement uses official channels)",
            "Requests citing investigations without verifiable case numbers",
            "Threats of legal action for non-compliance",
            "Requests for bulk data without specific warrants",
        ],
        "mitigations": [
            "Law enforcement requests must go through legal counsel",
            "Never provide data without verifying through official channels",
            "Contact legal department immediately for any LE requests",
            "Training on law enforcement impersonation scams",
        ],
    },
    {
        "category": "pretexting",
        "name": "Pretexting as Delivery Company",
        "mitre_ids": ["T1598"],
        "desc": "Impersonating delivery services (FedEx, UPS, DHL) to deliver malicious payloads or collect information.",
        "steps": "1. Send email mimicking package delivery notification from major carrier\n2. Create tracking link pointing to credential harvesting page or malware download\n3. Pretext: 'Package delivery requires signature', 'Customs fee due', 'Address correction needed'\n4. Alternative: call posing as delivery driver to gain physical access to building\n5. Physical: 'I have a delivery for {employee}, can you let me in?'\n6. Digital: link leads to fake tracking page that harvests credentials\n7. Time delivery emails around holidays or peak shipping seasons",
        "email_templates": [
            "Subject: Delivery Attempt - Action Required\n\nDear {name},\n\nWe attempted to deliver your package but no one was available to sign. Please click below to reschedule delivery or update your address:\n\n{phishing_link}\n\nPackage ID: {tracking_number}\n\nFedEx Delivery Services",
        ],
        "detection": [
            "Delivery notifications from unknown senders",
            "Tracking links pointing to non-carrier domains",
            "Unexpected package notifications",
            "Delivery emails with attachments or zip files",
        ],
        "mitigations": [
            "Verify delivery notifications through official carrier websites",
            "Never click tracking links in unsolicited emails",
            "Track packages using order confirmations, not email links",
            "Delivery phishing awareness training",
        ],
    },
    # === DEEPFAKE SE ===
    {
        "category": "deepfake_se",
        "name": "Deepfake Voice (Audio) Social Engineering",
        "mitre_ids": ["T1598.001"],
        "desc": "Using AI-generated voice cloning to impersonate executives in phone calls.",
        "steps": "1. Obtain audio samples: extract from earnings calls, interviews, podcasts (10-30 min needed)\n2. Train voice clone: use ElevenLabs, Resemble AI, or custom TTS model\n3. Test voice quality: verify with sample phrases\n4. Craft call script: urgent request with executive's speaking patterns\n5. Execute call: use VoIP with spoofed caller ID showing executive's number\n6. Record for evidence and analysis",
        "email_templates": [],
        "detection": [
            "Voice quality artifacts (slight robotic quality)",
            "Unusual call patterns from executives",
            "Caller ID spoofing detection",
            "Impossible communication patterns (executive in known meeting)",
        ],
        "mitigations": [
            "Codeword/phrase system for financial requests",
            "Out-of-band verification for sensitive requests",
            "Voice authentication as secondary factor",
            "Employee training on deepfake threats",
        ],
    },
    {
        "category": "deepfake_se",
        "name": "Deepfake Video Social Engineering",
        "mitre_ids": ["T1598.001"],
        "desc": "Using AI-generated video to impersonate executives in video calls.",
        "steps": "1. Obtain video samples: YouTube, company videos, webinars\n2. Train deepfake model: use DeepFaceLab or commercial tool\n3. Set up real-time deepfake: use DeepFaceLive for video calls\n4. Create meeting scenario: 'Emergency board meeting via Zoom'\n5. Execute: join meeting with deepfake video, make request\n6. Maintain cover: manage lip sync, expressions, and audio quality",
        "email_templates": [],
        "detection": [
            "Video quality artifacts around face edges",
            "Lip sync desynchronization",
            "Unusual meeting requests from executives",
            "Request for camera-off from 'executive'",
        ],
        "mitigations": [
            "Verification codewords for video calls",
            "Multi-factor identity verification",
            "Training on deepfake video detection",
            "Video authentication protocols for sensitive requests",
        ],
    },
    {
        "category": "deepfake_se",
        "name": "Deepfake Video for Remote Interview Fraud",
        "mitre_ids": ["T1598.001"],
        "desc": "Using deepfake video technology to impersonate job candidates in remote interviews for employment fraud.",
        "steps": "1. Obtain target's video from social media or create synthetic identity\n2. Train deepfake model on target's face or generate realistic synthetic face\n3. Set up real-time deepfake pipeline for video call: DeepFaceLive + virtual camera\n4. Apply for remote positions requiring video interviews\n5. Use deepfake to pass video interview while another person provides answers\n6. Complete hiring process, gain remote access to company systems\n7. Exfiltrate data or establish persistent access through employment",
        "email_templates": [],
        "detection": [
            "Video artifacts: face edge blending, unnatural blinking patterns",
            "Candidate voice not matching profile",
            "Refusal to turn on camera for second interview",
            "Technical difficulties when asked to show hands or move head",
        ],
        "mitigations": [
            "Multi-stage interview with different interviewers",
            "Identity verification before hiring (ID + live video)",
            "Background check with video call verification",
            "Training on deepfake detection for HR and hiring managers",
        ],
    },
    {
        "category": "deepfake_se",
        "name": "Deepfake Voice for Authorization Calls",
        "mitre_ids": ["T1598.001"],
        "desc": "Using deepfake voice to make authorization calls for financial transactions or access changes.",
        "steps": "1. Obtain executive voice samples from public appearances (earnings calls, keynotes)\n2. Train high-quality voice clone with emotional range\n3. Prepare authorization script matching the executive's communication style\n4. Call finance team: 'This is {CFO_name}, I'm authorizing a wire transfer'\n5. Use voice clone to match executive's tone, pace, and vocabulary\n6. Provide authorization codes or 'callback numbers' that route to attacker\n7. Finance processes transfer based on voice authorization",
        "email_templates": [],
        "detection": [
            "Unusual authorization calls from executives",
            "Voice quality anomalies in call recordings",
            "Authorization calls outside normal business patterns",
            "Callback numbers that don't match executive's known numbers",
        ],
        "mitigations": [
            "Dual authorization for all large transactions (voice + email)",
            "Callback to verified executive phone number",
            "Voice biometric verification for phone authorizations",
            "Transaction limits on phone-authorized transfers",
        ],
    },
    {
        "category": "deepfake_se",
        "name": "AI-Generated Phishing Email Content",
        "mitre_ids": ["T1566.001"],
        "desc": "Using LLMs to generate highly personalized, grammatically perfect phishing emails that bypass traditional detection.",
        "steps": "1. Gather target information from OSINT: LinkedIn, social media, company website\n2. Use LLM (GPT-4, Claude, etc.) to generate personalized email content\n3. Craft prompts: 'Write a professional email from {role} about {topic} to {target}'\n4. Generate multiple variants to A/B test effectiveness\n5. AI produces native-quality text: no grammar errors, proper tone, industry jargon\n6. Bypass traditional phishing detection that relies on grammar/spelling errors\n7. Combine with lookalike domain and email authentication setup",
        "email_templates": [
            "Subject: Project Update - {project_name}\n\nHi {name},\n\nFollowing up on our discussion from {recent_event}. I've put together the revised project timeline and need your input before we present to the board.\n\nCan you review and share your thoughts? I've uploaded the document here:\n\n{phishing_link}\n\nLet me know if you have any questions.\n\nBest regards,\n{spoofed_sender}",
        ],
        "detection": [
            "AI-generated text patterns (lack of personal quirks)",
            "Perfect grammar in phishing emails (traditionally a red flag in reverse)",
            "Highly personalized content without prior relationship",
            "Email authentication failures despite perfect content",
        ],
        "mitigations": [
            "DMARC enforcement (AI content bypasses text-based filters)",
            "Email authentication regardless of content quality",
            "Behavioral analysis of email patterns",
            "AI-powered phishing detection (detect LLM writing patterns)",
        ],
    },
    {
        "category": "deepfake_se",
        "name": "Deepfake for Social Media Impersonation",
        "mitre_ids": ["T1598.001"],
        "desc": "Using deepfake technology to create convincing social media profiles for long-term social engineering campaigns.",
        "steps": "1. Generate synthetic identity: AI face generation (ThisPersonDoesNotExist), synthetic background\n2. Create deepfake video content: short clips showing 'person' in professional settings\n3. Set up social media profiles: LinkedIn, Twitter, Facebook with synthetic identity\n4. Build network: connect with target's colleagues, engage with posts\n5. Maintain persona for weeks/months: post regularly, comment, share industry content\n6. Develop relationship with target through sustained professional engagement\n7. Execute: request information, share malicious links, or set up in-person meeting",
        "email_templates": [],
        "detection": [
            "Social media profiles with AI-generated faces (reverse image search reveals no history)",
            "Profiles with limited photo variety or perfect-looking images",
            "Rapid network growth with industry-specific connections",
            "Profile created recently with extensive professional history",
        ],
        "mitigations": [
            "Reverse image search on new professional connections",
            "Social media verification procedures for professional contacts",
            "In-person or video verification for online-only contacts",
            "Training on synthetic identity awareness",
        ],
    },
]

QUESTION_TEMPLATES = {
    "spear_phishing": [
        "Explain {name} with a detailed step-by-step attack walkthrough.",
        "How does {name} work? Include email templates and detection methods.",
        "For a red team engagement, detail the {name} technique with specific tactics.",
        "Walk me through a {name} attack scenario. What are the key indicators?",
        "Describe {name} from initial reconnaissance to payload delivery. Include mitigations.",
        "What are the most effective {name} techniques? Show email templates and evasion methods.",
        "In a {environment_type}, how would you test for {name}? Reference {framework}.",
        "What {severity_level}-severity indicators should detection teams monitor for {name}?",
        "Design a {framework} aligned test plan for {name} targeting a {environment_type}.",
        "How does {name} differ in a {environment_type} versus a corporate environment? Include detection strategies.",
        "What are the MITRE ATT&CK techniques associated with {name}? Map to {framework} testing methodology.",
        "Create a detection engineering rule set for {name} in a {environment_type}. Prioritize {severity_level} indicators.",
        "How would you validate {name} mitigations in a {environment_type} using {framework}?",
        "What email gateway rules would catch {name} in a {environment_type}? Reference {framework} controls.",
        "Describe the attack chain for {name} with {severity_level} severity classification per {framework}.",
        "What user behavior analytics would detect {name} in a {environment_type}? Map to {framework}.",
        "How would {name} technique be adapted for a {environment_type}? Include countermeasures from {framework}.",
        "What threat hunting hypotheses apply to {name} in a {environment_type}? Reference {framework}.",
        "Design a purple team exercise for {name} validation in a {environment_type} using {framework} methodology.",
        "What are the most common {name} pre-attack indicators in a {environment_type}? Reference {framework} threat model.",
        "How would an adversary chain {name} with other techniques in a {environment_type}? Map to {framework}.",
    ],
    "bec": [
        "Explain {name} with a realistic attack scenario and prevention strategies.",
        "How does {name} work? Detail the attack flow from compromise to fund diversion.",
        "For a financial fraud assessment, demonstrate {name} techniques.",
        "Walk me through {name} attack detection and response procedures.",
        "What makes {name} effective? Include email templates and mitigation strategies.",
        "In a {environment_type}, what {severity_level}-severity BEC risks does {name} pose?",
        "Design a {framework} assessment for {name} targeting a {environment_type}.",
        "What financial controls would prevent {name} in a {environment_type}? Reference {framework}.",
        "How would you detect {name} in a {environment_type}? Map detection to {framework}.",
        "What email authentication policies mitigate {name} in a {environment_type}? Cite {framework}.",
        "Create a response playbook for {name} in a {environment_type} with {severity_level} severity classification.",
        "What are the {framework} controls that address {name}?",
        "How does {name} exploit trust relationships in a {environment_type}? Include {framework} countermeasures.",
        "Design a tabletop exercise for {name} response in a {environment_type} per {framework} methodology.",
        "What vendor verification procedures would prevent {name} in a {environment_type}? Reference {framework}.",
        "How would you simulate {name} in a {environment_type} using {framework} methodology?",
        "What financial fraud indicators point to {name} in a {environment_type}? Map to {framework}.",
        "Describe the social engineering psychology behind {name} and {framework} recommended countermeasures.",
        "What wire transfer controls would stop {name} in a {environment_type}? Reference {framework} financial controls.",
        "How would {name} be adapted for a {environment_type}? Include {framework} testing approach.",
        "What are the earliest detection opportunities for {name} in a {environment_type}? Reference {framework}.",
    ],
    "credential_harvesting": [
        "Explain {name} with exploitation steps and defensive measures.",
        "How does {name} bypass MFA? Include technical setup and attack flow.",
        "For a credential security assessment, detail {name} techniques and detection.",
        "Walk me through {name} from attacker setup to credential capture.",
        "What are the most effective {name} tools? Include configuration and deployment steps.",
        "In a {environment_type}, what {severity_level}-severity credential theft risks does {name} pose?",
        "Design a {framework} aligned assessment for {name} in a {environment_type}.",
        "What authentication controls prevent {name} in a {environment_type}? Reference {framework}.",
        "How would you detect {name} credential capture in a {environment_type}? Map to {framework} indicators.",
        "What conditional access policies mitigate {name} in a {environment_type}? Cite {framework}.",
        "Create a detection rule set for {name} in a {environment_type} with {severity_level} priority indicators.",
        "What phishing-resistant authentication methods address {name} per {framework}?",
        "How does {name} exploit authentication workflows in a {environment_type}? Include {framework} mitigations.",
        "Design a credential harvesting simulation for {name} in a {environment_type} per {framework}.",
        "What session monitoring would detect {name} outcomes in a {environment_type}? Reference {framework}.",
        "How would you validate {name} mitigations using {framework} in a {environment_type}?",
        "What identity protection solutions address {name} in a {environment_type}? Map to {framework}.",
        "Describe the technical infrastructure behind {name} and {framework} recommended countermeasures.",
        "What token protection policies prevent {name} persistence in a {environment_type}? Reference {framework}.",
        "How does {name} adapt to different {environment_type} configurations? Include {framework} testing methodology.",
        "What are the authentication anomaly indicators for {name} in a {environment_type}? Reference {framework}.",
    ],
    "vishing": [
        "Explain {name} attack methodology and how to defend against it.",
        "How does {name} work? Include pretext scripts and social engineering tactics.",
        "For a social engineering assessment, detail {name} attack techniques.",
        "Walk me through {name} attack planning, execution, and detection.",
        "What makes {name} effective? Include detection indicators and prevention.",
        "In a {environment_type}, what {severity_level}-severity vishing risks does {name} present?",
        "Design a {framework} social engineering test for {name} in a {environment_type}.",
        "What voice authentication controls prevent {name} in a {environment_type}? Reference {framework}.",
        "How would you detect {name} call patterns in a {environment_type}? Map to {framework} indicators.",
        "What call verification procedures mitigate {name} in a {environment_type}? Cite {framework}.",
        "Create a vishing response playbook for {name} in a {environment_type} with {severity_level} classification.",
        "What are the {framework} controls addressing {name}?",
        "How does {name} exploit phone-based trust in a {environment_type}? Include {framework} countermeasures.",
        "Design a vishing awareness exercise for {name} in a {environment_type} per {framework}.",
        "What caller ID verification would prevent {name} in a {environment_type}? Reference {framework}.",
        "How would you simulate {name} in a {environment_type} using {framework} methodology?",
        "What telephony security indicators point to {name} in a {environment_type}? Map to {framework}.",
        "Describe the pretexting psychology behind {name} and {framework} recommended defenses.",
        "What MFA bypass techniques does {name} use in a {environment_type}? Reference {framework} mitigations.",
        "How would {name} be adapted for targeting a {environment_type}? Include {framework} testing approach.",
        "What are the earliest detection opportunities for {name} in a {environment_type}? Reference {framework}.",
    ],
    "pretexting": [
        "Explain {name} with detailed attack scenario and defense strategies.",
        "How does {name} work? Include pretext development and execution tactics.",
        "For a social engineering assessment, demonstrate {name} techniques.",
        "Walk me through {name} attack planning and execution. What are the key indicators?",
        "Describe {name} pretexts and how to identify them.",
        "In a {environment_type}, what {severity_level}-severity pretexting risks does {name} pose?",
        "Design a {framework} social engineering assessment for {name} in a {environment_type}.",
        "What identity verification controls prevent {name} in a {environment_type}? Reference {framework}.",
        "How would you detect {name} pretext scenarios in a {environment_type}? Map to {framework}.",
        "What verification procedures mitigate {name} in a {environment_type}? Cite {framework}.",
        "Create a pretexting incident response playbook for {name} in a {environment_type} with {severity_level} priority.",
        "What are the {framework} controls that address {name}?",
        "How does {name} exploit authority and trust in a {environment_type}? Include {framework} countermeasures.",
        "Design a pretexting awareness exercise for {name} in a {environment_type} per {framework}.",
        "What background verification would prevent {name} in a {environment_type}? Reference {framework}.",
        "How would you simulate {name} in a {environment_type} using {framework} methodology?",
        "What social engineering indicators point to {name} in a {environment_type}? Map to {framework}.",
        "Describe the psychological manipulation in {name} and {framework} recommended awareness training.",
        "What access control policies would stop {name} in a {environment_type}? Reference {framework} access controls.",
        "How would {name} be adapted for a {environment_type}? Include {framework} testing methodology.",
        "What are the trust exploitation patterns in {name} for a {environment_type}? Reference {framework}.",
    ],
    "deepfake_se": [
        "Explain {name} attack techniques and detection methods.",
        "How does {name} work? Include technical setup and social engineering tactics.",
        "For a modern threat assessment, detail {name} capabilities and mitigations.",
        "Walk me through {name} attack scenario. How can organizations defend?",
        "What are the latest {name} techniques? Include AI model training considerations.",
        "In a {environment_type}, what {severity_level}-severity deepfake risks does {name} present?",
        "Design a {framework} assessment for {name} in a {environment_type}.",
        "What deepfake detection controls prevent {name} in a {environment_type}? Reference {framework}.",
        "How would you detect {name} artifacts in a {environment_type}? Map to {framework} indicators.",
        "What identity verification procedures mitigate {name} in a {environment_type}? Cite {framework}.",
        "Create a deepfake incident response playbook for {name} in a {environment_type} with {severity_level} classification.",
        "What are the {framework} controls addressing {name}?",
        "How does {name} exploit AI-generated media trust in a {environment_type}? Include {framework} countermeasures.",
        "Design a deepfake awareness exercise for {name} in a {environment_type} per {framework}.",
        "What media verification would prevent {name} in a {environment_type}? Reference {framework}.",
        "How would you simulate {name} in a {environment_type} using {framework} methodology?",
        "What media forensics indicators point to {name} in a {environment_type}? Map to {framework}.",
        "Describe the technical capabilities behind {name} and {framework} recommended countermeasures.",
        "What verification codeword systems address {name} in a {environment_type}? Reference {framework}.",
        "How would {name} evolve in a {environment_type}? Include {framework} forward-looking defenses.",
        "What are the detection engineering opportunities for {name} in a {environment_type}? Reference {framework}.",
    ],
}


def generate_pairs(count: int = 5) -> list[dict]:
    """Generate synthetic phishing/SE training pairs with combinatorial expansion."""
    random.seed(SEED)
    pairs: list[dict] = []

    for scenario in SCENARIOS:
        category = scenario["category"]
        name = scenario["name"]
        mitre_ids = scenario["mitre_ids"]

        templates = QUESTION_TEMPLATES.get(
            category, QUESTION_TEMPLATES["spear_phishing"]
        )

        environment_types = random.sample(
            ENVIRONMENT_TYPES, min(count, len(ENVIRONMENT_TYPES))
        )
        severity_levels = random.sample(
            SEVERITY_LEVELS, min(count, len(SEVERITY_LEVELS))
        )
        frameworks = random.sample(
            TESTING_FRAMEWORKS, min(count, len(TESTING_FRAMEWORKS))
        )

        if count <= 5:
            n_variants = random.randint(3, min(count + 2, len(templates)))
            chosen = random.sample(templates, min(n_variants, len(templates)))
            combos = list(zip(chosen, environment_types, severity_levels, frameworks))
        else:
            all_combos = [
                (t, env, sev, fw)
                for t in templates
                for env in ENVIRONMENT_TYPES
                for sev in SEVERITY_LEVELS
                for fw in TESTING_FRAMEWORKS
            ]
            random.shuffle(all_combos)
            n_variants = min(count * 2, len(all_combos))
            combos = all_combos[:n_variants]

        for q_template, env_type, sev_level, framework in combos:
            user = q_template.format(
                name=name,
                category=category.replace("_", " ").title(),
                environment_type=env_type,
                severity_level=sev_level,
                framework=framework,
            )

            assistant = f"**{name}** (MITRE: {', '.join(mitre_ids)})\n\n"
            assistant += f"**Category:** {category.replace('_', ' ').title()}\n\n"
            assistant += f"**Environment:** {env_type}\n\n"
            assistant += f"**Severity:** {sev_level.title()}\n\n"
            assistant += f"**Testing Framework:** {framework}\n\n"
            assistant += f"**Description:** {scenario['desc']}\n\n"
            assistant += f"**Attack Steps:**\n{scenario['steps']}\n\n"

            if scenario.get("email_templates"):
                emails = "\n\n---\n\n".join(
                    t.replace("{name}", "Target")
                    .replace(
                        "{phishing_link}", "https://malicious-site.example.com/login"
                    )
                    .replace(
                        "{spoofed_sender}", "IT Security <security@spoofed-domain.com>"
                    )
                    .replace("{fraud_account}", "123456789")
                    .replace("{fraud_routing}", "021000021")
                    .replace("{fraud_bank}", "Fraudulent Bank")
                    .replace("{cfo_name}", "CFO")
                    .replace("{ceo_name}", "CEO")
                    .replace("{authority_name}", "Authority Figure")
                    .replace("{title}", "Title")
                    .replace("{invoice_num}", "INV-2025-001")
                    .replace("{law_firm}", "Law Associates LLP")
                    .replace("{attorney_name}", "J. Attorney")
                    .replace("{badge_number}", "7451")
                    .replace("{tracking_number}", "FX-8842910356")
                    .replace("{position}", "Senior Developer")
                    .replace("{company}", "Acme Corp")
                    .replace("{skill}", "cloud infrastructure")
                    .replace("{project_name}", "Project Atlas")
                    .replace("{recent_event}", "the last team meeting")
                    .replace("{spoofed_vendor_name}", "TechSecure Solutions")
                    .replace("{start_date}", "Monday")
                    .replace("{deadline}", "Friday")
                    .replace("{date}", "January 15, 2025")
                    for t in scenario["email_templates"]
                )
                assistant += f"**Example Email Templates:**\n\n{emails}\n\n"

            if scenario.get("detection"):
                det = "\n".join(f"- {d}" for d in scenario["detection"])
                assistant += f"**Detection Indicators:**\n{det}\n\n"

            if scenario.get("mitigations"):
                mit = "\n".join(f"- {m}" for m in scenario["mitigations"])
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
    parser = argparse.ArgumentParser(
        description="Acquire Phishing/Social Engineering dataset for AttackLM"
    )
    parser.add_argument("--output", default=None, help="Custom output directory")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of variant expansions per scenario (default: 5)",
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
                for cat in [
                    "spear_phishing",
                    "bec",
                    "credential_harvesting",
                    "vishing",
                    "pretexting",
                    "deepfake_se",
                ]:
                    if cat.replace("_", " ").title() in msg["content"]:
                        cat_counts[cat] += 1
                        break

    mitre_ids_all: list[str] = []
    for p in pairs:
        mitre_ids_all.extend(p.get("mitre_ids", []))
    unique_mitre = sorted(set(mitre_ids_all))

    metadata = {
        "name": "phishing",
        "display_name": "Phishing & Social Engineering",
        "category": "social_engineering",
        "mitre_tactic": "TA0001",
        "description": f"Phishing and social engineering dataset covering {len(cat_counts)} categories: spear phishing, BEC, credential harvesting, vishing, pretexting, and deepfake SE. Generated with {args.count} variant expansions per scenario.",
        "source_file": data_file.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "count": len(pairs),
        "sub_sources": {"human": 0, "llm": 0, "synth": len(pairs)},
        "mitre_ids": unique_mitre,
        "environment_types": ENVIRONMENT_TYPES,
        "severity_levels": SEVERITY_LEVELS,
        "testing_frameworks": TESTING_FRAMEWORKS,
        "scenario_count": len(SCENARIOS),
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nPhishing/Social Engineering dataset generated:")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Scenarios: {len(SCENARIOS)}")
    print(f"  Variant count: {args.count}")
    print(f"  Categories: {dict(cat_counts)}")
    print(f"  MITRE IDs: {unique_mitre}")
    print(f"  Environment types: {len(ENVIRONMENT_TYPES)}")
    print(f"  Severity levels: {len(SEVERITY_LEVELS)}")
    print(f"  Testing frameworks: {len(TESTING_FRAMEWORKS)}")
    print(f"  Output: {data_file}")
    print(f"  Metadata: {meta_file}")


if __name__ == "__main__":
    main()
