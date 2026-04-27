## Security
The `python:3.10-slim` base image currently flags a high-severity vulnerability related to unpacking `.whl` (wheel) files. However, this exploit requires manually unpacking an untrusted, malicious wheel file. As long as we are installing standard packages directly from PyPI using `pip install <package-name>` or a `requirements.txt` file, we are not at risk.

visit https://scout.docker.com/v/CVE-2026-24049 for more details.