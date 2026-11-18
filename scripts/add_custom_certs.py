# Add custom TLS certificates to the pixi/docker environment

from pathlib import Path

import certifi

CUSTOM_CERTS_DIR = Path("certs")


def add_certs():
    """Add custom certs to the certifi bundle."""
    sys_certs = Path(certifi.where())
    bundle_file = CUSTOM_CERTS_DIR / "bundle.pem"

    sources = [sys_certs] + sorted(CUSTOM_CERTS_DIR.glob("*.crt"))
    if len(sources) == 1:
        print("No certificates to add.")
        return
    print(f"Appending {len(sources) - 1} certificates to {sys_certs}")
    tmp = bundle_file.with_suffix(".pem.tmp")
    with tmp.open("w", encoding="ascii") as out:
        for src in sources:
            out.write(src.read_text(encoding="ascii").rstrip() + "\n")
    tmp.replace(sys_certs)
    print("Completed.")


if __name__ == "__main__":
    add_certs()
