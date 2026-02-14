import argparse
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def generate_keys(public_key_out: Path, private_key_out: Path, key_size: int = 4096) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    private_key_out.write_bytes(private_pem)
    public_key_out.write_bytes(public_pem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RSA key pair for submission encryption.")
    parser.add_argument("--public", default="encryption/public_key.pem", help="Public key output path")
    parser.add_argument("--private", default="encryption/private_key.pem", help="Private key output path")
    parser.add_argument("--key-size", type=int, default=4096, help="RSA key size")
    args = parser.parse_args()

    public_path = Path(args.public)
    private_path = Path(args.private)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)

    generate_keys(public_path, private_path, key_size=args.key_size)
    print(f"Public key: {public_path}")
    print(f"Private key: {private_path}")


if __name__ == "__main__":
    main()
