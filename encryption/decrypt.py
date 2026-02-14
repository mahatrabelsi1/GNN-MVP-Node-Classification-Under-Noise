import argparse
import base64
import json
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


def decrypt_file(input_enc: Path, private_key_pem: Path, output_csv: Path) -> None:
    payload = json.loads(input_enc.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported encrypted payload version")

    private_key = serialization.load_pem_private_key(private_key_pem.read_bytes(), password=None)
    enc_key = base64.b64decode(payload["enc_key_b64"])
    token = base64.b64decode(payload["ciphertext_b64"])

    symmetric_key = private_key.decrypt(
        enc_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    data = Fernet(symmetric_key).decrypt(token)
    output_csv.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decrypt encrypted predictions.")
    parser.add_argument("input_enc", type=Path, help="Path to predictions.csv.enc")
    parser.add_argument("private_key_pem", type=Path, help="Path to private key PEM")
    parser.add_argument("output_csv", type=Path, help="Output decrypted predictions.csv path")
    args = parser.parse_args()

    decrypt_file(args.input_enc, args.private_key_pem, args.output_csv)
    print(f"Decrypted submission written to: {args.output_csv}")


if __name__ == "__main__":
    main()
