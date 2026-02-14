import argparse
import base64
import json
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


def encrypt_file(input_csv: Path, public_key_pem: Path, output_enc: Path) -> None:
    data = input_csv.read_bytes()
    public_key = serialization.load_pem_public_key(public_key_pem.read_bytes())

    # Hybrid scheme: encrypt payload with symmetric key, then encrypt symmetric key with RSA.
    symmetric_key = Fernet.generate_key()
    token = Fernet(symmetric_key).encrypt(data)
    enc_key = public_key.encrypt(
        symmetric_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )

    payload = {
        "version": 1,
        "alg": "RSA_OAEP_SHA256+FERNET",
        "enc_key_b64": base64.b64encode(enc_key).decode("ascii"),
        "ciphertext_b64": base64.b64encode(token).decode("ascii"),
    }
    output_enc.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt predictions.csv for secure submission.")
    parser.add_argument("input_csv", type=Path, help="Path to predictions.csv")
    parser.add_argument("public_key_pem", type=Path, help="Path to public_key.pem")
    parser.add_argument("output_enc", type=Path, help="Output encrypted file path, e.g. predictions.csv.enc")
    args = parser.parse_args()

    encrypt_file(args.input_csv, args.public_key_pem, args.output_enc)
    print(f"Encrypted submission written to: {args.output_enc}")


if __name__ == "__main__":
    main()
