#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Encryption Module - For protecting mask parameters transmission

Uses RSA+AES hybrid encryption scheme:
- Use AES-256-GCM symmetric encryption for large data (mask_params)
- Use RSA-2048 public key to encrypt symmetric key (session_key)
"""

import os
import json
import base64
from typing import Dict, Tuple
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class HybridEncryption:
    """Hybrid encryption system class"""
    
    def __init__(self):
        """Initialize hybrid encryption system"""
        self.backend = default_backend()
        
    # ============ RSA Key Pair Management ============
    
    def generate_rsa_keypair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """
        Generate RSA key pair
        
        Args:
            key_size: RSA key length (default 2048 bits)
            
        Returns:
            (private_key_pem, public_key_pem): Private and public keys in PEM format
        """
        # Generate RSA private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )
        
        # Generate corresponding public key
        public_key = private_key.public_key()
        
        # Serialize to PEM format
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def save_keypair(self, private_key_pem: bytes, public_key_pem: bytes, 
                     private_path: str, public_path: str):
        """
        Save key pair to files
        
        Args:
            private_key_pem: Private key PEM
            public_key_pem: Public key PEM
            private_path: Private key save path
            public_path: Public key save path
        """
        with open(private_path, 'wb') as f:
            f.write(private_key_pem)
        
        with open(public_path, 'wb') as f:
            f.write(public_key_pem)
        
        # Set private key file permissions to owner read-only
        os.chmod(private_path, 0o600)
    
    def load_public_key(self, public_key_path: str):
        """
        Load public key
        
        Args:
            public_key_path: Public key file path
            
        Returns:
            Public key object
        """
        with open(public_key_path, 'rb') as f:
            public_pem = f.read()
        
        public_key = serialization.load_pem_public_key(
            public_pem,
            backend=self.backend
        )
        
        return public_key
    
    def load_private_key(self, private_key_path: str):
        """
        Load private key
        
        Args:
            private_key_path: Private key file path
            
        Returns:
            Private key object
        """
        with open(private_key_path, 'rb') as f:
            private_pem = f.read()
        
        private_key = serialization.load_pem_private_key(
            private_pem,
            password=None,
            backend=self.backend
        )
        
        return private_key
    
    # ============ AES Encryption/Decryption ============
    
    def generate_session_key(self, key_size: int = 32) -> bytes:
        """
        Generate random symmetric key (session key)
        
        Args:
            key_size: Key length in bytes, default 32 bytes = 256 bits
            
        Returns:
            Random session key
        """
        return os.urandom(key_size)
    
    def aes_encrypt(self, plaintext: bytes, session_key: bytes) -> Dict[str, str]:
        """
        Encrypt data using AES-256-GCM
        
        Args:
            plaintext: Plaintext data
            session_key: Symmetric key (32 bytes)
            
        Returns:
            Dictionary containing ciphertext, nonce, tag (Base64 encoded)
        """
        # Generate random nonce (12 bytes for GCM)
        nonce = os.urandom(12)
        
        # Create AES-GCM encryptor
        cipher = Cipher(
            algorithms.AES(session_key),
            modes.GCM(nonce),
            backend=self.backend
        )
        
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Get authentication tag
        tag = encryptor.tag
        
        # Return Base64 encoded result
        return {
            'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'tag': base64.b64encode(tag).decode('utf-8')
        }
    
    def aes_decrypt(self, encrypted_data: Dict[str, str], session_key: bytes) -> bytes:
        """
        Decrypt data using AES-256-GCM
        
        Args:
            encrypted_data: Dictionary containing ciphertext, nonce, tag (Base64 encoded)
            session_key: Symmetric key (32 bytes)
            
        Returns:
            Plaintext data
        """
        # Decode Base64
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        nonce = base64.b64decode(encrypted_data['nonce'])
        tag = base64.b64decode(encrypted_data['tag'])
        
        # Create AES-GCM decryptor
        cipher = Cipher(
            algorithms.AES(session_key),
            modes.GCM(nonce, tag),
            backend=self.backend
        )
        
        decryptor = cipher.decryptor()
        
        # Decrypt data
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    # ============ RSA Encryption/Decryption ============
    
    def rsa_encrypt(self, plaintext: bytes, public_key) -> bytes:
        """
        Encrypt data using RSA public key
        
        Args:
            plaintext: Plaintext data (small data, like session key)
            public_key: RSA public key object
            
        Returns:
            Ciphertext
        """
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return ciphertext
    
    def rsa_decrypt(self, ciphertext: bytes, private_key) -> bytes:
        """
        Decrypt data using RSA private key
        
        Args:
            ciphertext: Ciphertext
            private_key: RSA private key object
            
        Returns:
            Plaintext data
        """
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return plaintext
    
    # ============ Hybrid Encryption (High-level Interface) ============
    
    def hybrid_encrypt(self, mask_params: Dict, public_key_path: str) -> Dict:
        """
        Encrypt mask_params using hybrid encryption scheme
        
        Process:
        1. Generate random session_key (AES key)
        2. Encrypt mask_params with session_key (AES-256-GCM)
        3. Encrypt session_key with receiver public key (RSA-OAEP)
        4. Return encrypted data package
        
        Args:
            mask_params: Masking parameters dictionary
            public_key_path: Receiver public key file path
            
        Returns:
            Encrypted data package
        """
        # 1. Generate session key
        session_key = self.generate_session_key(32)  # 256-bit AES key
        
        # 2. Convert mask_params to JSON string
        params_json = json.dumps(mask_params, ensure_ascii=False)
        params_bytes = params_json.encode('utf-8')
        
        # 3. Encrypt mask_params with AES
        encrypted_params = self.aes_encrypt(params_bytes, session_key)
        
        # 4. Load receiver public key
        public_key = self.load_public_key(public_key_path)
        
        # 5. Encrypt session_key with RSA
        encrypted_session_key = self.rsa_encrypt(session_key, public_key)
        encrypted_session_key_b64 = base64.b64encode(encrypted_session_key).decode('utf-8')
        
        # 6. Build encrypted data package
        encrypted_package = {
            'version': '1.0',
            'encryption_method': 'RSA-2048-OAEP + AES-256-GCM',
            'encrypted_session_key': encrypted_session_key_b64,
            'encrypted_data': encrypted_params,
            'metadata': {
                'identifier': mask_params.get('identifier', 'unknown'),
                'timestamp': mask_params.get('timestamp', 0)
            }
        }
        
        return encrypted_package
    
    def hybrid_decrypt(self, encrypted_package: Dict, private_key_path: str) -> Dict:
        """
        Decrypt mask_params using hybrid encryption scheme
        
        Process:
        1. Decrypt session_key with receiver private key (RSA-OAEP)
        2. Decrypt mask_params with session_key (AES-256-GCM)
        3. Return original mask_params
        
        Args:
            encrypted_package: Encrypted data package
            private_key_path: Receiver private key file path
            
        Returns:
            Original mask_params dictionary
        """
        # 1. Load receiver private key
        private_key = self.load_private_key(private_key_path)
        
        # 2. Decode and decrypt session_key
        encrypted_session_key = base64.b64decode(encrypted_package['encrypted_session_key'])
        session_key = self.rsa_decrypt(encrypted_session_key, private_key)
        
        # 3. Decrypt mask_params with session_key
        encrypted_data = encrypted_package['encrypted_data']
        params_bytes = self.aes_decrypt(encrypted_data, session_key)
        
        # 4. Parse JSON
        params_json = params_bytes.decode('utf-8')
        mask_params = json.loads(params_json)
        
        return mask_params
    
    # ============ Utility Functions ============
    
    def save_encrypted_params(self, encrypted_package: Dict, output_path: str):
        """
        Save encrypted parameter package to file
        
        Args:
            encrypted_package: Encrypted data package
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(encrypted_package, f, indent=2, ensure_ascii=False)
    
    def load_encrypted_params(self, encrypted_path: str) -> Dict:
        """
        Load encrypted parameter package from file
        
        Args:
            encrypted_path: Encrypted file path
            
        Returns:
            Encrypted data package
        """
        with open(encrypted_path, 'r', encoding='utf-8') as f:
            encrypted_package = json.load(f)
        
        return encrypted_package


def batch_encrypt_params():
    """Batch encrypt all JSON parameter files in params directory"""
    print("=== Batch Encryption of Parameter Files ===\n")
    
    # Setup directories
    import os
    import glob
    from pathlib import Path
    
    base_output_dir = "dataset/output"
    encryption_dir = os.path.join(base_output_dir, "encryption")
    keys_dir = os.path.join(encryption_dir, "keys")
    params_dir = os.path.join(encryption_dir, "params")
    
    os.makedirs(keys_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    
    # Initialize encryption system
    crypto = HybridEncryption()
    
    # Check if keypair exists, if not generate one
    public_key_files = list(Path(keys_dir).glob("*_public.pem"))
    private_key_files = list(Path(keys_dir).glob("*_private.pem"))
    
    if not public_key_files or not private_key_files:
        print("1. No existing keypair found. Generating RSA key pair...")
        private_key_pem, public_key_pem = crypto.generate_rsa_keypair(2048)
        
        # Save key pair to dataset/output/encryption/keys directory
        private_key_path = os.path.join(keys_dir, 'batch_encryption_private.pem')
        public_key_path = os.path.join(keys_dir, 'batch_encryption_public.pem')
        
        crypto.save_keypair(
            private_key_pem, 
            public_key_pem,
            private_key_path,
            public_key_path
        )
        print(f"   ✓ Key pair saved to {private_key_path} and {public_key_path}\n")
    else:
        # Use existing keypair
        public_key_path = str(public_key_files[0])
        private_key_path = str(private_key_files[0])
        print(f"1. Using existing keypair:")
        print(f"   Public key: {public_key_path}")
        print(f"   Private key: {private_key_path}\n")
    
    # 2. Find all JSON parameter files to encrypt
    print("2. Scanning for parameter files to encrypt...")
    json_files = list(Path(params_dir).glob("*_mask_params_*.json"))
    
    if not json_files:
        print("   ⚠️  No parameter files found in params directory.")
        print("   Please run audio_privacy_system.py first to generate parameter files.\n")
        
        # Create a demo parameter file for testing
        print("   Creating demo parameter file for testing...")
        demo_params = {
            'seed': 3169164413,
            'length': 92501,
            'sample_rate': 16000,
            'mask_type': 'multi_tone',
            'scale_factor': 0.47830966114997864,
            'timestamp': 1760412807,
            'identifier': 'demo-batch-encryption',
            'version': '1.0',
            'target_snr_db': 0.0
        }
        demo_file = Path(params_dir) / 'demo_mask_params.json'
        with open(demo_file, 'w', encoding='utf-8') as f:
            json.dump(demo_params, f, indent=2, ensure_ascii=False)
        json_files = [demo_file]
        print(f"   ✓ Demo file created: {demo_file}")
    
    print(f"   Found {len(json_files)} parameter file(s) to encrypt:\n")
    for i, file_path in enumerate(json_files, 1):
        print(f"   {i}. {file_path.name}")
    print()
    
    # 3. Batch encrypt all parameter files
    print("3. Batch encrypting parameter files...")
    encrypted_files = []
    failed_files = []
    
    for file_path in json_files:
        try:
            # Load the parameter file
            with open(file_path, 'r', encoding='utf-8') as f:
                mask_params = json.load(f)
            
            # Check if already encrypted
            if 'encryption_method' in mask_params:
                print(f"   ⚠️  {file_path.name} is already encrypted, skipping...")
                continue
            
            # Encrypt the parameters
            encrypted_package = crypto.hybrid_encrypt(mask_params, public_key_path)
            
            # Save encrypted version (replace original)
            crypto.save_encrypted_params(encrypted_package, str(file_path))
            encrypted_files.append(file_path.name)
            
            print(f"   ✓ Encrypted: {file_path.name}")
            
        except Exception as e:
            print(f"   ✗ Failed to encrypt {file_path.name}: {e}")
            failed_files.append(file_path.name)
    
    print(f"\n   Encryption Summary:")
    print(f"   - Successfully encrypted: {len(encrypted_files)} files")
    print(f"   - Failed: {len(failed_files)} files")
    if encrypted_files:
        print(f"   - Encrypted files: {', '.join(encrypted_files)}")
    if failed_files:
        print(f"   - Failed files: {', '.join(failed_files)}")
    
    # 4. Verification (test decrypt one file if any were encrypted)
    if encrypted_files:
        print(f"\n4. Verifying encryption by testing decryption...")
        try:
            test_file = json_files[0] if json_files else None
            if test_file:
                encrypted_package_loaded = crypto.load_encrypted_params(str(test_file))
                decrypted_params = crypto.hybrid_decrypt(encrypted_package_loaded, private_key_path)
                print(f"   ✓ Verification successful! Decryption test passed for {test_file.name}")
        except Exception as e:
            print(f"   ✗ Verification failed: {e}")
    
    print(f"\n=== Batch Encryption Completed ===")
    print(f"All parameter files in {params_dir} have been encrypted using:")
    print(f"- Public key: {public_key_path}")
    print(f"- Private key: {private_key_path}")


def demo_usage():
    """Demonstrate hybrid encryption usage (original demo)"""
    print("=== Hybrid Encryption Module Demo ===\n")
    
    # Create organized output directories if they don't exist
    import os
    base_output_dir = "dataset/output"
    encryption_dir = os.path.join(base_output_dir, "encryption")
    keys_dir = os.path.join(encryption_dir, "keys")
    params_dir = os.path.join(encryption_dir, "params")
    
    os.makedirs(keys_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    
    # Initialize encryption system
    crypto = HybridEncryption()
    
    # 1. Generate key pair (receiver)
    print("1. Generating RSA key pair...")
    private_key_pem, public_key_pem = crypto.generate_rsa_keypair(2048)
    
    # Save key pair to dataset/output/encryption/keys directory
    private_key_path = os.path.join(keys_dir, 'receiver_private.pem')
    public_key_path = os.path.join(keys_dir, 'receiver_public.pem')
    
    crypto.save_keypair(
        private_key_pem, 
        public_key_pem,
        private_key_path,
        public_key_path
    )
    print(f"   ✓ Key pair saved to {private_key_path} and {public_key_path}\n")
    
    # 2. Simulate mask_params
    print("2. Preparing test data (mask_params)...")
    mask_params = {
        'seed': 3169164413,
        'length': 92501,
        'sample_rate': 16000,
        'mask_type': 'multi_tone',
        'scale_factor': 0.47830966114997864,
        'timestamp': 1760412807,
        'identifier': '48b2112f-41cd-4f7b-8b8b-376f0c215fb7',
        'version': '1.0',
        'target_snr_db': 0.0
    }
    print(f"   Original data size: {len(json.dumps(mask_params))} bytes\n")
    
    # 3. Encryption (sender)
    print("3. Encrypting data using hybrid encryption...")
    encrypted_package = crypto.hybrid_encrypt(mask_params, public_key_path)
    encrypted_params_path = os.path.join(params_dir, 'encrypted_params.json')
    crypto.save_encrypted_params(encrypted_package, encrypted_params_path)
    print(f"   ✓ Encryption completed, saved to {encrypted_params_path}")
    print(f"   Encrypted data size: {len(json.dumps(encrypted_package))} bytes\n")
    
    # 4. Decryption (receiver)
    print("4. Recovering data using hybrid decryption...")
    encrypted_package_loaded = crypto.load_encrypted_params(encrypted_params_path)
    decrypted_params = crypto.hybrid_decrypt(encrypted_package_loaded, private_key_path)
    print("   ✓ Decryption completed\n")
    
    # 5. Verification
    print("5. Verifying data integrity...")
    if mask_params == decrypted_params:
        print("   ✓ Verification successful! Decrypted data matches original data perfectly\n")
    else:
        print("   ✗ Verification failed! Data mismatch\n")
    
    print("=== Demo completed ===")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_usage()
    else:
        # Default: run batch encryption
        batch_encrypt_params()

