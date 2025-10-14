#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合加密模块 - 用于保护掩蔽参数传输
Hybrid Encryption Module - For protecting mask parameters transmission

使用RSA+AES混合加密方案：
- 用AES-256-GCM对称加密大数据（mask_params）
- 用RSA-2048公钥加密对称密钥（session_key）
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
    """混合加密系统类"""
    
    def __init__(self):
        """初始化混合加密系统"""
        self.backend = default_backend()
        
    # ============ RSA密钥对管理 ============
    
    def generate_rsa_keypair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """
        生成RSA密钥对
        
        Args:
            key_size: RSA密钥长度（默认2048位）
            
        Returns:
            (private_key_pem, public_key_pem): 私钥和公钥的PEM格式
        """
        # 生成RSA私钥
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )
        
        # 生成对应的公钥
        public_key = private_key.public_key()
        
        # 序列化为PEM格式
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
        保存密钥对到文件
        
        Args:
            private_key_pem: 私钥PEM
            public_key_pem: 公钥PEM
            private_path: 私钥保存路径
            public_path: 公钥保存路径
        """
        with open(private_path, 'wb') as f:
            f.write(private_key_pem)
        
        with open(public_path, 'wb') as f:
            f.write(public_key_pem)
        
        # 设置私钥文件权限为只有所有者可读
        os.chmod(private_path, 0o600)
    
    def load_public_key(self, public_key_path: str):
        """
        加载公钥
        
        Args:
            public_key_path: 公钥文件路径
            
        Returns:
            公钥对象
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
        加载私钥
        
        Args:
            private_key_path: 私钥文件路径
            
        Returns:
            私钥对象
        """
        with open(private_key_path, 'rb') as f:
            private_pem = f.read()
        
        private_key = serialization.load_pem_private_key(
            private_pem,
            password=None,
            backend=self.backend
        )
        
        return private_key
    
    # ============ AES加密/解密 ============
    
    def generate_session_key(self, key_size: int = 32) -> bytes:
        """
        生成随机对称密钥（session key）
        
        Args:
            key_size: 密钥长度（字节），默认32字节=256位
            
        Returns:
            随机session key
        """
        return os.urandom(key_size)
    
    def aes_encrypt(self, plaintext: bytes, session_key: bytes) -> Dict[str, str]:
        """
        使用AES-256-GCM加密数据
        
        Args:
            plaintext: 明文数据
            session_key: 对称密钥（32字节）
            
        Returns:
            包含密文、nonce、tag的字典（Base64编码）
        """
        # 生成随机nonce（12字节适用于GCM）
        nonce = os.urandom(12)
        
        # 创建AES-GCM加密器
        cipher = Cipher(
            algorithms.AES(session_key),
            modes.GCM(nonce),
            backend=self.backend
        )
        
        encryptor = cipher.encryptor()
        
        # 加密数据
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # 获取认证标签
        tag = encryptor.tag
        
        # 返回Base64编码的结果
        return {
            'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'tag': base64.b64encode(tag).decode('utf-8')
        }
    
    def aes_decrypt(self, encrypted_data: Dict[str, str], session_key: bytes) -> bytes:
        """
        使用AES-256-GCM解密数据
        
        Args:
            encrypted_data: 包含密文、nonce、tag的字典（Base64编码）
            session_key: 对称密钥（32字节）
            
        Returns:
            明文数据
        """
        # 解码Base64
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        nonce = base64.b64decode(encrypted_data['nonce'])
        tag = base64.b64decode(encrypted_data['tag'])
        
        # 创建AES-GCM解密器
        cipher = Cipher(
            algorithms.AES(session_key),
            modes.GCM(nonce, tag),
            backend=self.backend
        )
        
        decryptor = cipher.decryptor()
        
        # 解密数据
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    # ============ RSA加密/解密 ============
    
    def rsa_encrypt(self, plaintext: bytes, public_key) -> bytes:
        """
        使用RSA公钥加密数据
        
        Args:
            plaintext: 明文数据（小数据，如session key）
            public_key: RSA公钥对象
            
        Returns:
            密文
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
        使用RSA私钥解密数据
        
        Args:
            ciphertext: 密文
            private_key: RSA私钥对象
            
        Returns:
            明文数据
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
    
    # ============ 混合加密（高层接口）============
    
    def hybrid_encrypt(self, mask_params: Dict, public_key_path: str) -> Dict:
        """
        使用混合加密方案加密mask_params
        
        流程：
        1. 生成随机session_key（AES密钥）
        2. 用session_key加密mask_params（AES-256-GCM）
        3. 用接收方公钥加密session_key（RSA-OAEP）
        4. 返回加密后的数据包
        
        Args:
            mask_params: 掩蔽参数字典
            public_key_path: 接收方公钥文件路径
            
        Returns:
            加密后的数据包
        """
        # 1. 生成session key
        session_key = self.generate_session_key(32)  # 256-bit AES key
        
        # 2. 将mask_params转为JSON字符串
        params_json = json.dumps(mask_params, ensure_ascii=False)
        params_bytes = params_json.encode('utf-8')
        
        # 3. 用AES加密mask_params
        encrypted_params = self.aes_encrypt(params_bytes, session_key)
        
        # 4. 加载接收方公钥
        public_key = self.load_public_key(public_key_path)
        
        # 5. 用RSA加密session_key
        encrypted_session_key = self.rsa_encrypt(session_key, public_key)
        encrypted_session_key_b64 = base64.b64encode(encrypted_session_key).decode('utf-8')
        
        # 6. 构建加密数据包
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
        使用混合加密方案解密mask_params
        
        流程：
        1. 用接收方私钥解密session_key（RSA-OAEP）
        2. 用session_key解密mask_params（AES-256-GCM）
        3. 返回原始mask_params
        
        Args:
            encrypted_package: 加密数据包
            private_key_path: 接收方私钥文件路径
            
        Returns:
            原始mask_params字典
        """
        # 1. 加载接收方私钥
        private_key = self.load_private_key(private_key_path)
        
        # 2. 解码并解密session_key
        encrypted_session_key = base64.b64decode(encrypted_package['encrypted_session_key'])
        session_key = self.rsa_decrypt(encrypted_session_key, private_key)
        
        # 3. 用session_key解密mask_params
        encrypted_data = encrypted_package['encrypted_data']
        params_bytes = self.aes_decrypt(encrypted_data, session_key)
        
        # 4. 解析JSON
        params_json = params_bytes.decode('utf-8')
        mask_params = json.loads(params_json)
        
        return mask_params
    
    # ============ 便捷函数 ============
    
    def save_encrypted_params(self, encrypted_package: Dict, output_path: str):
        """
        保存加密后的参数包到文件
        
        Args:
            encrypted_package: 加密数据包
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(encrypted_package, f, indent=2, ensure_ascii=False)
    
    def load_encrypted_params(self, encrypted_path: str) -> Dict:
        """
        从文件加载加密参数包
        
        Args:
            encrypted_path: 加密文件路径
            
        Returns:
            加密数据包
        """
        with open(encrypted_path, 'r', encoding='utf-8') as f:
            encrypted_package = json.load(f)
        
        return encrypted_package


def demo_usage():
    """演示混合加密的使用方法"""
    print("=== 混合加密模块演示 ===\n")
    
    # 初始化加密系统
    crypto = HybridEncryption()
    
    # 1. 生成密钥对（接收方）
    print("1. 生成RSA密钥对...")
    private_key_pem, public_key_pem = crypto.generate_rsa_keypair(2048)
    
    # 保存密钥对
    crypto.save_keypair(
        private_key_pem, 
        public_key_pem,
        'receiver_private.pem',
        'receiver_public.pem'
    )
    print("   ✓ 密钥对已保存到 receiver_private.pem 和 receiver_public.pem\n")
    
    # 2. 模拟mask_params
    print("2. 准备测试数据（mask_params）...")
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
    print(f"   原始数据大小: {len(json.dumps(mask_params))} 字节\n")
    
    # 3. 加密（发送方）
    print("3. 使用混合加密加密数据...")
    encrypted_package = crypto.hybrid_encrypt(mask_params, 'receiver_public.pem')
    crypto.save_encrypted_params(encrypted_package, 'encrypted_params.json')
    print("   ✓ 加密完成，已保存到 encrypted_params.json")
    print(f"   加密后数据大小: {len(json.dumps(encrypted_package))} 字节\n")
    
    # 4. 解密（接收方）
    print("4. 使用混合解密恢复数据...")
    encrypted_package_loaded = crypto.load_encrypted_params('encrypted_params.json')
    decrypted_params = crypto.hybrid_decrypt(encrypted_package_loaded, 'receiver_private.pem')
    print("   ✓ 解密完成\n")
    
    # 5. 验证
    print("5. 验证数据完整性...")
    if mask_params == decrypted_params:
        print("   ✓ 验证成功！解密后的数据与原始数据完全一致\n")
    else:
        print("   ✗ 验证失败！数据不一致\n")
    
    print("=== 演示完成 ===")


if __name__ == "__main__":
    demo_usage()

