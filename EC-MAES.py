from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

# Generate ECC private and public keys
private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
public_key = private_key.public_key()

# Serialize and save the private key
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption()
)

with open('ecc_private_key.pem', 'wb') as f:
    f.write(private_pem)

# Serialize and save the public key
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

with open('ecc_public_key.pem', 'wb') as f:
    f.write(public_pem)

# Load the keys (usually done in separate processes)
with open('ecc_private_key.pem', 'rb') as f:
    private_pem = f.read()
    private_key = serialization.load_pem_private_key(private_pem, password=None, backend=default_backend())

with open('ecc_public_key.pem', 'rb') as f:
    public_pem = f.read()
    public_key = serialization.load_pem_public_key(public_pem, backend=default_backend())

# Encrypt a message using the public key
message = b'This is a secret message.'
ciphertext = public_key.encrypt(message, ec.ECIES())

# Decrypt the message using the private key
decrypted_message = private_key.decrypt(ciphertext, ec.ECIES())
print("Decrypted Message:", decrypted_message.decode())
