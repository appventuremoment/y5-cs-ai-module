from pwn import *
from Crypto.Util.number import *

nlst = []
ctlst = []

for x in range(50):
    conn = remote('146.190.6.88', 30005)
    n = int(conn.recvline().decode().split('is: ')[1][:-1])
    ct = int(conn.recvline().decode().split('is: ')[1][:-1])
    nlst.append(n)
    ctlst.append(ct)
    print(nlst)
print(ctlst)
conn.close()
