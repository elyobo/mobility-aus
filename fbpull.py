import os

import fbcode

if not __name__ == '__main__':
    raise Exception

repoPath = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(repoPath, 'fbids.txt'), 'r') as f:
    fbids = f.read().split('\n')[:-1]
with open(os.path.join(repoPath, '.credentials.txt'), 'r') as f:
    loginName, loginPass = f.read().strip('\n').split(' ')

# fbids = fbids[:1]
# fbids = fbids[-1:]
print(f"Pulling fbids {fbids}...")
fbcode.pull_datas(fbids, loginName, loginPass)
print("Pulled.")
