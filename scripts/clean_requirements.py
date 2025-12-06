
import re

BAD_PACKAGES = [
    'pyobjc', 'appnope', 'sounddevice', 'pynput', 'mss', 
    'pyautogui', 'keyboard', 'mouse', 'xattr', 'pywin32', 
    'windows-curses', 'winshell', 'wmi'
]

def clean_requirements():
    with open('requirements.txt', 'r') as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            cleaned.append(line)
            continue
        
        # Split package name from version
        parts = re.split(r'[=<>!~]', line, 1)
        pkg_name = parts[0].strip()
        
        # Filter bad packages (partial match to catch pyobjc-core etc)
        is_bad = False
        for bad in BAD_PACKAGES:
            if bad.lower() in pkg_name.lower():
                is_bad = True
                break
        
        if is_bad:
            print(f"Removing: {line}")
            continue

        # Keep package name only (unpin version)
        cleaned.append(pkg_name)

    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(cleaned) + '\n')
    
    print("requirements.txt cleaned and unpinned.")

if __name__ == '__main__':
    clean_requirements()
