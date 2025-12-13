"""
PyInstaller hook for ctranslate2

This hook ensures all CTranslate2 DLLs and data files are included in the bundle.
"""

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files
import os

# Collect all DLLs from ctranslate2 package
datas = collect_data_files('ctranslate2')
binaries = collect_dynamic_libs('ctranslate2')

# Manually add specific DLLs that might be missed
try:
    import ctranslate2
    ct2_path = os.path.dirname(ctranslate2.__file__)

    # Add all DLLs from ctranslate2 directory
    for file in os.listdir(ct2_path):
        if file.endswith('.dll'):
            dll_path = os.path.join(ct2_path, file)
            binaries.append((dll_path, 'ctranslate2'))
except Exception as e:
    print(f"Warning: Could not collect ctranslate2 DLLs: {e}")

# Ensure hiddenimports for CUDA support
hiddenimports = [
    'ctranslate2',
    'ctranslate2.converters',
]
