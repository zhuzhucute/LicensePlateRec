# -*- mode: python ; coding: utf-8 -*-
import sys
import os.path as osp
sys.setrecursionlimit(5000)
 
block_cipher = None
 
 
SETUP_DIR = 'E:\\GitHub\\LicensePlateRec\\'


a = Analysis(['main.py',
             'about.py',
             'detect.py',
             'ui_dialog.py',
             'ui_error.py',
             'ui_function.py',
             'ui_main.py',
             'E:\\GitHub\\LicensePlateRec\\utils\\datasets.py',
             'E:\\GitHub\\LicensePlateRec\\utils\\google_utils.py',
             'E:\\GitHub\\LicensePlateRec\\utils\\torch_utils.py',
             'E:\\GitHub\\LicensePlateRec\\utils\\useful.py',
             'E:\\GitHub\\LicensePlateRec\\models\\common.py',
             'E:\\GitHub\\LicensePlateRec\\models\\experimental.py',
             'E:\\GitHub\\LicensePlateRec\\models\\export.py',
             'E:\\GitHub\\LicensePlateRec\\models\\LPRNet.py',
             'E:\\GitHub\\LicensePlateRec\\models\\yolo.py'],
             pathex=['E:\\GitHub\\LicensePlateRec'],
             binaries=[],
             datas=[(SETUP_DIR+'icons','icons'),(SETUP_DIR+'weights','weights'),(SETUP_DIR+'models','models')],
             hiddenimports=['PySide2', 'numpy.core.multiarray'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='LicensePlateRec v1.0.0-alpha-x64',
          debug=False,
          strip=False,
          upx=True,
          console=False, 
          icon='E:\\GitHub\\LicensePlateRec\\icons\\car.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='LicensePlateRec v1.0.0-alpha')
