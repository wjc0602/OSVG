from bypy import ByPy
bp=ByPy()
localpath = 'outputs/flickr'
remotepath = 'OSVG/flickr'
bp.upload(localpath, remotepath)