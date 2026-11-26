"""Read only published participant CSV members of the PhysioNet archive."""
import concurrent.futures, io, re, struct, subprocess, sys, zipfile
from pathlib import Path

URL='https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/CGMacros_dateshifted365.zip'

def request(begin,end):
    size=end-begin+1
    data=subprocess.check_output(['curl','-fsSL','--retry','5','--retry-all-errors','--retry-delay','1','--max-time','60','--max-filesize',str(size),'--range',f'{begin}-{end}',URL])
    if len(data)!=size:raise ValueError('server did not honor exact range')
    return data

def main(root):
    hdr=subprocess.check_output(['curl','-fsSI',URL],text=True)
    size=int(re.findall(r'(?im)^Content-Length:\s*(\d+)',hdr)[-1])
    end=request(size-65536,size-1)
    eoc=end.rfind(b'PK\x05\x06')
    if eoc<0:raise ValueError('missing ZIP directory')
    off=struct.unpack_from('<I',end,eoc+16)[0]
    # Sparse local image: only the directory and required CSV regions occupy disk.
    archive=root/'catalog.zip'
    with archive.open('wb') as handle:
        handle.seek(off);handle.write(request(off,size-1))
    with zipfile.ZipFile(archive) as z:
        files=[r for r in z.infolist() if re.fullmatch(r'(?:.*/)?CGMacros-\d+\.csv',r.filename) and '__MACOSX' not in r.filename]
    assert files and len({Path(f.filename).name for f in files})==len(files)
    def fetch(f):
        header=request(f.header_offset,f.header_offset+29)
        n,x=struct.unpack_from('<HH',header,26)
        end=f.header_offset+30+n+x+f.compress_size-1
        return f.header_offset,request(f.header_offset,end)
    with archive.open('r+b') as out,concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for start,part in pool.map(fetch,files):
            out.seek(start);out.write(part)
    with zipfile.ZipFile(archive) as z:
        for f in files:(root/Path(f.filename).name).write_bytes(z.read(f))
    print('participant files',len(files))
if __name__=='__main__':main(Path(sys.argv[1]))
