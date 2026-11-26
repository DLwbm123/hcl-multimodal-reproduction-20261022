"""Import only CGM and timestamp; no other clinical columns leave the workbook reader."""
import collections,datetime as dt,io,json,math,re,sys,zipfile
from pathlib import Path
import openpyxl,xlrd

def records(z,n):
    if n.endswith('.xls'):
        book=xlrd.open_workbook(file_contents=z.read(n));sheet=book.sheet_by_index(0)
        hdr=sheet.row_values(0);i,j=columns(hdr)
        if j is None:return
        for r in range(1,sheet.nrows):
            cell=sheet.cell(r,i);t=cell.value
            if cell.ctype==xlrd.XL_CELL_DATE:t=xlrd.xldate_as_datetime(t,book.datemode)
            yield t,sheet.cell_value(r,j)
    else:
        book=openpyxl.load_workbook(io.BytesIO(z.read(n)),read_only=True,data_only=True)
        try:
            rows=book.active.values;i,j=columns(next(rows))
            if j is None:return
            for r in rows:yield r[i],r[j]
        finally:book.close()

def columns(h):
    norm=[re.sub(r'\s+','',str(s)).lower() for s in h]
    if 'cgm(mg/dl)' not in norm and 'cgm' in norm:
        return norm.index('date'),None
    return norm.index('date'),norm.index('cgm(mg/dl)')

def main():
    visits=[]
    with zipfile.ZipFile(sys.argv[1]) as z:
        for n in sorted(z.namelist()):
            m=re.fullmatch(r'Shanghai_T([12])DM/(\d+)_([^/]+)\.xlsx?',n)
            if not m:continue
            unique=collections.defaultdict(set)
            for t,val in records(z,n):
                try:
                    if not isinstance(t,dt.datetime):
                        t=dt.datetime.fromisoformat(str(t).strip().replace('/','-'))
                    x=float(val)
                except (ValueError,TypeError):continue
                if math.isfinite(x):unique[t].add(x)
            data=[]
            for t,vs in sorted(unique.items()):
                if len(vs)==1:data.append([t.date().toordinal(),58 if 6<=t.hour<10 else 62 if 18<=t.hour<22 else 48,next(iter(vs))])
            if data:visits.append(dict(type=m[1],person=m[1]+'-'+m[2],rows=data))
    assert visits
    Path(sys.argv[2]).write_text(json.dumps(visits,separators=(',',':')))
    print('visits',len(visits),'people',len({v['person'] for v in visits}),'readings',sum(len(v['rows']) for v in visits))
if __name__=='__main__':main()
