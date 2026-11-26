import csv,datetime as dt,json,math,sys
from collections import defaultdict
from pathlib import Path

def stamp(val):
    for fmt in ('%m/%d/%Y %H:%M','%m/%d/%Y %H:%M:%S','%Y-%m-%d %H:%M:%S','%Y-%m-%d %H:%M'):
        try:return dt.datetime.strptime(val.strip(),fmt)
        except ValueError:pass
    raise ValueError('Unsupported timestamp format')

def main():
    files=sorted(Path(sys.argv[1]).glob('CGMacros-*.csv'));data=[]
    assert len(files)==45
    for f in files:
        readings=defaultdict(set)
        with f.open(encoding='utf-8-sig',newline='') as handle:
            rows=csv.DictReader(handle)
            assert {'Timestamp','Dexcom GL'}<=set(rows.fieldnames)
            for row in rows:
                try:x=float(row['Dexcom GL'])
                except (ValueError,TypeError):continue
                if math.isfinite(x) and 40<=x<=400:
                    readings[stamp(row['Timestamp'])].add(x)
        points=[[t.date().toordinal(),58 if 6<=t.hour<10 else 62 if 18<=t.hour<22 else 48,next(iter(s))] for t,s in sorted(readings.items()) if len(s)==1]
        if points:data.append(dict(person=str(len(data)),rows=points))
    Path(sys.argv[2]).write_text(json.dumps(data,separators=(',',':')))
    print('files',len(files),'participants retained',len(data),'published values',sum(len(v['rows']) for v in data))
if __name__=='__main__':main()
