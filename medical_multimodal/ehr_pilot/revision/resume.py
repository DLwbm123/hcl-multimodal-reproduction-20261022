"""Retry the unfinished read-only requests with a longer transport timeout."""
import revise as w

if __name__=='__main__':
    get=w.p.session.get
    def patient_read(url, **kwargs):
        kwargs['timeout']=120
        return get(url, **kwargs)
    w.p.session.get=patient_read
    w.experiment('valid',9)
    w.report()
