import numpy as np

def metrics(y,pred,weights=None):
    y=np.asarray(y,float);pred=np.asarray(pred,float)
    good=np.isfinite(y)&np.isfinite(pred);y=y[good];pred=pred[good]
    if not len(y):return {'n':0}
    e=pred-y;ae=np.abs(e);positive=y>0;den=np.abs(y)+np.abs(pred)
    out={'n':len(y),'MAE':float(ae.mean()),'RMSE':float(np.sqrt(np.mean(e**2))),
       'MAPE':float(100*np.mean(ae[positive]/y[positive])) if positive.any() else None,
       'sMAPE':float(np.mean(np.divide(200*ae,den,out=np.zeros_like(ae),where=den!=0))),
       'WAPE':float(100*ae.sum()/np.abs(y).sum()) if np.abs(y).sum()>0 else None,
       'Bias':float(e.mean()),'MAPE_positive_actuals':int(positive.sum()),'MAPE_excluded':int((~positive).sum()),
       'MAPE_coverage':float(positive.mean()),'sMAPE_both_zero':int((den==0).sum()),'WMAE':None,'NWRMSLE':None}
    if weights is not None:
        w=np.asarray(weights,float)[good]
        assert (w>0).all() and (y>=0).all() and (pred>=0).all()
        out['WMAE']=float(np.sum(w*ae)/w.sum())
        out['NWRMSLE']=float(np.sqrt(np.sum(w*(np.log1p(pred)-np.log1p(y))**2)/w.sum()))
    return out
