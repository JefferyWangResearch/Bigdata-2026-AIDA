from setup import *
from pyproj import Geod

def main():
    rng=np.random.default_rng(SEED);theta=rng.uniform(0,2*np.pi,200);speed=rng.uniform(.1,12,200);bearing=.7;rotation=1.3
    A=np.mean(np.cos(theta-bearing));P=np.mean(np.sin(theta-bearing));actual=np.mean(np.cos(theta+rotation-bearing));assert np.isclose(actual,A*np.cos(rotation)-P*np.sin(rotation),atol=1e-12)
    assert np.isclose(np.mean(np.cos(theta+np.pi-bearing)),-A,atol=1e-12)
    pressure=np.mean(speed*np.maximum(0,np.cos(theta-bearing)));reverse=np.mean(speed*np.maximum(0,np.cos(theta+np.pi-bearing)))
    assert np.isclose(pressure-reverse,np.mean(speed*np.cos(theta-bearing)),atol=1e-12)
    geod=Geod(ellps='WGS84');east,_,_=geod.inv(-120,35,-119,35);assert 89<east<91
    north,_,_=geod.inv(-120,35,-120,36);assert np.isclose(north,0)
    # Logical source and negative labels at t: current-year sources prohibited.
    first=np.array([2016,2017,2018,2019,np.nan]);t=2018;source=first<t;pos=first==t;negative=~source&~pos
    assert source.tolist()==[True,True,False,False,False]
    assert negative.tolist()==[False,False,False,True,True]
    result=dict(status='PASS',checks=['analytical rotation equals direct daily cosine','reversal identity','positive-WSP real/reverse decomposition','geodesic bearing east/north','source and candidate time eligibility'],rotation_absolute_error=float(abs(actual-(A*np.cos(rotation)-P*np.sin(rotation)))))
    (R/'logs/task2/exposure_math_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
