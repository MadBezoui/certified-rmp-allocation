"""Recalcul independant par completion du carre, sans arrondi, pour auditer la borne."""
from fractions import Fraction as Q
import verificateur_quadratique as vq


def bound(D, sigma, weights, tangent_s, yc, yt):
    n=D.n
    k=[Q(0)]*n; a=[Q(0)]*n; b=[Q(0)]*n; const=Q(0)
    for w,obj in zip(weights,D.objs):
        w=w/Q(obj['scale'])
        if obj['kind']=='shortfall':
            k=[u+w*Q(v) for u,v in zip(k,obj['weights'])]
        elif obj['kind']=='movement':
            b=[u+w*Q(v) for u,v in zip(b,obj['weights'])]
        else:
            for group,g in zip(obj['groups'],obj['gammas']):
                if len(group)<2: continue
                coeff=w*Q(g)/len(group)
                mean=sum((tangent_s[i] for i in group),Q(0))/len(group)
                base=coeff*sum(((tangent_s[i]-mean)**2 for i in group),Q(0))
                const+=base
                for i in group:
                    grad=2*coeff*(tangent_s[i]-mean)/D.d[i]
                    a[i]+=grad
                    const-=grad*tangent_s[i]*D.d[i]
    for j,y in enumerate(yc):
        const-=y*D.c[j]
        a=[x+y*z for x,z in zip(a,D.A[j])]
    for j,y in enumerate(yt):
        const+=y*(D.D[j]-sigma[j])
        for i in D.tiers[j]: a[i]-=y
    for i in range(n):
        # Minimise separately on [lo,xp] and [xp,hi] using clipped vertices.
        choices=[]
        for left,right,sign in ((D.lo[i],min(D.hi[i],D.xp[i]),-1),
                                (max(D.lo[i],D.xp[i]),D.hi[i],1)):
            if left>right: continue
            if k[i]:
                vertex=D.d[i]-(a[i]+sign*b[i])/(2*k[i])
                z=max(left,min(right,vertex))
            else:
                z=left if a[i]+sign*b[i]>=0 else right
            choices.append(k[i]*(D.d[i]-z)**2+a[i]*z+b[i]*abs(z-D.xp[i]))
        const+=min(choices)
    return const


def compare(cert, item, refinement=False):
    D=vq.original.Data(cert); sigma=[Q(v) for v in cert['priority']['totals']]
    if refinement:
        weights=[Q(l)+Q(n) for l,n in zip(cert['refinement']['lambda'],item['nu'])]
    else:
        weights=[Q(v) for v in item['mu']]
    tx=[Q(v) for v in item['tangent_x']]; ts=[Q(v) for v in item['tangent_s']]
    yc=[max(Q(v),Q(0)) for v in item['y_cap']]; yt=[max(Q(v),Q(0)) for v in item['y_tier']]
    strict=vq.halfspace_bound(D,sigma,weights,tx,ts,yc,yt)
    independently=bound(D,sigma,weights,ts,yc,yt)
    # Rounding occurs once per coordinate and once per disparity component.
    if strict > independently or independently-strict > Q(1,10**24):
        raise AssertionError('quadratic lower bound disagrees with independent calculation')
