# 1. Simple models 

    1. LH 1 (remove from comparison)
        1 rhat at 1.01
        Trash fit (CO order is flat)

    2. LH 2
        Converged
        Trash fit (Bad ELPD) 

    3. **ER 2**
        Converged 
        Best model (Best ELPD)
        OH order is not good

    4.  LH 3
        3 rhats at 1.02
        Good ELPD
        OH binds too strongly
        COOH coverage too high
        OH order is not good
        No weight when ER 2 is in comparison

    5. **ER 3**
        2 rhats at 1.02
        Good ELPD 
        OH binds too strongly
        COOH coverage too high 
        OH order is not good 

    6. LH 4 (remove from comparison)
        Converged
        Trash fit (All KOs are constants)

    ER 2 is the best case. 
    ER 3 vs LH 3 is an interesting case - they effectively provide the same information. The posteriors also collapse into the same values. Given this, **I am neglecting LH 3**. But LH can't be completely neglected - it has been reported as the RDS in several papers. 
    ER 2, ER 3 and LH 3 all have indistinguishably good ELPDs (but chemical intuition tells us ER 2 is the correct model). 
    All models have bad OH orders. 

# 2. Composite models 

    1. **ER LH 2**
        Converged 
        Good ELPD
        Comparable G_act 
        OH order is not good

    2. **ER 2 3**
        1 shape factor > 0.7 
        Good ELPD
        G_act comparable at low potentials
        COOH can have high coverage at low potentials
        OH order is not good 

    3. LH 2 3 
        p_loo too high, multimodal fit 
        Bad ELPD
        Resolution is not easy - G_act for 3 > 2
        COOH coverage too high
        OH order is not good 

    ER LH 2 and ER 2 3 are indistinguishable. 
    Both have bad OH orders. 

# 3. Complex models 

    1. **ER LH 2 3** 
        Converged
        Good ELPD 
        G_act for ER and 3 comparable at low potentials 
        G_act for ER and LH comparable 
        COOH can have high coverage at low potentials
        OH order is not good 
        TODO: Calculate DRC of all steps (see LH vs ER)

    2. **ER 1 2 3** 
        Converged 
        Perfect ELPD
        Comparable G_act 
        CO adsorption rate limiting at high potentials 
        COOH can have high coverage at low potentials
        Great OH order

    3. ER LH 2 3 4 
        1 shape factor > 0.7
        Good ELPD
        Comparable G_act for ER and LH
        COOH can have high coverage at low potentials
        OH ads rate is never really limiing (3-10 times faster)
        OH order is not good 
    
    4. Full (ER LH 1 2 3 4)
        Priors from ER LH 2 3 4 fail
        Priors need to be from ER 1 2 3
        Converged
        Perfect ELPD
        COOH can have high coverage at low potentials
        Great OH order

# 4. Lateral interaction models 

    1. ER LH 2 Lateral
        Converged + negligible convergence errors
        Perfect ELPD 
        Comparable G_act
        CO and OH lateral interactions 
        CO coverage linearly decreases
        OH coverage linearly increases
        No COOH 
        KO fitting is not that great

    2. ER 2 3 Lateral 
        2 rhats at 1.01
        Good ELPD 
        CO and OH lateral interactions
        Low COOH coverage 
        G_act ER > G_act 3
        OH order is not good 
        CO coverage linearly decreases
        OH coverage linearly increases

    3. Full Lateral (ER LH 2 3 Lateral)
        6 rhats at 1.01
        Excellent ELPD
        Only OH lateral interactions
        Comparable G_act 
        High COOH coverage at low potentials
        OH coverage linearly increases
        KO fitting does not look that great 

GM Notes: 
1. What about this 
2. 