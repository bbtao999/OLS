Economists or statisticians may refer to different methods when approaching the same question. However, the core of the solutions may essentially be the same.
This file uses empirical examples to show two points: 

1/ the OLS (with covariate on the RHS) is equivalent to CUPED method. 

2/ with completely random experiment (i.e. the treatment status is independent of the covariate),
controlling for pre-treatment outcomes (or any relevant covariate) does not create significant change in the average treatment effects. 
That is, we can use simply use the cross-sectional estimate instead of the DID set up to back out the average treatment effect. 

3/ the delta method (for estimating the ratio (e.g. click-through rate)) generates similar estimate of the variance as the OLS coeffcient se of the treatment. In order words, the Var (z1_bar-z2_bar) is equivalent to se(D), where z_bar=the sample click-through rate of the treatment or control group.
