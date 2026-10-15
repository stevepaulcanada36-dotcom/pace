# PaceSense study report

71 completed coaches, 284 coach x runner observations.

## Observations (cf. Table 1)
| condition | Familiar runner | Unfamiliar runner | Total |
|---|---|---|---|
| Interactive | 52 | 108 | 160 |
| Non-interactive | 40 | 84 | 124 |
| Total | 92 | 192 | 284 |

## RQ1-RQ3: Acceptance
Mean acceptance by condition and runner familiarity:

| condition | Familiar runner | Unfamiliar runner |
|---|---|---|
| Interactive | 0.71 | 0.73 |
| Non-interactive | 0.55 | 0.58 |

Acceptance, interactive vs non-interactive (Welch): interactive 0.72 vs non-interactive 0.57; t(241.1) = 2.87, p = 0.005, n = 284

Multilevel model (cf. Table 2, Acceptance column):
```
Linear mixed model, random intercept per session_id (REML): n = 284, groups = 71, marginal R2 = 0.070, group variance = 0.0000, residual variance = 0.1844
                         coef    se      z     p  sig
Intercept               0.920 0.217  4.238 0.000  ***
interactive             0.133 0.066  2.026 0.043    *
familiar               -0.035 0.084 -0.416 0.677     
interactive_x_familiar  0.020 0.113  0.176 0.860     
seq                     0.005 0.023  0.227 0.821     
comp_eff               -0.044 0.019 -2.332 0.020    *
se_finish              -0.002 0.022 -0.077 0.939     
coaching_experience    -0.017 0.028 -0.606 0.544     
data_exp               -0.021 0.022 -0.936 0.349     
female                 -0.044 0.057 -0.759 0.448     
edu_vocational          0.051 0.078  0.656 0.512     
edu_highschool          0.102 0.090  1.129 0.259     
edu_graduate            0.097 0.066  1.478 0.139     
```

## RQ1: Trust (coach level, OLS)
Cronbach's alpha: perceived competence 0.913, willingness to depend 0.960

Perceived competence:
```
OLS: n = 71, adj. R2 = 0.365
                      coef    se      t     p  sig
Intercept            3.565 0.785  4.543 0.000  ***
interactive          0.919 0.209  4.388 0.000  ***
comp_eff             0.153 0.071  2.161 0.035    *
se_finish           -0.126 0.083 -1.515 0.135     
coaching_experience  0.053 0.105  0.507 0.614     
data_exp             0.184 0.084  2.189 0.032    *
female              -0.040 0.215 -0.187 0.853     
edu_vocational      -1.028 0.292 -3.516 0.001  ***
edu_highschool      -0.690 0.337 -2.046 0.045    *
edu_graduate        -0.438 0.247 -1.774 0.081     
```

Willingness to depend:
```
OLS: n = 71, adj. R2 = -0.014
                      coef    se      t     p sig
Intercept            2.779 0.977  2.845 0.006  **
interactive          0.146 0.261  0.561 0.577    
comp_eff             0.087 0.088  0.989 0.327    
se_finish           -0.019 0.104 -0.186 0.853    
coaching_experience -0.082 0.130 -0.633 0.529    
data_exp             0.117 0.105  1.113 0.270    
female               0.282 0.267  1.055 0.295    
edu_vocational       0.079 0.364  0.217 0.829    
edu_highschool       0.533 0.420  1.269 0.209    
edu_graduate         0.619 0.308  2.011 0.049   *
```

## Model adaptation and acceptance (cf. Table 4)
| class | runners | mean_acceptance |
|---|---|---|
| 0. Non-interactive condition | 124 | 0.57 |
| 1. No adaptation (<= 0.25) | 73 | 0.67 |
| 2. Some adaptation (0.25-0.75) | 40 | 0.72 |
| 3. Full adaptation (0.75-1.25) | 31 | 0.94 |
| 4. Over adaptation (> 1.25) | 16 | 0.56 |

## RQ4: Model accuracy (unfamiliar runners, interactive condition)
initial model error % 2.60 vs adapted model error % 2.62; t(107.0) = -0.09, p = 0.930, n = 108

## Coaches' own accuracy (unfamiliar runners, both conditions)
coach before % 4.41 vs coach after % 2.84; t(191.0) = 6.86, p < .001, n = 192

## Slider interactions per runner position (interactive, cf. Sec. 5.3)
| seq | mean | median | max |
|---|---|---|---|
| 1 | 4.8 | 4.0 | 11 |
| 2 | 5.0 | 4.0 | 11 |
| 3 | 4.7 | 4.0 | 11 |
| 4 | 5.1 | 4.0 | 10 |

Number of interactions ~ runner sequence (mixed model):
```
Linear mixed model, random intercept per session_id (REML): n = 160, groups = 40, marginal R2 = 0.066, group variance = 0.5589, residual variance = 6.2745
                      coef    se      z     p sig
Intercept            2.045 2.127  0.961 0.336    
seq                  0.119 0.179  0.662 0.508    
familiar             0.937 0.452  2.074 0.038   *
comp_eff             0.220 0.179  1.225 0.220    
se_finish            0.002 0.239  0.006 0.995    
coaching_experience -0.211 0.276 -0.766 0.443    
data_exp             0.172 0.211  0.816 0.415    
female               0.678 0.510  1.328 0.184    
edu_vocational       0.041 0.854  0.049 0.961    
edu_highschool      -0.480 0.746 -0.644 0.520    
edu_graduate        -0.124 0.629 -0.197 0.844    
```

## Acceptance by adaptation class (cf. Sec. 5.7; baseline: non-interactive)
```
Linear mixed model, random intercept per session_id (REML): n = 284, groups = 71, marginal R2 = 0.067, group variance = 0.0041, residual variance = 0.1761
            coef    se      z     p  sig
Intercept  0.570 0.039 14.468 0.000  ***
class_1    0.094 0.064  1.460 0.144     
class_2    0.150 0.078  1.911 0.056     
class_3    0.368 0.086  4.272 0.000  ***
class_4   -0.007 0.113 -0.058 0.954     
```

## Slider position ~ year of the race (cf. Sec. 5.8)
```
OLS: n = 573, adj. R2 = 0.039
             coef    se      t     p  sig
Intercept -36.027 7.412 -4.861 0.000  ***
year        0.018 0.004  4.926 0.000  ***
```

## Acceptance ~ runner motivation, familiar runners (cf. Sec. 5.10)
_One mixed model per motivation (indicator vs. not indicated), random intercept per coach._

| motivation | runners | coef | se | p |
|---|---|---|---|---|
| general health | 22 | 0.060 | 0.107 | 0.576 |
| weight concern | 12 | -0.159 | 0.134 | 0.237 |
| affiliation | 16 | -0.013 | 0.120 | 0.916 |
| recognition | 17 | 0.052 | 0.118 | 0.657 |
| psychological coping | 24 | 0.087 | 0.104 | 0.403 |
| life meaning | 23 | 0.006 | 0.106 | 0.954 |
| self esteem | 21 | -0.201 | 0.106 | 0.059 |
| competition | 27 | 0.053 | 0.100 | 0.597 |
| personal goal achievement | 22 | 0.052 | 0.107 | 0.625 |

## Robustness: binary acceptance (rounded at .5), same predictors (cf. Table 2 note)
_Linear probability mixed model; the paper used a multilevel logistic model._
```
Linear mixed model, random intercept per session_id (REML): n = 284, groups = 71, marginal R2 = 0.057, group variance = 0.0050, residual variance = 0.2182
                         coef    se      z     p  sig
Intercept               0.983 0.246  3.997 0.000  ***
interactive             0.133 0.074  1.798 0.072     
familiar                0.013 0.093  0.144 0.885     
interactive_x_familiar  0.016 0.124  0.127 0.899     
seq                     0.004 0.025  0.152 0.879     
comp_eff               -0.041 0.022 -1.892 0.059     
se_finish              -0.007 0.025 -0.260 0.795     
coaching_experience    -0.013 0.032 -0.424 0.671     
data_exp               -0.034 0.025 -1.345 0.179     
female                 -0.068 0.065 -1.049 0.294     
edu_vocational          0.013 0.088  0.151 0.880     
edu_highschool          0.013 0.102  0.127 0.899     
edu_graduate            0.079 0.075  1.054 0.292     
```
