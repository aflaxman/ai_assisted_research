# Verification runs of the authors' original supplementary code

Code: original/Ranavirus(Jul-13-26).py, modified only to take pd and pond count
from the command line and run headless. Seed 28473892 as shipped, 10,000 ponds,
120 iterations.

## pd = 0.15
```
Prob of death = 0.15 using 120 iterations, and 10000 ponds.
 Average time to all infected is  10.0222 iterations.
 Average time to all dead is  37.64055682526491 iterations, out of  9626  ponds.
{"com_peak": 36.8474, "com_peak_at": 10, "ulc_peak": 13.03, "hem_peak": 4.9572, "com_max_over_ponds": 65, "all_infected": 10.0222, "all_dead": 37.64055682526491, "dead_ponds": 9626}
```

## pd = 0.25
```
Prob of death = 0.25 using 120 iterations, and 10000 ponds.
 Average time to all infected is  8.8368 iterations.
 Average time to all dead is  24.75514993481095 iterations, out of  7670  ponds.
{"com_peak": 14.2938, "com_peak_at": 9, "ulc_peak": 12.867, "hem_peak": 5.43, "com_max_over_ponds": 45, "all_infected": 8.8368, "all_dead": 24.75514993481095, "dead_ponds": 7670}
```

## pd = 0.175
```
Prob of death = 0.175 using 120 iterations, and 10000 ponds.
 Average time to all infected is  9.9696 iterations.
 Average time to all dead is  32.78661400512382 iterations, out of  9368  ponds.
{"com_peak": 30.1285, "com_peak_at": 9, "ulc_peak": 13.2763, "hem_peak": 5.1277, "com_max_over_ponds": 58, "all_infected": 9.9696, "all_dead": 32.78661400512382, "dead_ponds": 9368}
```

## pd = 0.06
```
Prob of death = 0.06 using 120 iterations, and 10000 ponds.
 Average time to all infected is  9.7991 iterations.
 Average time to all dead is  85.13968837913872 iterations, out of  9242  ponds.
{"com_peak": 69.265, "com_peak_at": 10, "ulc_peak": 11.1436, "hem_peak": 4.15, "com_max_over_ponds": 88, "all_infected": 9.7991, "all_dead": 85.13968837913872, "dead_ponds": 9242}
```

