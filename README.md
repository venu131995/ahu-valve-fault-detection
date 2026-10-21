# Valve and Damper Faults in an Air-Handling Unit: What a Building System Sees, What a Smart Valve Adds, What It Costs

Leaking and stuck coil valves and stuck dampers waste energy silently. Which of them can be found from the points a
building automation system already logs, how much does valve position feedback add, and what does each fault cost?

**Data.** LBNL Fault Detection and Diagnostics Data Sets, single-duct AHU (Granderson et al. 2022,
doi 10.25984/1881324): one simulated year per case at 1-minute resolution (Chicago TMY weather, DOE large-office
reference building), a fault-free year and 20 faulted years, 30 points each.

## In short

| Question | Answer |
|---|---|
| Is the public data set usable as is? | **Partly.** The four "coil valve leaking 10/25/40/50 %" files are byte-identical (only 10 % is real), and so are the four outdoor-air sensor bias files, which also show **no bias** at all. Outdoor airflow is a constant (357,730), and the static-pressure set point is swapped or garbage (-400) in every fault file. The "coi_bias" files do not match the documented supply-air bias. 9 of the 21 cases are used. |
| Stuck coil valve | Found from building-system points alone on **82-100 %** of days (supply air too warm with the valve commanded fully open, or cooling while commanded closed); position feedback lifts the 10 % and 25 % cases to 100 %. |
| Leaking coil valve (10 %) | **Invisible** to the building-system points (1 % of days, at the false-alarm level) but flagged on **40 %** of days when the valve reports its position. |
| Stuck outdoor-air damper | Mixed-air balance inconsistent with the damper command: **48 %** of days when stuck near closed (the economiser days, when it should open), **97 %** when stuck at 75 %. |
| What do they cost? | Damper stuck at 10-25 %: **+52-56 % cooling coil load** (lost free cooling). Coil valve stuck at 75 %: **+48 %** coil load (overcooling). Valve stuck at 10-25 %: coil load -44 to -58 % but **fan energy +12-25 %** and supply air too warm (comfort lost). Damper stuck at 75 %: fan energy +18 %. |
| False alarms | 3 of 151 fault-free days (July-December, model fitted on January-June). |

The case for valve feedback is clearest for the leak: a valve that does not fully close looks normal to the building
system, but its position signal reveals it at once.

## Method

All rules are learned from the fault-free year (January-June) and applied to July-December of every year (the
damper-stuck-100 % file covers April-November only and is used whole). Occupied hours only, hourly means, a day is
flagged when at least two hours are.

* **Mixing balance** (damper faults): the outdoor-air fraction (MA - RA) / (OA - RA), when OA and RA differ by more
  than 5 F, compared with its fault-free median for the same damper command; flag if they differ by more than 0.15.
* **Coil, commanded closed** (leak, stuck open): temperature drop MA - SA above the fault-free 99th percentile + 1 F
  (fan heat makes it negative when the valve is shut).
* **Coil, commanded fully open** (stuck closed): supply air more than 2 F above its set point.
* **Valve feedback**: |position - command| > 0.05.
* **Energy**: fan power summed over the year; coil load = supply airflow x (MA - SA), sensible part, relative units.

## Limitations

* Simulated data from one building and one climate; real buildings add sensor noise and drift, and their faults
  come and go.
* Several cases of the public set are defective (above), so leak severity cannot be studied beyond 10 %.
* Airflow columns are in unclear units, so energy is reported as relative change, not kWh.
* Detection only (which rule fires names the fault family); no probabilistic diagnosis or isolation between
  coexisting faults.

## Run

    pip install numpy pandas matplotlib
    # data: https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_SDAHU/LBNL_FDD_Data_Sets_SDAHU.zip -> data/
    python scripts/run_all.py

Results: `results/summary.json`. Figure: `figures/detection_energy.png`.
